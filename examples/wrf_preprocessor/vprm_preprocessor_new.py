import sys
import os
import pathlib
this_file = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(this_file, '..'))
from lib.sat_manager import VIIRS, sentinel2, modis, earthdata,\
                        copernicus_land_cover_map, satellite_data_manager
from lib.functions import lat_lon_to_modis, add_corners_to_1d_grid, parse_wrf_grid_file
from VPRM import vprm
import yaml
import glob
import time
import numpy as np
import xarray as xr
import argparse
from shapely.geometry import box, Polygon
import geopandas as gpd
from datetime import datetime, timedelta
from pyproj import Transformer

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
p.add_argument("--year", type=int)
p.add_argument("--n_cpus", type=int, default=1)
p.add_argument("--chunk_x", type=int, default=1)
p.add_argument("--chunk_y", type=int, default=1)

args = p.parse_args()
print(args)

this_year = int(args.year)
with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

out_grid = parse_wrf_grid_file(cfg['geo_em_file'])

hvs = np.unique([lat_lon_to_modis(out_grid['lat_b'].values.flatten()[i], 
                                  out_grid['lon_b'].values.flatten()[i]) 
                for i in range(len(out_grid['lat_b'].values.flatten()))],
                  axis=0)
print(hvs)
insts = []

#Load the data

days =  [datetime(this_year, 1, 1)+ timedelta(days=i) for i in np.arange(365.)]

for c, i in enumerate(hvs):
    
    print(i)
    file_collection_before = sorted([f for f in glob.glob(os.path.join(cfg['sat_image_path'], str(this_year-1),
                                                               '*h{:02d}v{:02d}*.hdf'.format(i[0], i[1]))) if '.xml' not in f])[-3:]
    file_collection_this = sorted([f for f in glob.glob(os.path.join(cfg['sat_image_path'], str(this_year),
                                                                      '*h{:02d}v{:02d}*.hdf'.format(i[0], i[1]))) if '.xml' not in f])
    file_collection_after = sorted([f for f in glob.glob(os.path.join(cfg['sat_image_path'], str(this_year+1),
                                                                       '*h{:02d}v{:02d}*.hdf'.format(i[0], i[1]))) if '.xml' not in f])[:3]
    file_collections = np.concatenate([file_collection_before, file_collection_this,
                                       file_collection_after])
    
    if len(file_collections) == 0:
        continue

    new_inst = vprm(vprm_config_path = os.path.join(this_file, '..', 'vprm_configs',
                                                    'copernicus_land_cover.yaml') ,
                    n_cpus=args.n_cpus)
    for c0, fpath in enumerate(file_collections):
        if cfg['satellite'] == 'modis':
            print(fpath)
            handler = modis(sat_image_path=fpath)
            handler.load() 
        elif cfg['satellite'] == 'viirs':
            print(fpath)
            handler = VIIRS(sat_image_path=fpath)
            handler.load()
        else:
            print('Set the satellite in the cfg either to modis or viirs.')
            
        if c0 == 0:
            trans = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                           handler.sat_img.rio.crs)

            # To save memory crop satellite images to WRF grid
            x_a, y_a = trans.transform(out_grid['lon_b'], out_grid['lat_b'])
            b = box(float(np.min(x_a)), float(np.min(y_a)),
                    float(np.max(x_a)), float(np.max(y_a)))
            b = gpd.GeoSeries(Polygon(b), crs=handler.sat_img.rio.crs)

        handler.crop_box(b)

        if cfg['satellite'] == 'modis':
            new_inst.add_sat_img(handler, b_nir='sur_refl_b02', b_red='sur_refl_b01',
                                  b_blue='sur_refl_b03', b_swir='sur_refl_b06',
                                  which_evi='evi',
                                  drop_bands=True,
                                  timestamp_key='sur_refl_day_of_year',
                                  mask_bad_pixels=True,
                                  mask_clouds=True) 
        elif cfg['satellite'] == 'viirs':
            new_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                                  b_blue='no_blue_sensor', b_swir='SurfReflect_I3',
                                  which_evi='evi2',
                                  drop_bands=True)
           
    # Sort and merge satellite images
    new_inst.sort_and_merge_by_timestamp()
    
    # Apply lowess smoothing
    new_inst.lowess(keys=['evi', 'lswi'],
                   times=days,
                   frac=0.25, it=3) #0.2

    new_inst.clip_values('evi', 0, 1)
    new_inst.clip_values('lswi',-1, 1)
    new_inst.sat_imgs.sat_img = new_inst.sat_imgs.sat_img[['evi', 'lswi']]

    insts.append(new_inst)

insts = np.array(insts)
vprm_inst = insts[0]
if len(insts) > 1:
    vprm_inst.add_vprm_insts(insts[1:])
    
print(vprm_inst.sat_imgs.sat_img)

# Add the land cover map
if not os.path.exists(cfg['out_path']):
    os.makedirs(cfg['out_path'])
veg_file = os.path.join(cfg['out_path'], 'veg_map_on_modis_grid_{}_{}.nc'.format(args.chunk_x,
                                                                                 args.chunk_y))
# if os.path.exists(veg_file):
#     print('Load land cover map')
#     add_land_cover_map(vprm_inst,
#                        land_cover_on_modis_grid=veg_file)
# else:   
print('Generate land cover map')
regridder_path = os.path.join(cfg['out_path'], 'regridder_{}_{}_lcm.nc'.format(args.chunk_x,
                                                                               args.chunk_y))
if os.path.exists(veg_file):
    vprm_inst.add_land_cover_map(veg_file)

    
# if vprm_inst not pre-calculated, load the copernicus land cover tiles from the copernicus_data_path.
# Download needs to be done manually from here: https://lcviewer.vito.be/download
# If the land-cover-map on the modis grid needs to be calculated on the fly
# for checks interactive viewer can be useful https://lcviewer.vito.be/2019

handler_lt = None
copernicus_data_path = cfg['copernicus_path']

if copernicus_data_path is not None:
    tiles_to_add = []
    for i, c in enumerate(glob.glob(os.path.join(copernicus_data_path, '*'))):
        print(c)
        temp_map = copernicus_land_cover_map(c)
        temp_map.load()
        dj = vprm_inst.is_disjoint(temp_map)
        if dj:
            continue
        print('Add {}'.format(c))
        if handler_lt is None:
            handler_lt = temp_map
        else:
            tiles_to_add.append(temp_map)
            
    handler_lt.add_tile(tiles_to_add, reproject=False)
    
    trans = Transformer.from_crs(vprm_inst.sat_imgs.sat_img.rio.crs,
                                  handler_lt.sat_img.rio.crs,
                                  always_xy=True)
    xs = add_corners_to_1d_grid(vprm_inst.sat_imgs.sat_img.coords['x'])
    ys = add_corners_to_1d_grid(vprm_inst.sat_imgs.sat_img.coords['y'])
    xs, ys = np.meshgrid(xs,ys)
    x_a, y_a = trans.transform(xs, ys)
    b = box(float(np.min(x_a)), float(np.min(y_a)),
            float(np.max(x_a)), float(np.max(y_a)))
    b = gpd.GeoSeries(Polygon(b), crs=handler_lt.sat_img.rio.crs)
    handler_lt.crop_box(b)
    vprm_inst.add_land_cover_map(handler_lt, save_path=veg_file,
                                 regridder_save_path=regridder_path)
    del handler_lt

regridder_path = os.path.join(cfg['out_path'], 'regridder_{}_{}.nc'.format(args.chunk_x,
                                                                            args.chunk_y))
# if os.path.exists(regridder_path):
#     print('Use existing regridder')
#     wrf_op = vprm_inst.to_wrf_output(out_grid, weights_for_regridder=regridder_path)
# else:
print('Create regridder')
wrf_op = vprm_inst.to_wrf_output(out_grid, driver = 'xEMSF', #'ESMF_RegridWeightGen', 
                                 regridder_save_path=regridder_path)

# Save to NetCDF files
file_base = 'VPRM_input_'
filename_dict = {'lswi': 'LSWI', 'evi': 'EVI', 'veg_fraction': 'VEG_FRA',
                 'lswi_max': 'LSWI_MAX', 'lswi_min': 'LSWI_MIN', 
                 'evi_max': 'EVI_MAX', 'evi_min': 'EVI_MIN'} 
for key in wrf_op.keys():
    ofile = os.path.join(cfg['out_path'],file_base + filename_dict[key] +'_{}_part_{}_{}.nc'.format(this_year,
                                                                                               args.chunk_x,
                                                                                               args.chunk_y))
    if os.path.exists(ofile):
        os.remove(ofile)
    if ('lswi' in key) | ('evi' in key):
        t = wrf_op[key][key].loc[{'vprm_classes': 8}].values
        t[~np.isfinite(t)] = 0
        wrf_op[key][key].loc[{'vprm_classes': 8}] = t
    wrf_op[key].to_netcdf(ofile)