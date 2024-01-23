import os
from pyVPRM.sat_managers.viirs import VIIRS
from pyVPRM.sat_managers.modis import modis
from pyVPRM.sat_managers.copernicus import copernicus_land_cover_map
from pyVPRM.lib.functions import lat_lon_to_modis, add_corners_to_1d_grid, parse_wrf_grid_file
from pyVPRM.VPRM import vprm
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

# Read command line arguments
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

#Load config file
this_year = int(args.year)
with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Use WRF netcdf to generate an xarray instance. With chunk_x and chunk_y 
# the output grid can be calculated in disjoint peaces

out_grid = parse_wrf_grid_file(cfg['geo_em_file'], n_chunks=cfg['n_chunks'],
                              chunk_x=args.chunk_x, chunk_y=args.chunk_y)

# Calculate the required MODIS tiles for our out grid
hvs = np.unique([lat_lon_to_modis(out_grid['lat_b'].values.flatten()[i], 
                                  out_grid['lon_b'].values.flatten()[i]) 
                for i in range(len(out_grid['lat_b'].values.flatten()))],
                  axis=0)
print(hvs)
insts = []

#Load the data

# days of interest
days =  [datetime(this_year, 1, 1)+ timedelta(days=i) for i in np.arange(365.)]

# read the data
for c, i in enumerate(hvs):
    
    print(i)
    file_collections = glob.glob(os.path.join(cfg['sat_image_path'], 
                                                 '*h{:02d}v{:02d}*.nc'.format(i[0], i[1])))
    
    if len(file_collections) == 0:
        continue

    new_inst = vprm(vprm_config_path = os.path.join(this_file, '..', '..',
                                                    'pyVPRM', 'vprm_configs',
                                                    'copernicus_land_cover.yaml') ,
                    n_cpus=args.n_cpus)
    for c0, fpath in enumerate(file_collections):
        print(fpath)
        if cfg['satellite'] == 'modis':
            handler = modis(sat_image_path=fpath)
            handler.load() 
        elif cfg['satellite'] == 'viirs':
            handler = VIIRS(sat_image_path=fpath)
            handler.load()
        else:
            print('Set the satellite in the cfg either to modis or viirs.')

        # In order to save memory crop the satellite images to our WRF out grid  
        if c0 == 0:
            trans = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                           handler.sat_img.rio.crs)

            # To save memory crop satellite images to WRF grid
            x_a, y_a = trans.transform(out_grid['lon_b'], out_grid['lat_b'])
            b = box(float(np.min(x_a)), float(np.min(y_a)),
                    float(np.max(x_a)), float(np.max(y_a)))
            b = gpd.GeoSeries(Polygon(b), crs=handler.sat_img.rio.crs)
            b = b.scale(1.2, 1.2)

        handler.crop_box(b)

        # Add satellite image to VPRM instance
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
    
    # Apply lowess smoothing in time
    new_inst.lowess(keys=['evi', 'lswi'],
                   times=days,
                   frac=0.25, it=3) #0.2

    # Clip EVI and LSWI to the allowed range
    new_inst.clip_values('evi', 0, 1)
    new_inst.clip_values('lswi',-1, 1)
    new_inst.sat_imgs.sat_img = new_inst.sat_imgs.sat_img[['evi', 'lswi']]

    insts.append(new_inst)

# Merge the VPRM instances for the different tiles
insts = np.array(insts)
vprm_inst = insts[0]
if len(insts) > 1:
    vprm_inst.add_vprm_insts(insts[1:])
    
print(vprm_inst.sat_imgs.sat_img)

# Add the land cover map
if not os.path.exists(cfg['out_path']):
    os.makedirs(cfg['out_path'])

# Add the land cover map and perform regridding to the satellite grid
print('Generate land cover map')
veg_file = os.path.join(cfg['out_path'], 'veg_map_on_modis_grid_{}_{}.nc'.format(args.chunk_x,
                                                                                 args.chunk_y))
if os.path.exists(veg_file):
    vprm_inst.add_land_cover_map(veg_file)
else:
    regridder_path = os.path.join(cfg['out_path'], 'regridder_{}_{}_lcm.nc'.format(args.chunk_x,
                                                                                   args.chunk_y))
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
            
        # Crop the land cover map to the extend of our (cropped) satellite images to save memory
        geom = box(*vprm_inst.sat_imgs.sat_img.rio.bounds())
        df = gpd.GeoDataFrame({"id":1,"geometry":[geom]})
        df = df.set_crs(vprm_inst.sat_imgs.sat_img.rio.crs)
        df = df.scale(1.2, 1.2)
        handler_lt.crop_to_polygon(df) 
        vprm_inst.add_land_cover_map(handler_lt, save_path=veg_file,
                                     regridder_save_path=regridder_path)

regridder_path = os.path.join(cfg['out_path'], 'regridder_{}_{}.nc'.format(args.chunk_x,
                                                                            args.chunk_y))

# Use all the information in the VPRM instance to generate the WRF input files
print('Create regridder')
wrf_op = vprm_inst.to_wrf_output(out_grid, driver = 'xEMSF', # currently only xESMF works here when the WRF grid is 2D 
                                 regridder_save_path=regridder_path,
                                 mpi=False)

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
