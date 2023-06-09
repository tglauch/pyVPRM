import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'lib'))
from sat_manager import VIIRS, sentinel2, modis, earthdata,\
                        copernicus_land_cover_map, satellite_data_manager
from VPRM import vprm 
import glob
import time
import numpy as np
import xarray as xr
import argparse

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--config", type=str)
p.add_argument("--n_cpus", type=int, default=1)
args = p.parse_args()
print(args)

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def add_land_cover_map(vprm_inst, land_cover_on_modis_grid=None, copernicus_data_path=None,
                       save_path=None):
    # If the land-cover-map on the modis grid is pre-calculated
    
    if land_cover_on_modis_grid is not None:
        vprm_inst.add_land_cover_map(land_cover_on_modis_grid)

    # if vprm_inst not pre-calculated, load the copernicus land cover tiles from the copernicus_data_path.
    # Download needs to be done manually from here: https://lcviewer.vito.be/download
    # If the land-cover-map on the modis grid needs to be calculated on the fly
    # for checks interactive viewer can be useful https://lcviewer.vito.be/2019
    
    if copernicus_data_path is not None:
        tiles_to_add = []
        for i, c in enumerate(glob.glob(os.path.join(copernicus_data_path, '*'))):
            print(c)
            if i == 0:
                handler_lt = copernicus_land_cover_map(c)
                handler_lt.load()
            else:
                lcm=copernicus_land_cover_map(c)
                lcm.load()
                tiles_to_add.append(lcm)
        handler_lt.add_tile(tiles_to_add, reproject=False)
        vprm_inst.add_land_cover_map(handler_lt, save_path=save_path)
        return
    
lons = np.linspace(cfg['lon_min'], cfg['lon_max'] , cfg['n_bins_lon']) 
lats = np.linspace(cfg['lat_min'], cfg['lat_max'], cfg['n_bins_lat'])
hvs =  cfg['hvs']:
vprm_inst = vprm(n_cpus=args.n_cpus)
file_collections = np.unique([i.split('.')[1] for i in
                              glob.glob(os.path.join(cfg['sat_image_path'],
                                        '*h{:02d}v{:02d}*.h*'.format(hvs[0][0], hvs[0][1])))])
for c0, f in enumerate(sorted(file_collections)):
    print(c0)
    handlers = []
    if cfg['satellite'] == 'modis':
        for c, i in sorted(enumerate(glob.glob(os.path.join(cfg['sat_image_path'], '*{}*.h*'.format(f))))):
            if c == 0:
                handler = modis(sat_image_path=i)
                handler.load()
            else:
                handler2 = modis(sat_image_path=i)
                handler2.load()
                handlers.append(handler2)
    elif cfg['satellite'] == 'viirs':
        for c, i in sorted(enumerate(glob.glob(os.path.join(cfg['sat_image_path'], '*{}*.h*'.format(f))))):
            if c == 0:
                handler = VIIRS(sat_image_path=i)
                handler.load()
            else:
                handler2 = VIIRS(sat_image_path=i)
                handler2.load()
                handlers.append(handler2)
    else:
        print('Set the satellite in the cfg either to modis or viirs.')
    handler.add_tile(handlers, reproject=False)
    if cfg['satellite'] == 'modis':
        vprm_inst.add_sat_img(handler, b_nir='B02', b_red='B01',
                              b_blue='B03', b_swir='B06',
                             drop_bands=True) 
    elif cfg['satellite'] == 'viirs':
        vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                              b_blue='no_blue_sensor', b_swir='SurfReflect_I3',
                              which_evi='evi2',
                              drop_bands=True) 
       
vprm_inst.sort_and_merge_by_timestamp()

if os.path.exists(os.path.join(cfg['out_path'], 'veg_map_on_modis_grid.nc')):
    add_land_cover_map(vprm_inst,
                       land_cover_on_modis_grid=os.path.join(cfg['out_path'],
                                                             'veg_map_on_modis_grid.nc'))
else:    
    add_land_cover_map(vprm_inst,
                       copernicus_data_path=cfg['copernicus_path'],
                       save_path=os.path.join(cfg['out_path'],
                                              'veg_map_on_modis_grid.nc'))
 

vprm_inst.lowess(n_cpus=arg.n_cpus)

out_grid = dict()
out_grid['lons'] = lons
out_grid['lats'] = lats
if os.path.exists(os.path.join(cfg['out_path'], 'regridder.nc')):
    wrf_op = vprm_inst.to_wrf_output(out_grid, weights_for_regridder=os.path.join(cfg['out_path'], 'regridder.nc'))
else:
    wrf_op = vprm_inst.to_wrf_output(out_grid, driver = 'ESMF_RegridWeightGen', n_cpus=120, 
                                     regridder_save_path=os.path.join(cfg['out_path'], 'regridder.nc'))


file_base = 'VPRM_input_'
filename_dict = {'lswi': 'LSWI', 'evi': 'EVI', 'veg_fraction': 'VEG_FRA',
                 'lswi_max': 'LSWI_MAX', 'lswi_min': 'LSWI_MIN', 
                 'evi_max': 'EVI_MAX', 'evi_min': 'EVI_MIN'} 
for key in wrf_op.keys():
    wrf_op[key].to_netcdf(os.path.join(cfg['out_path'],file_base + filename_dict[key] +'_{}.nc'.format(cfg['year'])))
