import sys
import os
sys.path.append('/home/b/b309233/software/CO2KI/VPRM/')
sys.path.append('/home/b/b309233/software/SatManager/')
sys.path.append('/home/b/b309233/software/CO2KI/harvard_forest/')
import yaml 
from datetime import date
from sat_manager import VIIRS, sentinel2, modis, earthdata,\
                        copernicus_land_cover_map, satellite_data_manager
from VPRM import vprm 

import glob
import time
import yaml
with open("logins.yaml", "r") as stream:
    try:
        logins = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
import numpy as np
import xarray as xr


def add_land_cover_map(vprm_inst, land_cover_on_modis_grid=None, copernicus_data_path=None,
                       save_path=None):
    # If the land-cover-map on the modis grid is pre-calculated
    
    if land_cover_on_modis_grid is not None:
        vprm_inst.add_land_cover_map(land_cover_on_modis_grid) #'/work/bd1231/tglauch/Regridding/vprm_on_modis_grid.nc' ) 

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
    
year = 2015
sat_data_bpath = '/work/bd1231/tglauch/one_year_viirs_europe_new_{}'.format(year)
copernicus_data_path = '/work/bd1231/tglauch/copernicus_classification'
out_base = '/work/bd1231/tglauch/VPRM_input_viirs/'

# Define output grid
n_bins_lon = 400
n_bins_lat= 480
lo_min, lo_max = -15, 34.875
la_min, la_max = 33, 72.916664
lons = np.linspace(lo_min , lo_max , n_bins_lon) 
lats = np.linspace(la_min, la_max, n_bins_lat)

#If already downloaded, we can simply set the path to the satellite files, otherwise download and save first...

#File with earthdata login information
#format 

#    modis:
#        - username
#        - pwd
'''
with open("logins.yaml", "r") as stream:
    try:
        logins = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#tiles for europe
hv = [(17,2), (17,3), (17,4), (17,5), (18,2), (18,3), (18,4),
      (18,5), (19,2), (19,3), (19,4), (19,5), (20,2), (20,3), 
       (20,4), (20,5), (17,1), (18,1), (19,1), (20,1)]

for i in hv:
    print(i)
    handler = modis()
    handler.download(date(year, 1, 1),
                    savepath = sat_data_bpath,
                    username = logins['modis'][0],
                    pwd = logins['modis'][1],
                    hv=i,
                    delta = 1,
                    enddate=date(year+1, 1, 1))
'''

vprm_inst = vprm()
file_collections = sorted(np.unique([i.split('.')[1] for i in glob.glob(os.path.join(sat_data_bpath, '*h{:02d}v{:02d}*.h*'.format(18, 3)))]))
for c0, f in enumerate(file_collections):
    print(c0)
    handlers = []
    for c, i in sorted(enumerate(glob.glob(os.path.join(sat_data_bpath, '*{}*.h*'.format(f))))):
        if c == 0:
            handler = VIIRS(sat_image_path=i)
            handler.load()
        else:
            handler2 = VIIRS(sat_image_path=i)
            handler2.load()
            handlers.append(handler2)
    handler.add_tile(handlers, reproject=False)
    vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                          b_blue='no_blue_sensor', b_swir='SurfReflect_I3', smearing=False,
                          which_evi='evi2',
                          drop_bands=True) 
       
vprm_inst.sort_and_merge_by_timestamp()

if os.path.exists(os.path.join(out_base, 'veg_map_on_modis_grid.nc')):
    add_land_cover_map(vprm_inst,
                       land_cover_on_modis_grid=os.path.join(out_base, 'veg_map_on_modis_grid.nc'))
else:    
    add_land_cover_map(vprm_inst,
                       copernicus_data_path=copernicus_data_path,
                       save_path=os.path.join(out_base, 'veg_map_on_modis_grid.nc'))
 

vprm_inst.lowess(n_cpus=120)

out_grid = dict()
out_grid['lons'] = lons
out_grid['lats'] = lats
if os.path.exists(os.path.join(out_base, 'regridder.nc')):
    wrf_op = vprm_inst.to_wrf_output(out_grid, weights_for_regridder=os.path.join(out_base, 'regridder.nc'))
else:
    wrf_op = vprm_inst.to_wrf_output(out_grid, driver = 'ESMF_RegridWeightGen', n_cpus=120, 
                                     regridder_save_path=os.path.join(out_base, 'regridder.nc'))


file_base = 'VPRM_input_'
filename_dict = {'lswi': 'LSWI', 'evi': 'EVI', 'veg_fraction': 'VEG_FRA',
                 'lswi_max': 'LSWI_MAX', 'lswi_min': 'LSWI_MIN', 
                 'evi_max': 'EVI_MAX', 'evi_min': 'EVI_MIN'} 
for key in wrf_op.keys():
    wrf_op[key].to_netcdf(os.path.join(out_base,file_base + filename_dict[key] +'_{}.nc'.format(year) ))
