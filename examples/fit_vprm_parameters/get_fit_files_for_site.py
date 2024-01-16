import sys
import os
import pathlib
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), '../'))
from lib.flux_tower_class import fluxnet, icos
from lib.sat_manager import modis, VIIRS
from lib.sat_manager_add import munich_map
from VPRM import vprm 
import xarray as xr
import pickle
import yaml
import glob
import time
import numpy as np
import pandas as pd
import argparse
import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import pytz
from dateutil import parser
import datetime
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from astropy.convolution import convolve
from pyproj import Transformer
from astropy.convolution import Gaussian2DKernel
from functions import lat_lon_to_modis
import argparse

def all_files_exist(item):
    for key in item.assets.keys():
        path = str(item.assets[key].href[7:])
        if not os.path.exists(path):
            return False
    return True

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--site", type=str)
p.add_argument("--veg_type", type=str)
p.add_argument("--this_year", type=str)
p.add_argument("--cfg_path", type=str)
args = p.parse_args()

with open(args.cfg_path, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

s = args.site
veg_type = args.veg_type
this_year = int(args.this_year) 

veg_type_id = {'GRA': 7, 'MF': 3 , 'CRO':6, 
               'EF': 1, 'DF': 2, 'SH':4, 'WET': 9}

# GRA = Grassland | MF = Mixed Forest | CRO = Cropland | EF = Evergreen Forest | DF = Deciduous Forest | SH = Shrubland

# site_info = pd.read_pickle('../fluxnet_info/fluxnet_sites.pkl')
all_data = []
vprm_insts = []
 #  , 2012
tmin = parser.parse('{}0101'.format(this_year))
tmax = parser.parse('{}1231'.format(this_year))
tower_data_list = []
print(s)
# if len(tower_data_list) >1:
#     continue
if this_year == 2012:
    data_files = glob.glob(os.path.join(cfg['fluxnet_path'], '*{}*/*_FULLSET_H*'.format(s)))
    if len(data_files) == 0: exit()
    flux_tower_inst = fluxnet(data_files[0],'SW_IN_F', 'TA_F',
                              t_start= tmin, t_stop=tmax)
elif this_year in [2021, 2022]:
    data_files = glob.glob(os.path.join(cfg['icos_path'], '*/*_{}_FLUXNET_HH_L2.csv'.format(s)))
    if len(data_files) == 0: exit()
    flux_tower_inst = icos(data_files[0],'SW_IN_F', 'TA_F', 
                            t_start= tmin, t_stop=tmax)
lon = flux_tower_inst.get_lonlat()[0]
lat= flux_tower_inst.get_lonlat()[1]
okay = flux_tower_inst.add_tower_data()
if not okay: exit()
print(flux_tower_inst.get_site_name())
flux_tower_inst.set_land_type(veg_type_id[veg_type])

lat = flux_tower_inst.get_lonlat()[1]
lon = flux_tower_inst.get_lonlat()[0]
h, v = lat_lon_to_modis(lat, lon)
inp_files1 = sorted(glob.glob(os.path.join(cfg['sat_image_path'], str(this_year-1),
                                                    '*h{:02d}v{:02d}*.h*'.format(h, v))))[-3:]
inp_files2 = sorted(glob.glob(os.path.join(cfg['sat_image_path'], str(this_year),
                                                    '*h{:02d}v{:02d}*.h*'.format(h, v))))
inp_files3 = sorted(glob.glob(os.path.join(cfg['sat_image_path'], str(this_year+1),
                                                    '*h{:02d}v{:02d}*.h*'.format(h, v))))[:3]

inp_files = np.concatenate([inp_files1, inp_files2, inp_files3])
inp_files = np.array([i for i in inp_files if '.xml' not in i])
_, inds = np.unique([os.path.basename(f) for f in inp_files], 
                  return_index=True)
inp_files = inp_files[inds]

vprm_inst = vprm(n_cpus=1, sites=[flux_tower_inst])

for c, i in enumerate(inp_files):
    print(i)
    if cfg['satellite'] == 'modis':
        handler = modis(sat_image_path=i)
        handler.load()
        handler.crop(lonlat=(lon, lat), radius=4)
        vprm_inst.add_sat_img(handler, b_nir='sur_refl_b02',
                              b_red='sur_refl_b01',
                              b_blue='sur_refl_b03',b_swir='sur_refl_b06',
                              which_evi='evi', 
                              timestamp_key='sur_refl_day_of_year',
                              mask_bad_pixels=True,
                              mask_clouds=True,
                              drop_bands=True)
    else:
        handler = VIIRS(sat_image_path=i)
        handler.load()
        handler.crop(lonlat=(lon, lat), radius=4)
        vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                          b_blue='no_blue_sensor', b_swir='SurfReflect_I3',
                          which_evi='evi2',
                          drop_bands=True)


# Uncomment if you want to do a spatial smearing with size=(3,3). Otherwise give size manually.

# Sort the satellite images by time and merge internally for easier computations
vprm_inst.sort_and_merge_by_timestamp()
# vprm_inst.smearing(lonlats=[(lon, lat)],
#                    keys=['evi', 'lswi'],
#                    kernel=Gaussian2DKernel(1)) #(21, 21))

vprm_inst.reduce_along_lat_lon()

# Run lowess smoothing
vprm_inst.lowess(times='daily', frac=0.2, it=3,
                 keys=['evi', 'lswi'])

# Calculate necessary parameters for the vprm calculation
vprm_inst.calc_min_max_evi_lswi()
# vprm_insts.append(vprm_inst)
# continue
# ------------------------------------------------------------------

# Loop over the year and get the vprm_variables
data_list = vprm_inst.data_for_fitting()
opath = '/home/b/b309233/software/VPRM_preprocessor/analysis_scripts/site_data_for_fit_modis/{}_{}.pickle'.format(this_year, s)

if os.path.exists(opath):
    os.remove(opath)
    
with open(opath, 'wb+') as ofile:
    pickle.dump(data_list, ofile)
