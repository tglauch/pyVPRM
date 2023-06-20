#!/usr/bin/env python3
import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'lib'))
import yaml 
from sat_manager import VIIRS, sentinel2, modis, earthdata,\
                        copernicus_land_cover_map, satellite_data_manager
from VPRM import vprm 

import glob
import time
import yaml
import numpy as np
import xarray as xr
import rioxarray as rxr
from pyproj import Transformer
import xesmf as xe
import copy
import rasterio
import pandas as pd
import pickle
import argparse
from functions import lat_lon_to_modis
import calendar
from datetime import datetime, timedelta

def get_hourly_time_range(year, day_of_year):
    start_time = datetime(year, 1, 1) + timedelta(days=int(day_of_year)-1)  # Set the starting time based on the day of the year
    end_time = start_time + timedelta(hours=1)  # Add 1 hour to get the end time of the first hour

    hourly_range = []
    while start_time.timetuple().tm_yday == day_of_year:
        hourly_range.append((start_time))
        start_time = end_time
        end_time = start_time + timedelta(hours=1)

    return hourly_range

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--config", type=str)
p.add_argument("--n_cpus", type=int, default=1)
args = p.parse_args()
print('Run with args', args)

h = args.h
v = args.v

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

vprm_inst = vprm(n_cpus=args.n_cpus)
for c, i in enumerate(sorted(glob.glob(os.path.join(cfg['sat_image_path'], '*h{:02d}v{:02d}*.h*'.format(h, v))))):
    if cfg['satellite'] == 'modis':
        handler = modis(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, b_nir='B02', b_red='B01',
                              b_blue='B03', b_swir='B06',
                              which_evi='evi', max_evi=1,
                              drop_bands=True)
    else:
        handler = VIIRS(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                              b_blue='no_blue_sensor', b_swir='SurfReflect_I3',
                              which_evi='evi2',
                              drop_bands=True)

vprm_inst.sort_and_merge_by_timestamp()

vprm_inst.lowess()

vprm_inst.calc_min_max_evi_lswi()

lcm = None
for c in glob.glob(os.path.join(cfg['copernicus_path'], '*')):
    thandler = copernicus_land_cover_map(c)
    thandler.load()
    bounds = vprm_inst.prototype.sat_img.rio.transform_bounds(thandler.sat_img.rio.crs)
    dj = rasterio.coords.disjoint_bounds(bounds, thandler.sat_img.rio.bounds())
    if dj:
        print('Do not add {}'.format(c))
        continue
    print('Add {}'.format(c))
    if lcm is None:
        lcm=copernicus_land_cover_map(c)
        lcm.load()
    else:
        lcm.add_tile(thandler)

vprm_inst.add_land_cover_map(lcm)

vprm_inst.era5_inst = None

with open(cfg['vprm_params_dict'], 'rb') as ifile:
    res_dict = pickle.load(ifile)
    
days_in_year = 365 + calendar.isleap(int(cfg['year']))
regridder_weights = os.path.join(cfg['predictions_path'],
                                 'regridder_weights_{}_{}.nc'.format(h,v))
for i in np.arange(1,days_in_year+1,1):
    
    time_range=get_hourly_time_range(int(cfg['year']), i)
    preds = []
    ts = []
    for t in time_range[:]:
        t0=time.time()
        print(t)
        pred = vprm_inst.make_vprm_predictions(t, res_dict=res_dict, which_flux='GPP',
                                               regridder_weights=regridder_weights)
        if pred is None:
            continue
        preds.append(pred)
        ts.append(t)
        print(time.time()-t0)
    preds = xr.concat(preds, 'time')
    preds = preds.assign_coords({'time': ts})
    preds.to_netcdf(os.path.join(cfg['predictions_path'],
                                 'h{:02d}v{:02d}_{:03d}.h5'.format(h, v, i)))
    
