#!/usr/bin/env python3
import sys
import os
import pathlib
sys.path.append('/home/b/b309233/software/VPRM_preprocessor/')
from pyVPRM.lib.sat_manager import VIIRS, sentinel2, modis, earthdata,\
                        copernicus_land_cover_map, satellite_data_manager
from pyVPRM.VPRM import vprm 
from pyVPRM.meteorologies import era5_monthly_xr, era5_class_dkrz
from pyVPRM.lib.functions import lat_lon_to_modis
import glob
import time
import yaml
import numpy as np
import xarray as xr
import rasterio
import pandas as pd
import pickle
import argparse
import calendar
from datetime import datetime, timedelta
from shapely.geometry import box
import geopandas as gpd


def get_hourly_time_range(year, day_of_year):
    start_time = datetime(year, 1, 1) + timedelta(days=int(day_of_year)-1)  # Set the starting time based on the day of the year
    end_time = start_time + timedelta(hours=1)  # Add 1 hour to get the end time of the first hour

    hourly_range = []
    while start_time.timetuple().tm_yday == day_of_year:
        hourly_range.append((start_time))
        start_time = end_time
        end_time = start_time + timedelta(hours=1)
    return hourly_range

# Read command line arguments
p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--config", type=str)
p.add_argument("--n_cpus", type=int, default=1)
p.add_argument("--year", type=int)
args = p.parse_args()
print('Run with args', args)

h = args.h
v = args.v

#Read config
with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Initialize VPRM instance and add satellite images
vprm_inst = vprm(vprm_config_path='../../pyVPRM/vprm_configs/copernicus_land_cover.yaml',
                 n_cpus=args.n_cpus)
files = glob.glob(os.path.join(cfg['sat_image_path'],# str(args.year),
                               '*h{:02d}v{:02d}*.nc'.format(h, v)))
for c, i in enumerate(sorted(files)):
    if '.xml' in i:
        continue
    print(i)
    if cfg['satellite'] == 'modis':
        handler = modis(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, b_nir='sur_refl_b02', b_red='sur_refl_b01',
                              b_blue='sur_refl_b03', b_swir='sur_refl_b06',
                              which_evi='evi',
                              drop_bands=True,
                              timestamp_key='sur_refl_day_of_year',
                              mask_bad_pixels=True,
                              mask_clouds=True)
    else:
        handler = VIIRS(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                              b_blue='no_blue_sensor', b_swir='SurfReflect_I3',
                              which_evi='evi2', 
                              drop_bands=True)

# Pre-Calculations for the VPRM model. This shouldn't change
vprm_inst.sort_and_merge_by_timestamp()
vprm_inst.lowess(keys=['evi', 'lswi'],
                 times='daily',
                 frac=0.2, it=3)
vprm_inst.clip_values('evi', 0, 1)
vprm_inst.clip_values('lswi',-1, 1)
vprm_inst.calc_min_max_evi_lswi()

# Add land covery map(s)
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
        lcm.add_tile(thandler, reproject=False)

geom = box(*vprm_inst.sat_imgs.sat_img.rio.bounds())
df = gpd.GeoDataFrame({"id":1,"geometry":[geom]})
df = df.set_crs(vprm_inst.sat_imgs.sat_img.rio.crs)
df = df.scale(1.3, 1.3)
lcm.crop_to_polygon(df)

vprm_inst.add_land_cover_map(lcm, regridder_save_path=os.path.join(cfg['predictions_path'],
                                                                   'regridder.nc'), mpi=False)

# Set meteorology
era5_inst = era5_monthly_xr.met_data_handler(args.year, 1, 1, 0,
                                             './data/era5',
                                             keys=['t2m', 'ssrd']) 
vprm_inst.set_met(era5_inst)

# Load VPRM parameters from a dictionary 
with open(cfg['vprm_params_dict'], 'rb') as ifile:
    res_dict = pickle.load(ifile)
    
# Make NEE/GPP flux predictions
days_in_year = 365 + calendar.isleap(args.year)
met_regridder_weights = os.path.join(cfg['predictions_path'],
                                    'met_regridder_weights.nc')

for i in np.arange(160,161, 1):
    time_range=get_hourly_time_range(int(args.year), i)
    preds_gpp = []
    preds_nee = []
    ts = []
    for t in time_range[:]:
        t0=time.time()
        print(t)
        pred = vprm_inst.make_vprm_predictions(t, fit_params_dict=res_dict,
                                               met_regridder_weights=met_regridder_weights)
        print(pred)
        if pred is None:
            continue
        preds_gpp.append(pred['gpp'])
        preds_nee.append(pred['nee'])
        ts.append(t)
        print(time.time()-t0)

    preds_gpp = xr.concat(preds_gpp, 'time')
    preds_gpp = preds_gpp.assign_coords({'time': ts})
    outpath = os.path.join(cfg['predictions_path'],
                           'gpp_h{:02d}v{:02d}_{}_{:03d}.h5'.format(h, v, args.year, i))
    if os.path.exists(outpath):
        os.remove(outpath)
    preds_gpp.to_netcdf(outpath)
    preds_gpp.close()
    
    preds_nee = xr.concat(preds_nee, 'time')
    preds_nee = preds_nee.assign_coords({'time': ts})
    outpath = os.path.join(cfg['predictions_path'],
                           'nee_h{:02d}v{:02d}_{}_{:03d}.h5'.format(h, v,args.year, i))
    if os.path.exists(outpath):
        os.remove(outpath)
    preds_nee.to_netcdf(outpath)
    preds_nee.close()

