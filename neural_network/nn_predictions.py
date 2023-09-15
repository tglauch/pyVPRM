#!/usr/bin/env python3
import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), '..'))
#sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'lib'))
import yaml 
from lib.sat_manager import VIIRS, sentinel2, modis, earthdata,\
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
from lib.functions import lat_lon_to_modis
import calendar
from datetime import datetime, timedelta
import importlib.util
from keras.models import load_model
import copy

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
p.add_argument("--year", type=int)
p.add_argument("--hourly", action='store_true', default=False)
p.add_argument("--dnn_model", type=str)
args = p.parse_args()
print('Run with args', args)

spec = importlib.util.spec_from_file_location("model", args.dnn_model)
foo = importlib.util.module_from_spec(spec)
sys.modules["model"] = foo
spec.loader.exec_module(foo)

model = load_model(foo.model_name)
 
h = args.h
v = args.v

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

vprm_inst = vprm(n_cpus=args.n_cpus)
for c, i in enumerate(sorted(glob.glob(os.path.join(cfg['sat_image_path'], str(args.year),  '*h{:02d}v{:02d}*.h*'.format(h, v))))):
    if c>2:
        continue
    print(i)
    if cfg['satellite'] == 'modis':
        handler = modis(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, drop_bands=['500m'])
    else:
        handler = VIIRS(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler)


vprm_inst.sort_and_merge_by_timestamp()

#vprm_inst.lowess()

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
    
days_in_year = 365 + calendar.isleap(args.year)
regridder_weights = os.path.join(cfg['predictions_path'],
                                 'regridder_weights_{}_{}.nc'.format(h,v))

prototype = None
if args.hourly:
    for i in np.arange(1, 2, 1): #days_in_year + 1
        time_range=get_hourly_time_range(int(args.year), i)
        preds = []
        ts = []
        for t in time_range[:]:
            t0=time.time()
            vrbls = vprm_inst.get_neural_network_variables(t, era_variables=['ssrd', 't2m', 'swvl1',
                                                                             'swvl2', 'e',
                                                                             'stl1', 'stl2'],
                                                           regridder_weights=regridder_weights)
            t1= time.time()
            print('Get NN data {}'.format(t1-t0))
            data_arr = []
            for var in vrbls.keys(): 
                if prototype is None:
                    prototype = xr.Dataset({"x": (["x"], vrbls[var].coords['x'].values),
                                            "y": (["y"], vrbls[var].coords['y'].values)})
                data_arr.append(np.array(vrbls[var]))
            data_arr = np.array(data_arr).T
            data_arr = data_arr.reshape(2400 * 2400, 15)
            batch_size = 1200000
            t2 = time.time()
            print('Prepare Data {}'.format(t2-t1))
            predgen = foo.CustomDataGen(data_arr, batch_size, shuffle=False)
            t3= time.time()
            print('Init Generator {}'.format(t3-t2))
            pred = model.predict(predgen, predgen.n//batch_size + 1,
                                 workers=1, use_multiprocessing=False)[0].flatten()
            t4= time.time()
            print('Make prediction {}'.format(t4-t3))
            pred = pred.reshape(2400, 2400)
            preds.append(pred)
            ts.append(t)
            print(t, time.time()-t0)
        preds = prototype.assign({'gpp': (['time', 'y', 'x'], preds)})
        preds = preds.assign_coords({'time': ts})
        preds.to_netcdf(os.path.join(cfg['predictions_path'],
                                     'h{:02d}v{:02d}_{:03d}_nn.h5'.format(h, v, i)))
else:

    ts = []
    preds_gpp = []
    preds_nee = []
    
    for w in np.arange(20, 21, 1): 
        ts.append(w)
        t_gpp_preds = []
        t_nee_preds = []
        for i in np.arange((w-1) * 7 + 1, w * 7 + 1, 1):
            time_range=get_hourly_time_range(int(args.year), i)
            for t in time_range[:]:
                t0=time.time()
                vrbls = vprm_inst.get_neural_network_variables(t, era_variables=['ssrd', 't2m', 'swvl1',
                                                                                 'swvl2', 'e',
                                                                                 'stl1', 'stl2'],
                                                               regridder_weights=regridder_weights)
                data_arr = []
                for var in vrbls.keys(): 
                    if prototype is None:
                        prototype = xr.Dataset({"x": (["x"], vrbls[var].coords['x'].values),
                                                "y": (["y"], vrbls[var].coords['y'].values)})
                    data_arr.append(np.array(vrbls[var]))
                data_arr = np.array(data_arr).T
                data_arr = data_arr.reshape(2400 * 2400, 15)
                batch_size = 6000000
                predgen = foo.CustomDataGen(data_arr, batch_size, shuffle=False)
                pred = model.predict(predgen, predgen.n//batch_size + 1)
                t_gpp_preds.append(pred[0].flatten().reshape(2400, 2400))
                t_nee_preds.append(pred[1].flatten().reshape(2400, 2400))
                print(t, time.time()-t0)

        t_gpp_preds = xr.concat(t_gpp_preds, 'time')
        preds_gpp.append(t_gpp_preds.sum(dim='time'))

        t_nee_preds = xr.concat(t_nee_preds, 'time')
        preds_nee.append(t_nee_preds.sum(dim='time'))

    preds_gpp = xr.concat(preds_gpp, 'time')
    preds_gpp = preds_gpp.assign_coords({'time': ts})
    outpath = os.path.join(cfg['predictions_path'],
                          'gpp_h{:02d}v{:02d}_{}_dnn.h5'.format(h, v, args.year))
    if os.path.exists(outpath):
        os.remove(outpath)
    preds_gpp.to_netcdf(outpath)

    preds_nee = xr.concat(preds_nee, 'time')
    preds_nee = preds_nee.assign_coords({'time': ts})
    outpath = os.path.join(cfg['predictions_path'],
                           'nee_h{:02d}v{:02d}_{}_dnn.h5'.format(h, v, args.year))
    if os.path.exists(outpath):
        os.remove(outpath)
    preds_nee.to_netcdf(outpath)
            
