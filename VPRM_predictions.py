import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'lib'))
import yaml 
from datetime import date
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

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--config", type=str)
p.add_argument("--n_cpus", type=int, default=1)
args = p.parse_args()
print(args)

h = args.h
v = args.v

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

vprm_inst = vprm()
for c, i in enumerate(sorted(glob.glob(os.path.join(cfg['sat_image_path'], '*h{:02d}v{:02d}*.h*'.format(h, v))))):
    print(i)
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

vprm_inst.lowess(n_cpus=args.n_cpus)

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
    
for i in np.arange(1,13,1):
    d = pd.to_datetime('{:02d} {}'.format(i, cfg['year']))
    time_range = pd.date_range(start="{}-{:02d}-01 0:00:00".format(cfg['year'], i),
                               end="{}-{:02d}-{:02d} 23:00:00".format(cfg['year'], i,
                                                                      d.daysinmonth),
                               freq='H')
    preds = []
    ts = []
    for t in time_range[:]:
        t0=time.time()
        print(t)
        pred = vprm_inst.make_predictions(t, res_dict=res_dict, which_flux='GPP',
                                         regridder_weights='/work/bd1231/tglauch/tests/regridder_weights_test.nc')
        if pred is None:
            continue
        preds.append(pred)
        ts.append(t)
        print(time.time()-t0)
    preds = xr.concat(preds, 'time')
    preds = preds.assign_coords({'time': ts})
    preds.to_netcdf('/work/bd1231/tglauch/gpp_eu_2015/h{:02d}v{:02d}_{:02d}.h5'.format(h, v, i))
    
