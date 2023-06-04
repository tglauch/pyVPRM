import sys
import os
from sat_manager import VIIRS, modis, copernicus_land_cover_map
from VPRM import vprm 
import warnings
warnings.filterwarnings("ignore")
from era5_class_new import ERA5
map_function = lambda lon: (lon + 360) if (lon < 0) else lon
import yaml
import glob
import time
import numpy as np
import pandas as pd
import pickle
import argparse

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--config", type=str)
args = p.parse_args()
print(args)


with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

h = args.h
v = args.v

if not os.path.exists(cfg['out_path']):
    os.makedirs(cfg['out_path'])

outfile = os.path.join(cfg['out_path'], 'h{}v{}_{}.pickle'.format(h, v, cfg['year']))
print(outfile)


### ToDo: Provide (code for) a list of lats and lons
lats = []
lons = []

lonlats = []
for i in range(len(lons)):
    lonlats.append((lons[i], lats[i]))

# ----------- Using the new VPRM Processing Code --------------------

vprm_inst = vprm()

for c, i in enumerate(glob.glob(os.path.join(cfg['sat_image_path'], '*h{:02d}v{:02d}*.h*'.format(h, v)))):
    print(i)
    if cfg['satellite'] == 'modis':
        handler = modis(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, b_nir='B02', b_red='B01',
                              b_blue='B03', b_swir='B06',
                              which_evi='evi',
                              drop_bands=True)
    else:
        handler = VIIRS(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                          b_blue='no_blue_sensor', b_swir='SurfReflect_I3',
                          which_evi='evi2',
                          drop_bands=True)

vprm_inst.smearing(lonlats=lonlats)

vprm_inst.sort_and_merge_by_timestamp()

vprm_inst.lowess(n_cpus=1, lonlats=lonlats)

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
    if lcm is None:
        lcm=copernicus_land_cover_map(c)
        lcm.load()
    else:
        lcm.add_tile(thandler)
vprm_inst.add_land_cover_map(lcm)


# ------------------------------------------------------------------

# Loop over the year and get the vprm_variables

time_range = pd.date_range(start="{}-01-01".format(cfg['year']),
                           end="{}-01-01".format(cfg['year'] + 1),
                           freq='H')

for c,t in enumerate(time_range):
    t0 = time.time()
    # Returns dict with VPRM input variables for the different lats and lons
    # Additional ERA5 variables can be added trough the  add_era_variables key
    vrbls = vprm_inst.get_vprm_variables(t, lat=lats, lon=lons,
                                         add_era_variables=['svwl1'])
    
# ToDo: write to your own format and store for further analysis
