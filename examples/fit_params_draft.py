import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'lib'))
from sat_manager import VIIRS, modis, copernicus_land_cover_map
from VPRM import vprm 
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
p.add_argument("--n_cpus", type=int, default=1)
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


### ToDo: Provide a list of lats and lons for the data extraction, i.e. the flux tower positions
lats = []
lons = []

lonlats = []
for i in range(len(lons)):
    lonlats.append((lons[i], lats[i]))

# ----------- Using the new VPRM Processing Code --------------------

vprm_inst = vprm(n_cpus=args.n_cpus)

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

# Uncomment if you want to do a spatial smearing with size=(3,3). Otherwise give size manually.
#vprm_inst.smearing(lonlats=lonlats)

# Sort the satellite images by time and merge internally for easier computations
vprm_inst.sort_and_merge_by_timestamp()

# Run lowess smoothing
vprm_inst.lowess(lonlats=lonlats)

# Calculate necessary parameters for the vprm calculation
vprm_inst.calc_min_max_evi_lswi()

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
