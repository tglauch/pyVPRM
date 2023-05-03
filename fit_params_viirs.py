import sys
import os
sys.path.append('/home/b/b309233/software/CO2KI/VPRM/')
sys.path.append('/home/b/b309233/software/CO2KI/harvard_forest/')
sys.path.append('/home/b/b309233/software/SatManager')
from sat_manager import VIIRS, sentinel2, modis, copernicus_land_cover_map, satellite_data_manager
import earthpy.plot as ep
from VPRM import vprm 
import warnings
import sys
import pandas as pd
warnings.filterwarnings("ignore")
from era5_class import ERA5
map_function = lambda lon: (lon + 360) if (lon < 0) else lon
from datetime import datetime
from datetime import date
import yaml
import glob
import time
from datetime import timedelta
from dateutil import parser
from scipy.ndimage import uniform_filter
with open("logins.yaml", "r") as stream:
    try:
        logins = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import copy
from joblib import Parallel, delayed
import scipy.interpolate as si
import numpy as np
import rasterio
import datetime
import pytz
from tzwhere import tzwhere
from pyproj import Proj
import math
import pickle
import argparse

def lat_lon_to_modis(lat, lon):
    CELLS = 2400
    VERTICAL_TILES = 18
    HORIZONTAL_TILES = 36
    EARTH_RADIUS = 6371007.181
    EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

    TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
    TILE_HEIGHT = TILE_WIDTH
    CELL_SIZE = TILE_WIDTH / CELLS
    MODIS_GRID = Proj(f'+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext')
    x, y = MODIS_GRID(lon, lat)
    h = (EARTH_WIDTH * .5 + x) / TILE_WIDTH
    v = -(EARTH_WIDTH * .25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
    return int(h), int(v)

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--out_folder", type=str,
               default='/work/bd1231/tglauch/VPRM_output_viirs_new_sites')
p.add_argument("--year", type=int,
               default=2012)
args = p.parse_args()
print(args)

h = args.h
v = args.v
if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)
outfile = os.path.join(args.out_folder, 'h{}v{}_{}.pickle'.format(h, v, args.year))
print(outfile)

site_info = pd.read_pickle('/home/b/b309233/software/CO2KI/VPRM/fluxnet_sites.pkl')

site_list = {'grassland': ['AT-Neu', 'CH-Cha', 'CH-Fru', 'CZ-BK2', 'DE-Gri', 'DE-RuR', 'IT-MBo', 'IT-Tor'],
             'mixed_forests': ['BE-Bra', 'BE-Vie', 'CH-Lae'],
             'cropland': ['BE-Lon', 'CH-Oe2', 'DE-Geb', 'DE-Kli', 'DE-RuS', 'FR-Gri', 'IT-BCi', 'IT-CA2'],
             'evergeen': ['CH-Dav', 'CZ-BK1', 'DE-Lkb', 'DE-Obe', 'DE-Tha', 'FI-Hyy', 'FI-Let', 'FI-Sod',
                          'IT-Lav', 'IT-Ren', 'IT-SRo', 'NL-Loo', 'RU-Fyo'],
             'wetland': ['CZ-wet', 'DE-Akm', 'DE-SfN', 'DE-Spw', 'FR-Pue', 'IT-Cp2'],
             'deciduous': ['DE-Hai', 'DE-Lnf', 'DK-Sor', 'FR-Fon', 'IT-CA1', 'IT-CA3', 'IT-Col', 'IT-Ro2'],
             'shrubland': ['ES-Amo', 'ES-LJu', 'IT-Noe']}

site_dict = dict()
variables = ['NEE_CUT_REF', 'NEE_VUT_REF', 'NEE_CUT_REF_QC', 'NEE_VUT_REF_QC',
            'GPP_NT_VUT_REF', 'GPP_NT_CUT_REF', 'GPP_DT_VUT_REF', 'GPP_DT_CUT_REF',
            'TIMESTAMP_START', 'TIMESTAMP_END']
all_sites = np.concatenate([site_list[i] for i in site_list.keys()])
for s in all_sites:
    row = site_info.loc[site_info['SITE_ID'] == s]
    lat = float(row['lat'])
    long = float(row['long'])
    hv = lat_lon_to_modis(lat, long)
    if not ((hv[0] == h) & (hv[1] == v)):
        continue
    em = ''
    data_files = glob.glob(os.path.join('/work/bd1231/tglauch/data/*{}*/*_FULLSET_H*'.format(s)))
    if len(data_files) == 0:
        print('No data for {}'.format(s))
        continue
    else:
        print('Load {}'.format(data_files[0]))
    idata = pd.read_csv(data_files[0], usecols=variables)
    tzw = tzwhere.tzwhere()
    timezone_str = tzw.tzNameAt(lat, long) # Seville coordinates
    timezone = pytz.timezone(timezone_str)
    dt = datetime.datetime.now()
    datetime_u = []
    for i, row in idata.iterrows():
        datetime_u.append(parser.parse(str(int(row['TIMESTAMP_START'])))  -  timezone.utcoffset(dt))
    years = [t.year for t in datetime_u]
    if args.year not in years:
        print('No data for {}'.format(args.year))
        continue
    idata['datetime_utc'] = datetime_u
    site_dict[s] = {'lonlat': (long, lat), 'input_data': [],
                    'fluxnet_data': idata, 'input_data_timestamps': [] }
print(site_dict)    
print('Load VIIRS tiles')
vprm_inst = vprm()
for c, i in enumerate(glob.glob('/work/bd1231/tglauch/one_year_viirs_europe_new_{}/*h{:02d}v{:02d}*.h*'.format(str(args.year), h, v))):
    print(i)
    handler = VIIRS(sat_image_path=i)
    handler.load()
    vprm_inst.add_sat_img(handler, b_nir='SurfReflect_I2', b_red='SurfReflect_I1',
                          b_blue='no_blue_sensor', b_swir='SurfReflect_I3', smearing=False,
                          which_evi='evi2',
                          drop_bands=True) 
vprm_inst.sort_sat_imgs_by_timestamp()

vprm_inst.lowess(n_cpus=120)

vprm_inst.calc_min_max_evi_lswi()

print('Load Copernicus')
bounds = vprm_inst.prototype.rio.bounds()
lcm = None
for c in glob.glob('/work/bd1231/tglauch/copernicus_classification/*'):
    thandler = copernicus_land_cover_map(c)
    thandler.load()
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

t0 = time.time()
target = 'GPP_DT_VUT_REF'
targets = []
vprm_inst.counter =0
inputs = []
time_range = pd.date_range(start="{}-01-01".format(args.year),end="{}-01-01".format(args.year + 1), freq='H')
for c,t in enumerate(time_range):
    if (t.minute != 0):
        continue
    if (t.hour % 2 != 0):
        continue
    for key in site_dict.keys():
        vrbls = vprm_inst.get_gee_variables(t, lat=site_dict[key]['lonlat'][1],
                                            lon=site_dict[key]['lonlat'][0])
        if vrbls is None:
            continue
        site_dict[key]['input_data'].append(vrbls)
        site_dict[key]['input_data_timestamps'].append(t)
    print(t, (time.time() - t0)/60)

with open(outfile, 'w+b') as handle:
    pickle.dump(site_dict, handle,
                protocol=pickle.HIGHEST_PROTOCOL)
