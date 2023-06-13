import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'lib'))
from sat_manager import VIIRS, sentinel2, modis, copernicus_land_cover_map, satellite_data_manager
from VPRM import vprm 
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from era5_class_new import ERA5
from pyproj import Proj
import yaml
import glob
import time
from dateutil import parser
import numpy as np
import rasterio
import pytz
from tzwhere import tzwhere
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



#Prepare Fluxnet Dataset
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
            'TIMESTAMP_START', 'TIMESTAMP_END', 'WD', 'WS', 'WS_MAX']
all_sites = np.concatenate([site_list[i] for i in site_list.keys()])

for s in all_sites:
    row = site_info.loc[site_info['SITE_ID'] == s]
    lat = float(row['lat'])
    lon = float(row['long'])
    hv = lat_lon_to_modis(lat, lon)
    if not ((hv[0] == h) & (hv[1] == v)):
        continue
    data_files = glob.glob(os.path.join(cfg['fluxnet_path'], '*{}*/*_FULLSET_H*'.format(s)))
    if len(data_files) == 0:
        print('No data for {}'.format(s))
        continue
    else:
        print('Load {}'.format(data_files[0]))
    idata = pd.read_csv(data_files[0], usecols=lambda x: x in variables)
    tzw = tzwhere.tzwhere()
    timezone_str = tzw.tzNameAt(lat, lon) 
    timezone = pytz.timezone(timezone_str)
    dt = parser.parse('200001010000') # pick a date that is definitely standard time and not DST /// old: datetime.datetime.now()
    datetime_u = []
    for i, row in idata.iterrows():
        datetime_u.append(parser.parse(str(int(row['TIMESTAMP_START'])))  -  timezone.utcoffset(dt))
    years = [t.year for t in datetime_u]
    if cfg['year'] not in years:
        print('No data for {}'.format(cfg['year']))
        continue
    idata['datetime_utc'] = datetime_u
    site_dict[s] = {'lonlat': (lon, lat), 'input_data': [],
                    'fluxnet_data': idata, 'input_data_timestamps': [] }

if len(site_dict.keys()) == 0 :
    exit()

lats = []
lons = []
lonlats = []
for key in site_dict.keys():
    lonlats.append((site_dict[key]['lonlat'][0], site_dict[key]['lonlat'][1]))
    lats.append(site_dict[key]['lonlat'][1])
    lons.append(site_dict[key]['lonlat'][0])

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
vprm_inst.smearing(lonlats=lonlats)

vprm_inst.sort_and_merge_by_timestamp()

vprm_inst.lowess(lonlats=lonlats)

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


# ------------------------------------------------------------------

# Loop over the year
targets = []
vprm_inst.counter =0
inputs = []
time_range = pd.date_range(start="{}-01-01".format(cfg['year']),
                           end="{}-01-01".format(cfg['year'] + 1),
                           freq='H')

for c,t in enumerate(time_range):
    t0 = time.time()
    if (t.hour % 1 != 0):
        continue
    vrbls = vprm_inst.get_vprm_variables(t, lat=lats, lon=lons)
    if vrbls is None:
        continue
    else:
        for key_c, key in enumerate(site_dict.keys()):
            add_arr = []
            for var in vrbls.keys(): 
                add_arr.append(vrbls[var][key_c])
            site_dict[key]['input_data'].append(add_arr)
            site_dict[key]['input_data_timestamps'].append(t)
    print(t, (time.time() - t0))

with open(outfile, 'w+b') as handle:
    pickle.dump(site_dict, handle,
                protocol=pickle.HIGHEST_PROTOCOL)
