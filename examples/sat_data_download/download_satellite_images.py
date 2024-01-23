import os
from pyVPRM.lib.sat_manager import VIIRS, sentinel2, modis
import yaml 
from datetime import date
import argparse
import shutil

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
p.add_argument("--login_data", type=str)
p.add_argument("--year", type=int, default=None)
p.add_argument("--h", type=int, default=None)
p.add_argument("--v", type=int, default=None)
args = p.parse_args()

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(args.login_data, "r") as stream:
    try:
        logins  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if args.year is not None:
    years = [args.year]
else:
    years = cfg['years']

hvs=cfg['hvs']
if (args.h is not None) & (args.v is not None):
    hvs = [(args.h, args.v)]

for year in years:
    savepath = os.path.join(cfg['sat_image_path'], str(year))
    if cfg['satellite'] == 'modis':
        for i in hvs:
            print('Tile {}'.format(i))
            handler = modis()
            try:
                handler.download(date(year, 1, 1),
                                savepath = savepath,
                                username = logins['modis'][0],
                                pwd = logins['modis'][1],
                                hv=i,
                                delta = 1,
                                enddate=date(year + 1 , 1, 1))
            except Exception as e:
                print(e)

    elif cfg['satellite'] == 'viirs':
        for i in hvs:
            print('Tile {}'.format(i))
            handler = VIIRS()
            try:
                handler.download(date(year, 1, 1),
                                savepath = savepath,
                                username = logins['modis'][0],
                                pwd = logins['modis'][1],
                                hv = i,
                                delta = 1,
                                enddate=date(year + 1, 1, 1))
            except Exception as e:
                print(e)

    else:
        print('No download function for this satellite implemented')
