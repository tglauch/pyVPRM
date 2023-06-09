import sys
sys.path.append('/home/b/b309233/software/SatManager')
from sat_manager import VIIRS, sentinel2, modis
import yaml 
from datetime import date
import argparse

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
p.add_argument("--login_data", type=str)
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



if cfg['satellite'] == 'modis':
    for i in cfg['hvs']:
        handler = modis()
        handler.download(date(cfg['year'], 1, 1),
                        savepath = cfg['sat_image_path'],
                        username = logins['modis'][0],
                        pwd = logins['modis'][1],
                        hv=i,
                        delta = 1,
                        enddate=date(cfg['year'] + 1 , 1, 1))

elif cfg['satellite'] == 'viirs':
    for i in cfg['hvs']:
        print(i)
        handler = VIIRS()
        handler.download(date(cfg['year'], 1, 1),
                        savepath = cfg['sat_image_path'],
                        username = logins['modis'][0],
                        pwd = logins['modis'][1],
                        hv = i,
                        delta = 1,
                        enddate=date(cfg['year'] + 1, 1, 1))

else:
    print('No download function for this satellite implemented')   
