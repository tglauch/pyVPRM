import sys
sys.path.append('/home/b/b309233/software/SatManager')
from sat_manager import VIIRS, sentinel2, modis
import yaml 
from datetime import date

with open("logins.yaml", "r") as stream:
    try:
        logins = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

hv = [(17,2), (17,3), (17,4), (17,5), (18,2), (18,3), (18,4), (18,5), (19,2), (19,3), (19,4), (19,5)]
for i in hv:
    print(i)
    handler = VIIRS()
    handler.download(date(2012, 1, 1),
                    savepath = '/work/bd1231/tglauch/one_year_viirs_europe_new_2012',
                    username = logins['modis'][0],
                    pwd = logins['modis'][1],
                    hv = i,
                    delta = 1,
                    enddate=date(2013, 1, 1))
