import xarray as xr
import glob
import os
import time
import numpy as np
from dateutil import parser
from scipy.interpolate import interp2d
import pygrib
import copy
import xesmf as xe
import uuid
import datetime

#map_function = lambda lon: (lon - 360) if (lon > 180) else lon
map_function = lambda lon: (lon + 360) if (lon < 0) else lon


bpaths = {'sf00': '/pool/data/ERA5/E5/sf/an/1H', # '/work/bk1099/data/sf00_1H'i,
          'sf12': '/pool/data/ERA5/E5/sf/fc/1H', #'/work/bk1099/data/sf12_1H',
          'pl00': '/pool/data/ERA5/E5/pl/an/1H', #'/work/bk1099/data/pl00_1H'i,
          'ml00': '/pool/data/ERA5/E5/ml/an/1H'} #'/work/bk1099/data/ml00_1H/'}

# Check documentation under 
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Spatialgrid

keys_dict = {'ssrd': [169, 'sf12'],#surface solar radiation downwards(J/m**2)
             't2m': [167, 'sf00'], # temperature 2 m 
#             'slt': [43, 'sf00'], #soil type, not sure it's useful...MODIS should do better
             'sp': [134, 'sf00'], # surface pressure
             'tcc': [164, 'sf00'],  #total cloud cover
             'stl1': [139, 'sf00'], # soil temperature level1
#             'stl2': [170, 'E5sf00'], # soil temperature level2
             'stl3': [183, 'sf00'],# soil temperature level3
             'swvl1': [39, 'sf00'],# soil water level 1
             'swvl2': [40, 'sf00'],# soil water level 2
             'swvl3': [41, 'sf00'],# soil water level 3
             'swvl4': [42, 'sf00'],# soil water level 3             
#             'tp': [228, 'sf12'], #total precipitation over given time
#             'ssr': [176, 'sf12'], #net surface solar radiation (J/m**2)
#             'str': [177, 'sf12'],#net surface thermal radiation (J/m**2)
             'src': [198, 'sf00'],
             'q': [133, 'ml00'], # specific humidity (%)
             'e': [182, 'sf12']} # evaporation


class ERA5:
    '''
    Class for using ERA5 data available on Levante's DKRZ cluster.
    '''
    
    def __init__(self, year, month, day, hour, keys=[]):
       # Init with year, month, day, hour and the required era5 keys as given in the 
       #  keys_dict above

    def change_date(self, hour, day, month, year):
        # Load data for new date

    def regrid(self, lats=None, lons=None, dataset=None, n_cpus=1,
               weights=None, overwrite_regridder=False):
        # Regrid  the ERA5 data to given grid with lats and lons or an xarray 
        # with given coords alternatively. If regridding weights are given to not recalculate 
        # them.  

    def get_data(self, lonlat=None, key=None):
        # Return ERA5 data for lonlat if lonlat is not None else return all data.
        # Pick a specific key if key is not None. Return as xarray dataset

if(__name__ == '__main__'):
    year = '2000'
    month = 2
    day = 20
    hour = 5  #UTC hour
    position = {'lat': 50.30493, 'long': 5.99812}
    era5_handler = ERA5(year, month, day) 
    era5_handler.change_date(hour)
    ret = era5_handler.get_data()
    print(ret)
