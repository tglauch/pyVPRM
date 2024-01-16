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
