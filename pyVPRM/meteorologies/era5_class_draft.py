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
from pyVPRM.meteorologies.met_base_class import met_data_handler_base
from loguru import logger

class met_data_handler(met_data_handler_base):

    def __init__(self, year, month, day, hour, keys=[]):
        super().__init__()

    # Init with year, month, day, hour and the required era5 keys as given in the
    #  keys_dict above

    def regrid(
        self,
        lats=None,
        lons=None,
        dataset=None,
        n_cpus=1,
        weights=None,
        overwrite_regridder=False,
    ):
        # Regrid  the ERA5 data to given grid with lats and lons or an xarray
        # with given coords alternatively. If regridding weights are given to not recalculate
        # them.
        return

    def get_data(self, lonlat=None, key=None):
        # Return ERA5 data for lonlat if lonlat is not None else return all data.
        # Pick a specific key if key is not None. Return as xarray dataset
        return

    def _init_data_for_day(self):
        # If something should be done if a new date is provided
        return

    def _load_data_for_hour(self):
        # If something should be done if only the hour is changed
        return


if __name__ == "__main__":
    year = "2000"
    month = 2
    day = 20
    hour = 5  # UTC hour
    era5_handler = class_name(year, month, day)
    era5_handler.change_date(hour)
    ret = era5_handler.get_data()
    logger.info(ret)
