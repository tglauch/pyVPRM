import xarray as xr
import glob
import os
import time
import numpy as np
from dateutil import parser
import pygrib
import copy
import uuid
import datetime
import pandas as pd
import itertools
from scipy.optimize import curve_fit
from loguru import logger

class vprm_base:
    """
    Base class for all meteorologies
    """

    def __init__(self, vprm_pre=None, met=None, fit_params_dict=None):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.buffer = dict()
        self.buffer["cur_lat"] = None
        self.buffer["cur_lon"] = None
        self.fit_params_dict = fit_params_dict
        return

    def load_weather_data(self, hour, day, month, year, era_keys):
        """
        Load meteorlocial data from the available (on DKRZ's levante) data storage

            Parameters:
                    hour (int): hour in UTC
                    day (int): day in UTC
                    month (int): month in UTC
                    year (int): year in UTC
                    era_keys (list): list of ERA5 variables using the shortNames.
                                     See https://confluence.ecmwf.int/displau /CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
            Returns:
                    None
        """
        if self.era5_inst is None:
            logger.info(
                "Not meteorology given. Provide meteorology instance using the set_met method first."
            )

        self.era5_inst.change_date(year=year, month=month, day=day, hour=hour)
        self.hour = hour
        self.day = day
        self.year = year
        self.month = month
        self.date = "{}-{}-{} {}:00:00".format(year, month, day, hour)
        return

    def get_neural_network_variables(
        self,
        datetime_utc,
        lat=None,
        lon=None,
        era_variables=["ssrd", "t2m"],
        regridder_weights=None,
        sat_img_keys=None,
    ):
        """
        Get the variables for an neural network based vegetation model

            Parameters:
                datetime_utc (datetime): The END time of the 1-hour integration period
                lat (float or list of floats): latitude (optional)
                lon (float or list of floats): longitude (optional)
                era_variables (list): ERA5 variables (optional)
                regridder_weights (str): Path to the pre-computed weights for the ERA5 regridder (optional)
                sat_img_keys (str): List of data_vars from the satellite images to be used (optional)
            Returns:
                    None
        """

        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year
        self._set_sat_img_counter(datetime_utc)
        if sat_img_keys is None:
            sat_img_keys = list(self.sat_imgs.sat_img.data_vars)

        if self.prototype_lat_lon is None:
            self._set_prototype_lat_lon()

        self.load_weather_data(hour, day, month, year, era_keys=era_variables)
        if (lat is None) & (lon is None):
            self.era5_inst.regrid(
                dataset=self.prototype_lat_lon,
                weights=regridder_weights,
                n_cpus=self.n_cpus,
            )
        ret_dict = dict()
        # sat_inds = np.concatenate([np.arange(self.counter-8, self.counter-2, 3),
        #                            np.arange(self.counter-2, self.counter+1)])
        for_ret_dict = self.get_sat_img_values_for_all_keys(
            counter_range=self.counter, lon=lon, lat=lat
        )
        for i, key in enumerate(sat_img_keys):
            ret_dict[key] = for_ret_dict[key]
        for key in era_variables:
            if lat is None:
                ret_dict[key] = self.era5_inst.get_data(key=key)
            else:
                ret_dict[key] = self.era5_inst.get_data(
                    lonlat=(lon, lat), key=key
                ).values.flatten()
        if lon is not None:
            land_type = self.land_cover_type.value_at_lonlat(
                lon, lat, key="land_cover_type", interp_method="nearest", as_array=False
            ).values.flatten()
        else:
            land_type = self.land_cover_type.sat_img["land_cover_type"].values
        land_type[
            (land_type != 1)
            & (land_type != 2)
            & (land_type != 3)
            & (land_type != 4)
            & (land_type != 6)
            & (land_type != 7)
        ] = 0
        land_type[~np.isfinite(land_type)] = 0
        ret_dict["land_cover_type"] = land_type
        return ret_dict
