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

    def __init__(self, vprm_pre=None, met=None, met_keys=[]):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.buffer = dict()
        self.buffer["cur_lat"] = None
        self.buffer["cur_lon"] = None
        self.fit_params_dict = fit_params_dict
        self.met_keys = met_keys
        return

    def load_weather_data(self, hour, day, month, year):
        """
        Load meteorlocial data from the available (on DKRZ's levante) data storage

            Parameters:
                    hour (int): hour in UTC
                    day (int): day in UTC
                    month (int): month in UTC
                    year (int): year in UTC
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
        
    def data_for_fitting(self):
        self.vprm_pre.sat_imgs.sat_img.load()
        for s in self.vprm_pre.sites:
            self.new = True
            site_name = s.get_site_name()
            ret_dict = dict()
            for k in self.vprm_pre.sat_imgs.sat_img.keys():
                ret_dict[k] = []
            drop_rows = []
            for index, row in s.get_data().iterrows():
                datetime_utc = row["datetime_utc"]
                img_status = self.vprm_pre._set_sat_img_counter(datetime_utc)
                if self.era5_inst is not None:
                    hour = datetime_utc.hour
                    day = datetime_utc.day
                    month = datetime_utc.month
                    year = datetime_utc.year
                    self.load_weather_data(hour, day, month, year,
                                           era_keys=self.met_keys)
                # logger.info(self.counter)
                if img_status == False:
                    drop_rows.append(index)
                    continue
                temp_data = self.vprm_pre.get_sat_img_values_for_all_keys()
                for k in temp_data.keys():
                    ret_dict[k].append(temp_data[k])
                if self.era5_inst is None:
                    print('For neural network applications an ERA5 instance needs to be provided')
                else:
                    lonlat = s.get_lonlat()
                    for key in met_keys:
                        ret_dict[key].append(float(self.era5_inst.get_data(lonlat=(lon, lat), key=key)))
            s.drop_rows_by_index(drop_rows)
            s.add_columns(ret_dict)
        return self.vprm_pre.sites

    def _get_vprm_variables(
        self,
        land_cover_type,
        datetime_utc=None,
        lat=None,
        lon=None,
        add_era_variables=[],
        regridder_weights=None,
    ):
        """
        Get the variables for the Vegetation Photosynthesis and Respiration Model

            Parameters:
                datetime_utc (datetime): The time of interest
                lat (float): A latitude (optional)
                lon (float): A longitude (optional)
                add_era_variables (list): Additional era variables for modifications of the VPRM
                regridder_weights (str): Path to the pre-computed weights for the ERA5 regridder
                tower_dict (dict): Alternatively to a model meteorology and land cover map also the data from the flux tower can be passed in a dictionary Minimaly required are the variables 't2m', 'ssrd', 'land_cover_type'
            Returns:
                    None
        """
        pass

    def make_vprm_predictions(
        self,
        date=None,
        met_regridder_weights=None,
        inputs=None,
        no_flux_veg_types=[0, 8],
        land_cover_type=None,
        concatenate_fluxes=True,
    ):
        """
        Using the VPRM fit parameters make predictions on the entire satellite image.

            Parameters:
                date (datetime object): The date for the prediction
                regridder_weights (str): Path to the weights file for regridding from ERA5
                                         to the satellite grid
                no_flux_veg_types (list of ints): flux type ids that get a default GPP/NEE of 0
                                                  (e.g. oceans, deserts...)
            Returns:
                    None
        """
        pass

    def fit_vprm_data(
        self,
        data_list,
        variable_dict,
        same_length=True,
        fit_nee=True,
        fit_resp=True,
        fit_combined=False,
        best_fit_params_dict=None,
    ):
        pass
