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
from pyVPRM.flux_tower_libs.FFP_footprint_class import FFP_footprint_manager
from pyVPRM.flux_tower_libs.KM_footprint_class import KM_footprint_manager

class vprm_base_model_footprint:
    """
    Base class for all meteorologies
    """

    def __init__(self, vprm_pre=None, met=None, fit_params_dict=None,
                 footprint=None, flux_tower_instance=None):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.fit_params_dict = fit_params_dict
        self.footprint = footprint
        self.flux_tower_instance = flux_tower_instance
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

    def get_t_scale(self, lon=None, lat=None, land_cover_type=None, temperature=None):
        """
        Get VPRM t_scale

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    t_scale array
        """

        tmin = self.vprm_pre.temp_coefficients[land_cover_type][0]
        topt = self.vprm_pre.temp_coefficients[land_cover_type][1]
        tmax = self.vprm_pre.temp_coefficients[land_cover_type][2]
        tlow = self.vprm_pre.temp_coefficients[land_cover_type][3]
        if temperature is not None:
            t = temperature
        elif lon is not None:
            t = (
                float(self.era5_inst.get_data(lonlat=(lon, lat), key="t2m")) - 273.15
            )  # to grad celsius
        else:
            t = self.era5_inst.get_data(key="t2m") - 273.15
        ret = ((t - tmin) * (t - tmax)) / ((t - tmin) * (t - tmax) - (t - topt) ** 2)
        if isinstance(ret, float):
            if (ret < 0) | (t < tmin):
                ret = 0
            if t < tmin:
                t = tmin
        else:
            ret = xr.where((ret < 0) | (t < tmin), 0, ret)
            t = xr.where(t < tlow, tlow, t)
        return (t, ret)

    def get_p_scale(self, lon=None, lat=None, site_name=None, land_cover_type=None):
        # ToDo
        """
        Get VPRM p_scale for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    p_scale array
        """

        return p_scale

    def get_par(self, lon=None, lat=None, ssrd=None):
        """
        Get VPRM par

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    par array
        """
        if ssrd is not None:
            ret = ssrd / 0.505
        elif lon is not None:
            ret = (
                float(self.era5_inst.get_data(lonlat=(lon, lat), key="ssrd"))
                / 0.505
                / 3600
            )
        else:
            ret = self.era5_inst.get_data(key="ssrd") / 0.505 / 3600
        return ret

    def get_w_scale(self, lon=None, lat=None, site_name=None, land_cover_type=None):
        """
        Get VPRM w_scale

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    w_scale array
        """

        # if (self.new is False) & ('w_scale' in self.buffer.keys()):
        #     return self.buffer['w_scale']
        lswi = self.get_lswi(lon, lat, site_name)
        
        if land_cover_type in [4, 7]:
            self.buffer["w_scale"] = (lswi - min_lswi) / (max_lswi - min_lswi)
        else:
            self.buffer["w_scale"] = (1 + lswi) / (1 + max_lswi)
        return self.buffer["w_scale"]

    def get_evi(self, lon=None, lat=None, site_name=None):
        """
        Get EVI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    EVI array
        """

    def get_lswi(self, lon=None, lat=None, site_name=None):
        """
        Get LSWI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    LSWI array
        """

    def data_for_fitting(self, footprint_regridder_path='./'):
        ds = vprm_pre.sat_imgs.sat_img.drop(['scl','ndvi']).drop(['time'])
        ds = ds.assign_attrs(crs=ds.rio.crs)
        ds['min_evi'] = vprm_pre.min_max_evi.sat_img['min_evi']
        ds['max_evi'] = vprm_pre.min_max_evi.sat_img['max_evi']
        ds['th'] = vprm_pre.min_max_evi.sat_img['th']
        ds['min_lswi'] =  vprm_pre.min_lswi.sat_img['min_lswi']
        ds['max_lswi'] =  vprm_pre.max_lswi.sat_img['max_lswi']
        flux_tower_keys = ['t2m', 'ssrd', 'ZL', 'FETCH_90',
                          'NEE_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_DT_VUT_REF']
        for key in flux_tower_keys:
            ds[key] = xr.DataArray(
                self.flux_tower_instance.flux_data[key].values,
                dims=('datetime_utc',),
                coords={'datetime_utc': self.flux_tower_instance.flux_data['datetime_utc']})
        t0 = np.datetime64(vprm_pre.timestamp_start)
        ds = ds.assign_coords(
            days_since_t0=(
                "datetime_utc",
                ((ds.datetime_utc.data - t0) / np.timedelta64(1, "D")).astype(int)))

        # Only possible to calculate footprints under this condition
        footprint_timestamps=ds['ZL'][ds['ZL']>0]['datetime_utc']
        ffp_handler = FFP_footprint_manager(time_stamps=footprint_timestamps,
                                            flux_tower_manager=self.flux_tower_instance, 
                                            calculation_grid_side_length=1500,
                                            calculation_grid_pixels_per_side=300)
        km_handler = KM_footprint_manager(time_stamps=footprint_timestamps,
                                          flux_tower_manager=self.flux_tower_instance, 
                                          calculation_grid_side_length=1500,
                                          calculation_grid_pixels_per_side=300)

        km_handler.make_calculation_grid()
        km_handler.calculate_footprints()
        
        ffp_handler.make_calculation_grid()
        ffp_handler.calculate_footprints()

        km_handler.regrid_calculation_grid_to_satellite_grid(handler.sat_img,
                                                             footprint_regridder_path)
        ffp_handler.regrid_calculation_grid_to_satellite_grid(handler.sat_img,
                                                              footprint_regridder_path)

        ds['km_footprint'] = km_handler.footprint_on_satellite_grid['footprint']
        ds['ffp_footprint'] = ffp_handler.footprint_on_satellite_grid['footprint']
        if vprm_pre.land_cover_type is not None:
            ds['land_cover_map'] = vprm_pre.land_cover_type.sat_img

        era5_meteo_data = dict()
        for key in era5_meteo_data.keys():
            ds[key] = era5_meteo_data[key]

        return 

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

        era_keys = ["ssrd", "t2m"]
        era_keys.extend(add_era_variables)


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
        """
        Run a VPRM fit
        Parameters:
            data_list (list): A list of instances from type flux_tower_data
            variable_dict (dict): A dictionary giving the target keys for gpp und respiration
                                  i.e. {'gpp': 'GPP_DT_VUT_REF', 'respiration': 'RECO_NT_VUT_REF',
                                        'nee': 'NEE_DT_VUT_REF'}
            same_length (bool): If true all sites have the same number of input data for the fit.
        Returns:
            A dictionary with the fit parameters
        """

        return best_fit_params_dict
