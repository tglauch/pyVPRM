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

    def get_t_scale(self, lon=None, lat=None, land_cover_type=None, temperature=None):
        # ToDo
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

        # if (self.new is False) & ('p_scale' in self.buffer.keys()):
        #     return self.buffer['p_scale']
        lswi = self.get_lswi(lon, lat, site_name)
        evi = self.get_evi(lon, lat, site_name)
        p_scale = (1 + lswi) / 2
        if site_name is not None:
            th = float(
                self.vprm_pre.min_max_evi.sat_img.sel(site_names=site_name)["th"]
            )
        elif lon is not None:
            #  land_type = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type', interp_method='nearest', as_array=False)
            th = self.vprm_pre.min_max_evi.value_at_lonlat(
                lon, lat, key="th", interp_method="nearest", as_array=False
            )
        else:
            #   land_type = self.land_cover_type.sat_img['land_cover_type']
            th = self.vprm_pre.min_max_evi.sat_img["th"]
        if land_cover_type == 1:  # Always above threshold. So p_scale is 1
            th = -np.inf
        if land_cover_type in [
            5,
            7,
        ]:  # Never above threshold. So p_scale always (1+lswi)/2
            th = np.inf
        if site_name is not None:
            if evi > th:
                p_scale = 1
        else:
            p_scale = xr.where(evi > th, 1, p_scale)
        # self.buffer['p_scale'] = p_scale
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
        if land_cover_type == 1:
            # How to calculate Max LSWI for Evergreen? What's the growing season?
            key = "max_lswi_evergreen"
            if key not in self.vprm_pre.max_lswi.sat_img.keys():
                self.vprm_pre.max_lswi.sat_img[key] = (
                    self.vprm_pre.sat_imgs.sat_img["lswi"]
                    .where(
                        (
                            self.vprm_pre.sat_imgs.sat_img["evi"]
                            > self.vprm_pre.min_max_evi.sat_img["th"]
                        ),
                        np.nan,
                    )
                    .max(self.vprm_pre.time_key, skipna=True)
                )
        else:
            key = "max_lswi_others"
            if key not in self.vprm_pre.max_lswi.sat_img.keys():
                self.vprm_pre.max_lswi.sat_img[key] = (
                    self.vprm_pre.sat_imgs.sat_img["lswi"]
                    .where(
                        (
                            self.vprm_pre.sat_imgs.sat_img["evi"]
                            > self.vprm_pre.min_max_evi.sat_img["th"]
                        ),
                        np.nan,
                    )
                    .max(self.vprm_pre.time_key, skipna=True)
                )

        if site_name is not None:
            max_lswi = float(
                self.vprm_pre.max_lswi.sat_img.sel(site_names=site_name)[key]
            )
            min_lswi = float(
                self.vprm_pre.min_lswi.sat_img.sel(site_names=site_name)["min_lswi"]
            )
            diff = max_lswi - min_lswi
        elif lon is not None:
            max_lswi = float(
                self.vprm_pre.max_lswi.value_at_lonlat(
                    lon, lat, key=key, as_array=False
                )
            )
            min_lswi = float(
                self.vprm_pre.min_lswi.value_at_lonlat(
                    lon, lat, key="min_lswi", as_array=False
                )
            )
            diff = max_lswi - min_lswi
            if diff < 0.01:
                diff = 0.01
        else:
            max_lswi = self.vprm_pre.max_lswi.sat_img[key]
            min_lswi = self.vprm_pre.min_lswi.sat_img["min_lswi"]
            diff = max_lswi - min_lswi
            diff = xr.where(diff < 0.01, 0.01, diff)

        # Doesn't show any improvements, but increases instability
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

        # if (self.new is False) & ('evi' in self.buffer.keys()):
        #     return self.buffer['evi']
        if site_name is not None:
            self.buffer["evi"] = float(
                self.vprm_pre.sat_imgs.sat_img.sel(site_names=site_name).isel(
                    {self.vprm_pre.time_key: self.vprm_pre.counter}
                )["evi"]
            )
        elif lon is not None:
            self.buffer["evi"] = self.vprm_pre.sat_imgs.value_at_lonlat(
                lon,
                lat,
                as_array=False,
                key="evi",
                isel={self.vprm_pre.time_key: self.vprm_pre.counter},
            )
        else:
            self.buffer["evi"] = self.vprm_pre.sat_imgs.sat_img["evi"].isel(
                {self.vprm_pre.time_key: self.vprm_pre.counter}
            )
        return self.buffer["evi"]

    def get_lswi(self, lon=None, lat=None, site_name=None):
        """
        Get LSWI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    LSWI array
        """

        # if (self.new is False) & ('lswi' in self.buffer.keys()):
        #     return self.buffer['lswi']
        if site_name is not None:
            self.buffer["lswi"] = float(
                self.vprm_pre.sat_imgs.sat_img.sel(site_names=site_name).isel(
                    {self.vprm_pre.time_key: self.vprm_pre.counter}
                )["lswi"]
            )
        elif lon is not None:
            self.buffer["lswi"] = self.vprm_pre.sat_imgs.value_at_lonlat(
                lon, lat, as_array=False, key="lswi", isel={self.time_key: self.counter}
            )
        else:
            self.buffer["lswi"] = self.vprm_pre.sat_imgs.sat_img["lswi"].isel(
                {self.vprm_pre.time_key: self.vprm_pre.counter}
            )
        return self.buffer["lswi"]

    def data_for_fitting(self):
        self.vprm_pre.sat_imgs.sat_img.load()
        for s in self.vprm_pre.sites:
            self.new = True
            site_name = s.get_site_name()
            ret_dict = dict()
            for k in ["evi", "Ps", "par", "Ts", "Ws", "lswi", "tcorr"]:
                ret_dict[k] = []
            drop_rows = []
            for index, row in s.get_data().iterrows():
                datetime_utc = row["datetime_utc"]
                img_status = self.vprm_pre._set_sat_img_counter(datetime_utc)
                if self.era5_inst is not None:
                    era_keys = ["ssrd", "t2m"]
                    hour = datetime_utc.hour
                    day = datetime_utc.day
                    month = datetime_utc.month
                    year = datetime_utc.year
                    self.load_weather_data(hour, day, month, year, era_keys=era_keys)
                # logger.info(self.counter)
                if img_status == False:
                    drop_rows.append(index)
                    continue
                ret_dict["evi"].append(self.get_evi(site_name=site_name))
                ret_dict["Ps"].append(
                    self.get_p_scale(
                        site_name=site_name, land_cover_type=s.get_land_type()
                    )
                )
                if self.era5_inst is None:
                    ret_dict["par"].append(self.get_par(ssrd=row["ssrd"]))
                else:
                    lonlat = s.get_lonlat()
                    ret_dict["par"].append(self.get_par(lon=lonlat[0], lat=lonlat[1]))
                if self.era5_inst is None:
                    Ts_all = self.get_t_scale(
                        land_cover_type=s.get_land_type(), temperature=row["t2m"]
                    )
                else:
                    lonlat = s.get_lonlat()
                    Ts_all = self.get_t_scale(
                        land_cover_type=s.get_land_type(), lon=lonlat[0], lat=lonlat[1]
                    )
                ret_dict["Ts"].append(Ts_all[1])
                ret_dict["tcorr"].append(Ts_all[0])
                ret_dict["Ws"].append(
                    self.get_w_scale(
                        site_name=site_name, land_cover_type=s.get_land_type()
                    )
                )
                ret_dict["lswi"].append(self.get_lswi(site_name=site_name))
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

        era_keys = ["ssrd", "t2m"]
        era_keys.extend(add_era_variables)

        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year

        img_status = self.vprm_pre._set_sat_img_counter(datetime_utc)
        if img_status is False:
            logger.info("No sat image for {}. Return None.".format(datetime_utc))
            return None

        if (lat != self.buffer["cur_lat"]) | (lon != self.buffer["cur_lon"]):
            self.new = True  # Change in lat lon needs new query from satellite images
            self.buffer["cur_lat"] = lat
            self.buffer["cur_lon"] = lon

        if len(era_keys) > 0:

            if self.vprm_pre.prototype_lat_lon is None:
                self.vprm_pre._set_prototype_lat_lon()

            self.load_weather_data(hour, day, month, year, era_keys=era_keys)

            if (lat is None) & (lon is None):
                self.era5_inst.regrid(
                    dataset=self.vprm_pre.prototype_lat_lon,
                    weights=regridder_weights,
                    n_cpus=self.vprm_pre.n_cpus,
                )

        ret_dict = dict()
        ret_dict["evi"] = self.get_evi(lon, lat)
        ret_dict["Ps"] = self.get_p_scale(lon, lat, land_cover_type=land_cover_type)
        ret_dict["par"] = self.get_par(lon, lat)
        Ts_all = self.get_t_scale(lon, lat, land_cover_type=land_cover_type)
        ret_dict["Ts"] = Ts_all[1]
        ret_dict["Ws"] = self.get_w_scale(lon, lat, land_cover_type=land_cover_type)
        ret_dict["tcorr"] = Ts_all[0]
        if add_era_variables != []:
            for i in add_era_variables:
                if lon is not None:
                    ret_dict[i] = self.era5_inst.get_data(
                        lonlat=(lon, lat), key=i
                    ).values.flatten()
                else:
                    ret_dict[i] = self.era5_inst.get_data(key=i).values.flatten()
        return ret_dict

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

        ret_res = dict()
        gpps = []
        respirations = []
        if inputs is None:
            mode = "2d"
        else:
            mode = "1d"
        if mode == "2d":
            if met_regridder_weights is not None:
                if not os.path.exists(os.path.dirname(met_regridder_weights)):
                    os.makedirs(os.path.dirname(met_regridder_weights))
            lc_classes = self.vprm_pre.land_cover_type.sat_img.vprm_classes.values
        else:
            lc_classes = [land_cover_type]

        lc_classes = [i for i in lc_classes if ((i not in no_flux_veg_types) & (i in self.fit_params_dict.keys()))]

        for i in lc_classes:
            if mode == "2d":
                inputs = self._get_vprm_variables(
                    i, date, regridder_weights=met_regridder_weights
                )
                lcf = self.vprm_pre.land_cover_type.sat_img.sel({"vprm_classes": i})
                if inputs is None:
                    return None
            else:
                lcf = 1
            gpps.append(
                lcf
                * (
                    self.fit_params_dict[i]["lamb"]
                    * inputs["Ps"]
                    * inputs["Ws"]
                    * inputs["Ts"]
                    * inputs["evi"]
                    * inputs["par"]
                    / (1 + inputs["par"] / self.fit_params_dict[i]["par0"])
                )
            )
            respirations.append(
                np.maximum(
                    lcf
                    * (
                        self.fit_params_dict[i]["alpha"] * inputs["tcorr"]
                        + self.fit_params_dict[i]["beta"]
                    ),
                    0,
                )
            )
        if concatenate_fluxes:
            if isinstance(gpps[0], pd.core.series.Series):
                ret_res["gpp"] = gpps[0]
                ret_res["nee"] = -gpps[0] + respirations[0]
            else:
                ret_res["gpp"] = xr.concat(gpps, dim="z").sum(dim="z")
                ret_res["nee"] = -ret_res["gpp"] + xr.concat(respirations, dim="z").sum(
                    dim="z"
                )
        else:
            ret_res["gpp"] = xr.concat(gpps, dim="veg_classes")
            ret_res["gpp"] = ret_res["gpp"].assign_coords({"veg_classes": lc_classes})

            ret_res["nee"] = -ret_res["gpp"] + xr.concat(
                respirations, dim="veg_classes"
            )
            ret_res["nee"] = ret_res["nee"].assign_coords({"veg_classes": lc_classes})

            ret_res["nee"] = ret_res["nee"].to_dataset(name="NEE")
            ret_res["gpp"] = ret_res["gpp"].to_dataset(name="GPP")
        return ret_res

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

        variable_dict = {v: k for k, v in variable_dict.items()}
        fit_dict = dict()
        for i in data_list:
            lt = i.get_land_type()
            if lt in fit_dict.keys():
                fit_dict[lt].append(i)
            else:
                fit_dict[lt] = [i]
        if best_fit_params_dict is None:
            best_fit_params_dict = dict()
        for key in fit_dict.keys():
            min_len = np.min([i.get_len() for i in fit_dict[key]])
            logger.info(str(key), str(min_len))
            data_for_fit = []
            for s in fit_dict[key]:
                t_data = s.get_data()
                if same_length:
                    if len(t_data) > min_len:
                        inds = np.random.choice(
                            np.arange(len(t_data)), min_len, replace=False
                        )
                        t_data = t_data.iloc[inds]
                data_for_fit.append(t_data.rename(variable_dict, axis=1))
            data_for_fit = pd.concat(data_for_fit)

            # Respiration
            if fit_resp:
                best_mse = np.inf
                for i in range(200):
                    func = lambda x, a, b: np.maximum(a * x["tcorr"] + b, 0)
                    mask = data_for_fit["par"] == 0
                    fit_respiration = curve_fit(
                        func,
                        data_for_fit[mask],
                        data_for_fit["nee"][mask],
                        maxfev=5000,
                        p0=[np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)],
                    )
                    mse = np.mean(
                        (
                            func(
                                data_for_fit[mask],
                                fit_respiration[0][0],
                                fit_respiration[0][1],
                            )
                            - data_for_fit["nee"][mask]
                        )
                        ** 2
                    )
                    if mse < best_mse:
                        best_mse = mse
                        best_fit_params = fit_respiration
                best_fit_params_dict[key] = {
                    "alpha": best_fit_params[0][0],
                    "beta": best_fit_params[0][1],
                }
                logger.info("Best MSE Respiration: {}".format(best_mse))

            # #GPP
            # best_mse = np.inf
            # for i in range(100):
            #     func = lambda x, lamb, par0: (lamb * data_for_fit['Ws'] * data_for_fit['Ts'] * data_for_fit['Ps']) * data_for_fit['evi'] * data_for_fit['par'] / (1 + data_for_fit['par']/par0)
            #     fit_gpp = curve_fit(func,
            #                         data_for_fit, data_for_fit['gpp'], maxfev=5000,
            #                         p0=[np.random.uniform(0,1), np.random.uniform(0,1000)])
            #     mse = np.mean((func(data_for_fit, fit_gpp[0][0], fit_gpp[0][1]) - data_for_fit['gpp'])**2)
            #     if mse < best_mse:
            #         best_mse = mse
            #         best_fit_params = fit_gpp
            #     best_fit_params_dict[key]['lamb'] = best_fit_params[0][0]
            #     best_fit_params_dict[key]['par0'] = best_fit_params[0][1]

            # NEE
            if fit_nee:
                best_mse = np.inf
                for i in range(200):
                    func = (
                        lambda x, lamb, par0: -1
                        * (lamb * x["Ws"] * x["Ts"] * x["Ps"])
                        * x["evi"]
                        * x["par"]
                        / (1 + x["par"] / par0)
                        + best_fit_params_dict[key]["alpha"] * x["tcorr"]
                        + best_fit_params_dict[key]["beta"]
                    )
                    fit_nee_res = curve_fit(
                        func,
                        data_for_fit,
                        data_for_fit["nee"],
                        maxfev=5000,
                        p0=[np.random.uniform(0, 0.5), np.random.uniform(100, 1000)],
                    )
                    mse = np.mean(
                        (
                            func(data_for_fit, fit_nee_res[0][0], fit_nee_res[0][1])
                            - data_for_fit["nee"]
                        )
                        ** 2
                    )
                    if mse < best_mse:
                        best_mse = mse
                        best_fit_params = fit_nee_res
                best_fit_params_dict[key]["lamb"] = best_fit_params[0][0]
                best_fit_params_dict[key]["par0"] = best_fit_params[0][1]
                logger.info("Best MSE NEE: {}".format(best_mse))
            elif fit_combined:
                best_mse = np.inf
                for i in range(200):
                    func = lambda x, lamb, par0, a, b: -1 * (
                        lamb * x["Ws"] * x["Ts"] * x["Ps"]
                    ) * x["evi"] * x["par"] / (1 + x["par"] / par0) + np.maximum(
                        a * x["tcorr"] + b, 0
                    )
                    fit_nee_res = curve_fit(
                        func,
                        data_for_fit,
                        data_for_fit["nee"],
                        maxfev=5000,
                        p0=[
                            np.random.uniform(0, 0.5),
                            np.random.uniform(100, 1000),
                            0.3,
                            0,
                        ],
                    )
                    mse = np.mean(
                        (
                            func(
                                data_for_fit,
                                fit_nee_res[0][0],
                                fit_nee_res[0][1],
                                fit_nee_res[0][2],
                                fit_nee_res[0][3],
                            )
                            - data_for_fit["nee"]
                        )
                        ** 2
                    )
                    if mse < best_mse:
                        best_mse = mse
                        best_fit_params = fit_nee_res
                best_fit_params_dict[key] = {
                    "lamb": best_fit_params[0][0],
                    "par0": best_fit_params[0][1],
                    "alpha": best_fit_params[0][2],
                    "beta": best_fit_params[0][3],
                }
                logger.info("Best MSE NEE: {}".format(best_mse))
        return best_fit_params_dict
