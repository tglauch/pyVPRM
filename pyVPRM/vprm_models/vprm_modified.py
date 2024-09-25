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
from pyVPRM.vprm_models.vprm_base import vprm_base
import pandas as pd
import itertools
from scipy.optimize import curve_fit
from loguru import logger

class vprm_modified(vprm_base):
    """
    Base class for all meteorologies
    """

    def __init__(self, vprm_pre=None, met=None, fit_params_dict=None):
        super().__init__(vprm_pre, met, fit_params_dict)
        return

    def get_w2_scale(self, lon=None, lat=None, site_name=None, land_cover_type=None):
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
            key = "max_lswi_evergreen"
            if key not in self.vprm_pre.max_lswi.sat_img.keys():
                self.vprm_pre.max_lswi.sat_img[key] = (
                    self.vprm_pre.sat_imgs.sat_img["lswi"]
                    .where(
                        (
                            self.vprm_pre.sat_imgs.sat_img["evi"]
                            > self.vprm_pre.max_lswi.sat_img["growing_season_th"]
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
                            > self.vprm_pre.max_lswi.sat_img["growing_season_th"]
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
        elif lon is not None:
            max_lswi = self.vprm_pre.max_lswi.value_at_lonlat(
                lon, lat, key=key, as_array=False
            )
            min_lswi = self.vprm_pre.min_lswi.value_at_lonlat(
                lon, lat, key="min_lswi", as_array=False
            )
        else:
            max_lswi = self.vprm_pre.max_lswi.sat_img[key]
            min_lswi = self.vprm_pre.min_lswi.sat_img["min_lswi"]
        w2_scale = (lswi - min_lswi) / (max_lswi - min_lswi)
        if land_cover_type in [4, 7]:
            self.buffer["w_scale"] = w2_scale
        else:
            self.buffer["w_scale"] = (1 + lswi) / (1 + max_lswi)
        return self.buffer["w_scale"], w2_scale

    def get_temperature(self, lon=None, lat=None, temperature=None):
        if temperature is not None:
            t = temperature
        elif lon is not None:
            t = (
                self.era5_inst.get_data(lonlat=(lon, lat), key="t2m") - 273.15
            )  # to grad celsius
        else:
            t = self.era5_inst.get_data(key="t2m") - 273.15
        return t

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
        t = self.get_temperature(lon, lat, temperature)
        ret = ((t - tmin) * (t - tmax)) / ((t - tmin) * (t - tmax) - (t - topt) ** 2)
        if isinstance(ret, float):
            if (ret < 0) | (t < tmin):
                ret = 0
        else:
            ret = xr.where((ret < 0) | (t < tmin), 0, ret)
        return ret

    def get_t_dash(self, lat=None, lon=None, land_cover_type=None, temperature=None):
        tcrit = self.vprm_pre.temp_coefficients[land_cover_type][3]
        tmult = self.vprm_pre.temp_coefficients[land_cover_type][4]
        t = self.get_temperature(lon=None, lat=None, temperature=temperature)
        t = xr.where(t < tcrit, tcrit - tmult * (tcrit - t), t)
        return t

    def data_for_fitting(self):
        self.vprm_pre.sat_imgs.sat_img.load()
        for s in self.vprm_pre.sites:
            self.new = True
            site_name = s.get_site_name()
            ret_dict = dict()
            for k in ["evi", "Ps", "par", "Ts", "Ws", "lswi", "tcorr", "Ws2"]:
                ret_dict[k] = []
            drop_rows = []
            for index, row in s.get_data().iterrows():
                img_status = self.vprm_pre._set_sat_img_counter(row["datetime_utc"])
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
                ret_dict["par"].append(self.get_par(ssrd=row["ssrd"]))
                ret_dict["Ts"].append(
                    self.get_t_scale(
                        temperature=row["t2m"], land_cover_type=s.get_land_type()
                    )
                )
                ret_dict["tcorr"].append(self.get_temperature(temperature=row["t2m"]))
                wscales = self.get_w2_scale(site_name=site_name)
                ret_dict["Ws"].append(wscales[0])
                ret_dict["Ws2"].append(wscales[1])
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
        ret_dict["Ts"] = self.get_t_scale(lon, lat, land_cover_type=land_cover_type)
        wscales = self.get_w2_scale(lon, lat, land_cover_type=land_cover_type)
        ret_dict["Ws"] = wscales[0]
        ret_dict["Ws2"] = wscales[2]
        ret_dict["tcorr"] = self.get_temperature(lon, lat)
        ret_dict["tdash"] = self.get_t_dash(lon, lat, land_cover_type=land_cover_type)

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
        no_flux_veg_types=[0, 8],
        inputs=None,
        land_cover_type=None,
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

            if met_regridder_weights is not None:
                if not os.path.exists(os.path.dirname(met_regridder_weights)):
                    os.makedirs(os.path.dirname(met_regridder_weights))

            lc_classes = self.vprm_pre.land_cover_type.sat_img.vprm_classes.values
        else:
            lc_classes = [land_cover_type]

        for i in lc_classes:
            if i in no_flux_veg_types:
                continue
            if inputs is None:
                inputs = self._get_vprm_variables(
                    i, date, regridder_weights=met_regridder_weights
                )
                lcf = self.vprm_pre.land_cover_type.sat_img.sel({"vprm_classes": i})
                if inputs is None:
                    return None
            else:
                lcf = 1
                if "tdash" not in inputs.keys():
                    inputs["tdash"] = inputs["tcorr"]
                    inputs["tdash"][
                        inputs["tdash"] < self.fit_params_dict[i]["tcrit"]
                    ] = self.fit_params_dict[i]["tcrit"] - self.fit_params_dict[i][
                        "tmult"
                    ] * (
                        self.fit_params_dict[i]["tcrit"] - inputs["tdash"]
                    )

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
                        self.fit_params_dict[i]["beta"]
                        + self.fit_params_dict[i]["alpha1"] * inputs["tdash"]
                        + self.fit_params_dict[i]["alpha2"] * inputs["tdash"] ** 2
                        + self.fit_params_dict[i]["gamma"] * inputs["evi"]
                        + self.fit_params_dict[i]["theta1"] * inputs["Ws2"]
                        + self.fit_params_dict[i]["theta2"]
                        * inputs["Ws2"]
                        * inputs["tdash"]
                        + self.fit_params_dict[i]["theta3"]
                        * inputs["Ws2"]
                        * inputs["tdash"] ** 2
                    ),
                    0,
                )
            )
        if isinstance(gpps[0], pd.core.series.Series):
            ret_res["gpp"] = gpps[0]
            ret_res["nee"] = -gpps[0] + respirations[0]
        else:
            ret_res["gpp"] = xr.concat(gpps, dim="z").sum(dim="z")
            ret_res["nee"] = -ret_res["gpp"] + xr.concat(respirations, dim="z").sum(
                dim="z"
            )
        return ret_res

    def fit_vprm_data(
        self,
        data_list,
        variable_dict,
        same_length=True,
        fit_nee=True,
        fit_resp=True,
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
            logger.info(key, min_len)
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
                func_resp = lambda x, b, a1, a2, g, t1, t2, t3: np.maximum(
                    b
                    + a1 * x["tdash"]
                    + a2 * x["tdash"] ** 2
                    + g * x["evi"]
                    + t1 * x["Ws2"]
                    + t2 * x["Ws2"] * x["tdash"]
                    + t3 * x["Ws2"] * x["tdash"] ** 2,
                    0,
                )
                for i in list(
                    itertools.product(
                        np.linspace(-5, 20, 10), np.linspace(0.0, 1.0, 10)
                    )
                ):
                    for c in range(3):
                        data_for_fit["tdash"] = copy.deepcopy(data_for_fit["tcorr"])
                        data_for_fit["tdash"][data_for_fit["tdash"] < i[0]] = i[0] - i[
                            1
                        ] * (i[0] - data_for_fit["tdash"])
                        mask = data_for_fit["par"] == 0
                        fit_respiration = curve_fit(
                            func_resp,
                            data_for_fit[mask],
                            data_for_fit["respiration"][mask],
                            maxfev=5000,
                            p0=[
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                            ],
                        )
                        func_values = func_resp(
                            data_for_fit,
                            fit_respiration[0][0],
                            fit_respiration[0][1],
                            fit_respiration[0][2],
                            fit_respiration[0][3],
                            fit_respiration[0][4],
                            fit_respiration[0][5],
                            fit_respiration[0][6],
                        )
                        mse = np.mean(
                            (func_values[mask] - data_for_fit["respiration"][mask]) ** 2
                        )
                        if mse < best_mse:
                            best_mse = mse
                            best_fit_params = fit_respiration
                            best_fit_temperatures = [i[0], i[1]]
                            best_fit_respiration = func_values

                logger.info("Best MSE Respiration: {}".format(best_mse))
                best_fit_params_dict[key] = {
                    "beta": best_fit_params[0][0],
                    "alpha1": best_fit_params[0][1],
                    "alpha2": best_fit_params[0][2],
                    "gamma": best_fit_params[0][3],
                    "theta1": best_fit_params[0][4],
                    "theta2": best_fit_params[0][5],
                    "theta3": best_fit_params[0][6],
                    "tcrit": best_fit_temperatures[0],
                    "tmult": best_fit_temperatures[1],
                }
            if fit_nee:
                best_mse = np.inf
                data_for_fit["tdash"] = copy.deepcopy(data_for_fit["tcorr"])
                data_for_fit["tdash"][
                    data_for_fit["tdash"] < best_fit_params_dict[key]["tcrit"]
                ] = best_fit_params_dict[key]["tcrit"] - best_fit_params_dict[key][
                    "tmult"
                ] * (
                    best_fit_params_dict[key]["tcrit"] - data_for_fit["tdash"]
                )
                for i in range(200):
                    func = (
                        lambda x, lamb, par0, b, a1, a2, g, t1, t2, t3: -1
                        * (lamb * x["Ws"] * x["Ts"] * x["Ps"])
                        * x["evi"]
                        * x["par"]
                        / (1 + x["par"] / par0)
                        + best_fit_respiration
                        + func_resp(x, b, a1, a2, g, t1, t2, t3)
                    )
                    fit_nee = curve_fit(
                        func,
                        data_for_fit,
                        data_for_fit["nee"],
                        maxfev=5000,
                        p0=[
                            np.random.uniform(0, 0.5),
                            np.random.uniform(100, 1000),
                            best_fit_params_dict[key]["beta"],
                            best_fit_params_dict[key]["alpha1"],
                            best_fit_params_dict[key]["alpha2"],
                            best_fit_params_dict[key]["gamma"],
                            best_fit_params_dict[key]["theta1"],
                            best_fit_params_dict[key]["theta2"],
                            best_fit_params_dict[key]["theta3"],
                        ],
                    )
                    mse = np.mean(
                        (
                            func(
                                data_for_fit,
                                fit_nee[0][0],
                                fit_nee[0][1],
                                fit_nee[0][2],
                                fit_nee[0][3],
                                fit_nee[0][4],
                                fit_nee[0][5],
                                fit_nee[0][6],
                                fit_nee[0][7],
                                fit_nee[0][8],
                            )
                            - data_for_fit["nee"]
                        )
                        ** 2
                    )
                    if mse < best_mse:
                        best_mse = mse
                        best_fit_params = fit_nee
                best_fit_params_dict[key]["lamb"] = best_fit_params[0][0]
                best_fit_params_dict[key]["par0"] = best_fit_params[0][1]
                best_fit_params_dict[key]["beta"] = best_fit_params[0][2]
                best_fit_params_dict[key]["alpha1"] = best_fit_params[0][3]
                best_fit_params_dict[key]["alpha2"] = best_fit_params[0][4]
                best_fit_params_dict[key]["gamma"] = best_fit_params[0][5]
                best_fit_params_dict[key]["theta1"] = best_fit_params[0][6]
                best_fit_params_dict[key]["theta2"] = best_fit_params[0][7]
                best_fit_params_dict[key]["theta3"] = best_fit_params[0][8]
                logger.info("Best MSE NEE: {}".format(best_mse))

        return best_fit_params_dict
