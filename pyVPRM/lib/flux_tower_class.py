from pyproj import Proj
import pandas as pd
import pytz
from tzwhere import tzwhere
from dateutil import parser
import numpy as np
import os
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
import pathlib
import glob


class flux_tower_data:
    # Class to store flux tower data in unique format

    def __init__(self, t_start, t_stop, ssrd_key, t2m_key, site_name):
        self.tstart = t_start
        self.tstop = t_stop
        self.t2m_key = t2m_key
        self.ssrd_key = ssrd_key
        self.len = None
        self.site_dict = None
        self.site_name = site_name
        return

    def set_land_type(self, lt):
        self.land_cover_type = lt
        return

    def get_utcs(self):
        return self.site_dict[list(self.site_dict.keys())[0]]["flux_data"][
            "datetime_utc"
        ].values

    def get_lonlat(self):
        return (self.lon, self.lat)

    def get_site_name(self):
        return self.site_name

    def get_data(self):
        return self.flux_data

    def get_len(self):
        return len(self.flux_data)

    def get_land_type(self):
        return self.land_cover_type

    def drop_rows_by_index(self, indices):
        self.flux_data = self.flux_data.drop(indices)

    def add_columns(self, add_dict):
        for i in add_dict.keys():
            self.flux_data[i] = add_dict[i]
        return

    def cut_to_timewindow(self, tstart, tstop, key="datetime_utc"):
        mask = (self.flux_data[key] >= tstart) & (self.flux_data[key] <= tstop)
        self.flux_data = self.flux_data[mask]
        return


class brazil_flux_data(flux_tower_data):
    # https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1842

    def __init__(
        self,
        data_path,
        ssrd_key=None,
        t2m_key=None,
        use_vars=None,
        t_start=None,
        t_stop=None,
    ):

        site_name = data_path.split("/")[-1].split("_")[0]
        self.data_path = data_path

        super().__init__(t_start, t_stop, ssrd_key, t2m_key, site_name)

        if use_vars is None:
            self.vars = variables = [
                "Year_LBAMIP",
                "DoY_LBAMIP",
                "Hour_LBAMIP",
                "NEE",
                "NEEf",
                "par",
                "mrs",
                "sco2",
                "GEP_model",
                "NEE_model",
                "Re_model",
                "Tair_LBAMIP",
                "Qair_LBAMIP",
                "SWdown_LBAMIP",
                "Wind_LBAMIP",
                "GF_Tair_LBAMIP",
                "tsoil1",
                "tsoil2",
                "par_fill",
                "ust",
            ]
        if use_vars is "all":
            self.vars = "all"
        site_info_dict = dict()
        site_info_dict["K67"] = dict(lat=-2.857, lon=-54.959, veg_class="EF")
        site_info_dict["K77"] = dict(lat=-3.0202, lon=-54.8885, veg_class="CRO")
        site_info_dict["K83"] = dict(lat=-3.017, lon=-54.9707, veg_class="EF")
        site_info_dict["K34"] = dict(lat=-2.6091, lon=-60.2093, veg_class="EF")
        site_info_dict["CAX"] = dict(lat=-1.7483, lon=-51.4536, veg_class="EF")
        site_info_dict["FNS"] = dict(lat=-10.7618, lon=-62.3572, veg_class="GRA")
        site_info_dict["RJA"] = dict(lat=-10.078, lon=-61.9331, veg_class="EF")
        site_info_dict["BAN"] = dict(
            lat=-9.824416667, lon=-50.1591111, veg_class="EF"
        )  # transitional forest
        site_info_dict["PDG"] = dict(lat=-21.61947222, lon=-47.6498889, veg_class="SH")
        self.lat = site_info_dict[site_name]["lat"]
        self.lon = site_info_dict[site_name]["lon"]

    def add_tower_data(self):
        if self.vars == "all":
            idata = pd.read_csv(self.data_path, delim_whitespace=True, skiprows=[1])
        else:
            idata = pd.read_csv(
                self.data_path,
                usecols=lambda x: x in self.vars,
                delim_whitespace=True,
                skiprows=[1],
            )
        idata.rename({self.ssrd_key: "ssrd", self.t2m_key: "t2m"}, inplace=True, axis=1)
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
        # tzw = tzwhere.tzwhere()
        # timezone_str = tzw.tzNameAt(self.lat, self.lon)
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse(
            "200001010000"
        )  # pick a date that is definitely standard time and not DST
        datetime_u = []
        for i, row in idata.iterrows():
            datetime_u.append(
                datetime(int(row["Year_LBAMIP"]), 1, 1)
                + timedelta(row["DoY_LBAMIP"] - 1)
                + timedelta(hours=row["Hour_LBAMIP"])
                - timezone.utcoffset(dt)
            )
        datetime_u = np.array(datetime_u)
        idata["datetime_utc"] = datetime_u
        if (self.tstart is not None) & (self.tstop is not None):
            mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
            flux_data = idata[mask]
        else:
            flux_data = idata
        this_len = len(flux_data)
        if this_len < 2:
            print("No data for {} in given time range".format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print("Data only available for the following years {}".format(years))
            return False
        else:
            self.flux_data = flux_data
            if self.t2m_key is not None:
                self.flux_data["t2m"] = self.flux_data["t2m"] - 273.15
            return True


class ameri_fluxnet(flux_tower_data):

    def __init__(
        self,
        data_path,
        ssrd_key=None,
        t2m_key=None,
        use_vars=None,
        t_start=None,
        t_stop=None,
    ):

        site_name = (
            glob.glob(os.path.join(data_path, "*.csv"))[0]
            .split("/")[-1]
            .split("AMF_")[1]
            .split("_")[0]
        )
        self.data_path = data_path

        super().__init__(t_start, t_stop, ssrd_key, t2m_key, site_name)
        if (use_vars is None) | (use_vars is "all"):
            self.vars = "all"
        else:
            self.vars = use_vars
        idat_info = pd.read_excel(glob.glob(os.path.join(self.data_path, "*.xlsx"))[0])
        self.lat = float(
            idat_info[idat_info["VARIABLE"] == "LOCATION_LAT"]["DATAVALUE"]
        )
        self.lon = float(
            idat_info[idat_info["VARIABLE"] == "LOCATION_LONG"]["DATAVALUE"]
        )
        self.land_cover_type = str(
            idat_info.loc[idat_info["VARIABLE"] == "IGBP"]["DATAVALUE"]
        )
        return

    def add_tower_data(self):
        if self.vars is "all":
            idata = pd.read_csv(
                glob.glob(os.path.join(self.data_path, "*.csv"))[0], skiprows=2
            )
        else:
            idata = pd.read_csv(
                glob.glob(os.path.join(self.data_path, "*.csv"))[0],
                skiprows=2,
                usecols=lambda x: x in self.vars,
            )
        idata.rename({self.ssrd_key: "ssrd", self.t2m_key: "t2m"}, inplace=True, axis=1)
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
        # tzw = tzwhere.tzwhere()
        # timezone_str = tzw.tzNameAt(self.lat, self.lon)
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse(
            "200001010000"
        )  # pick a date that is definitely standard time and not DST
        datetime_u = []
        for i, row in idata.iterrows():
            datetime_u.append(
                parser.parse(str(int(row["TIMESTAMP_END"]))) - timezone.utcoffset(dt)
            )
        datetime_u = np.array(datetime_u)
        idata["datetime_utc"] = datetime_u
        if (self.tstart is not None) & (self.tstop is not None):
            mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
            flux_data = idata[mask]
        else:
            flux_data = idata
        this_len = len(flux_data)
        if this_len < 2:
            print("No data for {} in given time range".format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print("Data only available for the following years {}".format(years))
            return False
        else:
            self.flux_data = flux_data
            return True


class fluxnet(flux_tower_data):

    def __init__(
        self,
        data_path,
        ssrd_key=None,
        t2m_key=None,
        use_vars=None,
        t_start=None,
        t_stop=None,
    ):

        site_name = data_path.split("FLX_")[1].split("_")[0]
        self.data_path = data_path

        super().__init__(t_start, t_stop, ssrd_key, t2m_key, site_name)

        if use_vars is None:
            self.vars = [
                "NEE_CUT_REF",
                "NEE_VUT_REF",
                "NEE_CUT_REF_QC",
                "NEE_VUT_REF_QC",
                "GPP_NT_VUT_REF",
                "GPP_NT_CUT_REF",
                "GPP_DT_VUT_REF",
                "GPP_DT_CUT_REF",
                "TIMESTAMP_START",
                "TIMESTAMP_END",
                "WD",
                "WS",
                "SW_IN_F",
                "TA_F",
                "USTAR",
                "RECO_NT_VUT_REF",
                "RECO_DT_VUT_REF",
                "TA_F_QC",
                "SW_IN_F_QC",
            ]
        elif use_vars is "all":
            self.vars = "all"
        else:
            self.vars = use_vars

        site_info = pd.read_pickle(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "fluxnet_info",
                "fluxnet_sites.pkl",
            )
        )
        self.lat = site_info.loc[site_info["SITE_ID"] == site_name]["lat"].values
        self.lon = site_info.loc[site_info["SITE_ID"] == site_name]["long"].values
        self.land_cover_type = site_info.loc[site_info["SITE_ID"] == site_name][
            "IGBP"
        ].values
        return

    def add_tower_data(self):
        if self.vars == "all":
            idata = pd.read_csv(self.data_path)
        else:
            idata = pd.read_csv(self.data_path, usecols=lambda x: x in self.vars)
        idata.rename({self.ssrd_key: "ssrd", self.t2m_key: "t2m"}, inplace=True, axis=1)
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
        # tzw = tzwhere.tzwhere()
        # timezone_str = tzw.tzNameAt(self.lat, self.lon)
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse(
            "200001010000"
        )  # pick a date that is definitely standard time and not DST
        datetime_u = []
        for i, row in idata.iterrows():
            datetime_u.append(
                parser.parse(str(int(row["TIMESTAMP_END"]))) - timezone.utcoffset(dt)
            )
        datetime_u = np.array(datetime_u)
        idata["datetime_utc"] = datetime_u
        if (self.tstart is not None) & (self.tstop is not None):
            mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
            flux_data = idata[mask]
        else:
            flux_data = idata
        this_len = len(flux_data)
        if this_len < 2:
            print("No data for {} in given time range".format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print("Data only available for the following years {}".format(years))
            return False
        else:
            self.flux_data = flux_data
            return True


class icos(flux_tower_data):
    def __init__(
        self,
        data_path,
        ssrd_key=None,
        t2m_key=None,
        use_vars=None,
        t_start=None,
        t_stop=None,
    ):

        self.data_path = data_path
        site_name = data_path.split("ICOSETC_")[1].split("_")[0]

        super().__init__(t_start, t_stop, ssrd_key, t2m_key, site_name)

        if use_vars is None:
            self.vars = variables = [
                "NEE_CUT_REF",
                "NEE_VUT_REF",
                "NEE_CUT_REF_QC",
                "NEE_VUT_REF_QC",
                "GPP_NT_VUT_REF",
                "GPP_NT_CUT_REF",
                "GPP_DT_VUT_REF",
                "GPP_DT_CUT_REF",
                "TIMESTAMP_START",
                "TIMESTAMP_END",
                "WD",
                "WS",
                "SW_IN_F",
                "TA_F",
                "USTAR",
                "RECO_NT_VUT_REF",
                "RECO_DT_VUT_REF",
                "TA_F_QC",
                "SW_IN_F_QC",
            ]
        elif use_vars is "all":
            self.vars = "all"
        else:
            self.vars = use_vars

        site_info = pd.read_csv(
            os.path.join(
                os.path.dirname(self.data_path),
                "ICOSETC_{}_SITEINFO_L2.csv".format(self.site_name),
            ),
            on_bad_lines="skip",
        )
        self.land_cover_type = site_info.loc[site_info["VARIABLE"] == "IGBP"][
            "DATAVALUE"
        ].values[0]
        self.lat = float(
            site_info.loc[site_info["VARIABLE"] == "LOCATION_LAT"]["DATAVALUE"].values
        )
        self.lon = float(
            site_info.loc[site_info["VARIABLE"] == "LOCATION_LONG"]["DATAVALUE"].values
        )

        return

    def add_tower_data(self):
        if self.vars is "all":
            idata = pd.read_csv(self.data_path, on_bad_lines="skip")
        else:
            idata = pd.read_csv(
                self.data_path, usecols=lambda x: x in self.vars, on_bad_lines="skip"
            )
        idata.rename({self.ssrd_key: "ssrd", self.t2m_key: "t2m"}, inplace=True, axis=1)
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
        # tzw = tzwhere.tzwhere()
        # timezone_str = tzw.tzNameAt(self.lat, self.lon)
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse(
            "200001010000"
        )  # pick a date that is definitely standard time and not DST
        datetime_u = []
        for i, row in idata.iterrows():
            datetime_u.append(
                parser.parse(str(int(row["TIMESTAMP_END"]))) - timezone.utcoffset(dt)
            )
        datetime_u = np.array(datetime_u)
        idata["datetime_utc"] = datetime_u
        if (self.tstart is not None) & (self.tstop is not None):
            mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
            flux_data = idata[mask]
        else:
            flux_data = idata
        this_len = len(flux_data)

        if this_len < 2:
            print("No data for {} in given time range".format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print("Data only available for the following years {}".format(years))
            return False
        else:
            self.flux_data = flux_data
            return True
