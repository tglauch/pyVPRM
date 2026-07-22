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
from pyVPRM.lib.functions import get_eth_canopy_height, get_elevation_copernicus_dem

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
        if use_vars == "all":
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

    def add_tower_data(self, missing_displacement_heights_dict = None):
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
        if (use_vars is None) | (use_vars == "all"):
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
        if self.vars == "all":
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
        elif use_vars == "all":
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

    def add_tower_data(self, missing_displacement_heights_dict = None):
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
        icos_2025=True,
        need_footprint_variables=False,
    ):

        self.data_path = data_path
        site_name = data_path.split("ICOSETC_")[1].split("_")[0]
        self.need_footprint_variables = need_footprint_variables

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
        elif use_vars == "all":
            self.vars = "all"
        else:
            self.vars = use_vars
        self.icos_2025 = icos_2025

        if self.icos_2025:
            self.site_info = pd.read_csv(
                os.path.join(
                    os.path.dirname(self.data_path).replace('FLUXNET_HH_L2', 'ARCHIVE_L2'),
                    "ICOSETC_{}_SITEINFO_L2.csv".format(self.site_name),
                ),
                on_bad_lines="skip",
            )
            self.footprint = pd.read_csv(
                os.path.join(
                    os.path.dirname(self.data_path).replace('FLUXNET_HH_L2', 'ARCHIVE_L2'),
                    "ICOSETC_{}_FLUXES_L2.csv".format(self.site_name),
                ),
                on_bad_lines="skip",
            )
            if 'FETCH_50' in self.footprint.keys():
                self.fetch50 = np.percentile(self.footprint['FETCH_50'][self.footprint['FETCH_50']>0], 90)
            else:
                self.fetch50 = None

            if 'FETCH_90' in self.footprint.keys():
                self.fetch90 = np.percentile(self.footprint['FETCH_90'][self.footprint['FETCH_90']>0], 50)
            else:
                self.fetch90 = None
        else:
            self.site_info = pd.read_csv(
                os.path.join(
                    os.path.dirname(self.data_path),
                    "ICOSETC_{}_SITEINFO_L2.csv".format(self.site_name),
                ),
                on_bad_lines="skip",
            )
            self.fetch90 = None # Not yet implemented
            
        self.land_cover_type = self.site_info.loc[self.site_info["VARIABLE"] == "IGBP"][
            "DATAVALUE"
        ].values[0]
        self.lat = float(
            self.site_info.loc[self.site_info["VARIABLE"] == "LOCATION_LAT"]["DATAVALUE"].values
        )
        self.lon = float(
            self.site_info.loc[self.site_info["VARIABLE"] == "LOCATION_LONG"]["DATAVALUE"].values
        )
        try:
            self.elev = float(self.site_info.loc[self.site_info['VARIABLE']=='LOCATION_ELEV']['DATAVALUE'].values[0])
        except:
            print('failed to load elevation. continue.')


        return

    def add_tower_data(self, missing_displacement_heights_dict = None):
        if self.vars == "all":
            idata = pd.read_csv(self.data_path, on_bad_lines="skip")
        else:
            idata = pd.read_csv(
                self.data_path, usecols=lambda x: x in self.vars, on_bad_lines="skip"
            )
        if self.need_footprint_variables:
            #get measurement height
            if self.icos_2025:
                variable_info = pd.read_csv(os.path.join(os.path.dirname(self.data_path).replace('FLUXNET_HH_L2', 'ARCHIVE_L2'),
                                                         'ICOSETC_{}_VARINFO_FLUXNET_HH_L2.csv'.format(self.site_name)),
                                    on_bad_lines='skip')
            else:
                variable_info = pd.read_csv(os.path.join(os.path.dirname(self.data_path),  'ICOSETC_{}_VARINFO_FLUXNET_HH_L2.csv'.format(self.site_name)),
                                    on_bad_lines='skip')

            #get groups of data that describe NEE_VUT_REF parameter
            group_ids_nee_entries = variable_info['GROUP_ID'].iloc[variable_info.index[variable_info['DATAVALUE'] == 'NEE_VUT_REF']].values
            #get indices of the groups
            indices_group_id = variable_info.index[variable_info['GROUP_ID'].isin(group_ids_nee_entries)]
            #get indices that describe heigth parameter
            indices_height_parameter = variable_info.index[variable_info['VARIABLE'] == 'VAR_INFO_HEIGHT']
            #get wanted measurement heights
            indices_heights = indices_group_id.intersection(indices_height_parameter)
            heights = variable_info['DATAVALUE'][indices_heights].values
            
            #get indices that describe measured times of the heights
            indices_parameter_date = variable_info.index[variable_info['VARIABLE'] == 'VAR_INFO_DATE']
            #get wanted dates
            indices_dates = indices_group_id.intersection(indices_parameter_date)
            dates_height_measurement = variable_info['DATAVALUE'][indices_dates].values
            #print('Is sorted?', (dates_height_measurement == np.sort(dates_height_measurement)).all())
            if not (dates_height_measurement == np.sort(dates_height_measurement)).all():
                sorting_indices = np.argsort(dates_height_measurement)
                dates_height_measurement = dates_height_measurement[sorting_indices]
                heights = heights[sorting_indices]
            
            #add height parameter
            idata['z_measurement'] = float(heights[0])
            for idx_height, height in enumerate(heights):
                mask = (idata['TIMESTAMP_END'] >= float(dates_height_measurement[idx_height]))
                idata.loc[mask, 'z_measurement'] = float(heights[idx_height])
            #float(variable_info['DATAVALUE'].iloc[variable_info.index[variable_info['DATAVALUE']=='NEE_VUT_REF']+1].values[0])
            #print(idata['z_measurement'])
            self.mean_z_measurement = idata['z_measurement'].mean()

            #get FLUXES file data
            wanted_fluxes_variables = ['TIMESTAMP_END', 'MO_LENGTH',' PBLH','ZL' ,'V_SIGMA', 'FETCH_MAX', 'FETCH_70', 'FETCH_80', 'FETCH_90']
            if self.icos_2025:
                fluxes_file_data = pd.read_csv(os.path.join(os.path.dirname(self.data_path).replace('FLUXNET_HH_L2', 'ARCHIVE_L2'),
                                                            'ICOSETC_{}_FLUXES_L2.csv'.format(self.site_name)),
                                               usecols=lambda x: x in wanted_fluxes_variables, on_bad_lines='skip')
            else:
                fluxes_file_data = pd.read_csv(os.path.join(os.path.dirname(self.data_path),
                                                            'ICOSETC_{}_FLUXES_L2.csv'.format(self.site_name)),
                                               usecols=lambda x: x in wanted_fluxes_variables, on_bad_lines='skip')
            print('loaded from fluxes file:', fluxes_file_data.columns)
            print('not available in fluxes file:', [col for col in wanted_fluxes_variables if col not in fluxes_file_data.columns])
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

        if self.need_footprint_variables:
            #get utc datetime from local time for FLUXES data
            fluxes_file_data['datetime_utc'] = self.get_utc_times(fluxes_file_data)
            mask = np.ones(len(fluxes_file_data), dtype=bool)
            if self.tstart is not None:
                mask &= (fluxes_file_data['datetime_utc'] >= self.tstart).values
            if self.tstop is not None:
                mask &= (fluxes_file_data['datetime_utc'] < self.tstop).values
            fluxes_file_data = fluxes_file_data[mask]
            #merge FLUXNET and FLUXES data  
            flux_data = flux_data.merge(fluxes_file_data, how='inner', on=['datetime_utc', 'TIMESTAMP_END'])

        this_len = len(flux_data)
        if self.need_footprint_variables:
            if 'MO_LENGTH' in flux_data.columns:
                flux_data['z_footprint'] = np.nan 
                for idx in range(len(dates_height_measurement)+1):
                    if idx == 0:
                        lower_lim = 0
                        upper_lim = float(dates_height_measurement[idx])
                    elif idx == len(dates_height_measurement):
                        upper_lim = 1000000000000000000
                        lower_lim = float(dates_height_measurement[idx-1])
                    else:
                        lower_lim = float(dates_height_measurement[idx-1])
                        upper_lim = float(dates_height_measurement[idx])
                    mask = (flux_data['TIMESTAMP_END'] >= lower_lim) & (flux_data['TIMESTAMP_END'] <= upper_lim)
                    #print(mask)
                    try:
                        
                        flux_data.loc[mask, 'z_footprint'] = np.max([0, np.nanmean(np.where(flux_data.loc[mask, 'ZL']>-9999, flux_data.loc[mask,'ZL'], np.nan)*np.where(flux_data.loc[mask,'MO_LENGTH']>-9999, flux_data.loc[mask,'MO_LENGTH'], np.nan))])
                    except RuntimeWarning: 
                        print('no valid data between', lower_lim, upper_lim)
                #print(flux_data['z_footprint'])
            
            elif missing_displacement_heights_dict is not None and self.site_name in missing_displacement_heights_dict.keys(): 
                flux_data['z_footprint'] = flux_data['z_measurement'] - missing_displacement_heights_dict[self.site_name]
            else: 
                flux_data['z_footprint'] = flux_data['z_measurement']
            flux_data['z_displacement'] = flux_data['z_measurement'] - flux_data['z_footprint']
            self.mean_z_footprint = flux_data['z_footprint'].mean()
            self.mean_z_displacement = flux_data['z_displacement'].mean()
            print('sensor height above ground:', self.mean_z_measurement)
            print('distance footprint/u(0) and sensor:', self.mean_z_footprint)
            print('displacement height above ground:', self.mean_z_displacement)

        if this_len < 2:
            print("No data for {} in given time range".format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print("Data only available for the following years {}".format(years))
            return False
        else:
            self.flux_data = flux_data
            return True

    def get_utc_times(self, dataframe):
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse('200001010000') # pick a date that is definitely standard time and not DST 
        utc_datetime = []
        for i, row in dataframe.iterrows():
            utc_datetime.append(parser.parse(str(int(row['TIMESTAMP_END'])))  -  timezone.utcoffset(dt))
        return np.array(utc_datetime)

class fluxnet_shuttle(flux_tower_data):
    def __init__(
        self,
        data_path,
        ssrd_key=None,
        t2m_key=None,
        use_vars=None,
        t_start=None,
        t_stop=None,
        need_footprint_variables=False,
        temp_folder='./',
    ):
        self.data_path = data_path
        site_name = data_path.split('/')[-1].split('_')[1]
        self.need_footprint_variables = need_footprint_variables

        super().__init__(t_start, t_stop, ssrd_key, t2m_key, site_name)

        if use_vars in (None, "all"):
            self.vars = "all"
        else:
            self.vars = use_vars

        self.temp_folder = temp_folder
        self.site_info = pd.read_csv(
            self.data_path.replace('FLUXMET_HH', 'BIF'),
            on_bad_lines="skip",
        )

        self.land_cover_type = self.site_info.loc[
            self.site_info["VARIABLE"] == "IGBP", "DATAVALUE"
        ].values[0]
        self.lat = float(
            self.site_info.loc[self.site_info["VARIABLE"] == "LOCATION_LAT", "DATAVALUE"].values[0]
        )
        self.lon = float(
            self.site_info.loc[self.site_info["VARIABLE"] == "LOCATION_LONG", "DATAVALUE"].values[0]
        )
        try:
            self.elev = float(
                self.site_info.loc[self.site_info['VARIABLE'] == 'LOCATION_ELEV', 'DATAVALUE'].values[0]
            )
        except (IndexError, KeyError, ValueError):
            self.elev = get_elevation_copernicus_dem(self.lon, self.lat)

        # Resolve the site's timezone once and cache it.
        tf = TimezoneFinder()
        self.timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
        self.standard_utc_offset = self._get_standard_utc_offset(self.timezone_str)
        return

    @staticmethod
    def _get_standard_utc_offset(timezone_str):
        """
        Return the fixed UTC offset (timedelta) for LOCAL STANDARD TIME
        (i.e. non-DST) in the given IANA timezone.

        FLUXNET/AmeriFlux data is documented to be reported in local
        standard time year-round -- DST is never applied, regardless of
        hemisphere or time of year. So this must NOT depend on picking a
        date that happens to fall in standard time (that breaks for
        Southern Hemisphere sites, where e.g. January is DST). Forcing
        is_dst=False makes pytz return the standard-time offset
        unconditionally.
        """
        tz = pytz.timezone(timezone_str)
        reference_naive = datetime(2000, 1, 1, 12, 0, 0)
        localized = tz.localize(reference_naive, is_dst=False)
        return localized.utcoffset()

    def local_to_utc(self, timestamp_end):
        """
        Vectorized conversion of FLUXNET TIMESTAMP_END values (local
        STANDARD time, formatted YYYYMMDDHHMM) to naive UTC datetimes.

        A single fixed offset is correct here (and is *not* a
        simplification) because the source data has no DST component by
        definition -- see FLUXNET timestamp/timezone convention docs.
        """
        local_naive = pd.to_datetime(
            timestamp_end.astype(int).astype(str), format="%Y%m%d%H%M"
        )
        return (local_naive - self.standard_utc_offset).values

    def get_utc_times(self, dataframe):
        return self.local_to_utc(dataframe["TIMESTAMP_END"])

    def add_tower_data(self, met_inst=None,
                       canopy_height_path='.',
                       measurement_height=None):
        self.met_inst = met_inst

        if self.vars == "all":
            idata = pd.read_csv(self.data_path, on_bad_lines="skip")
        else:
            idata = pd.read_csv(
                self.data_path, usecols=lambda x: x in self.vars, on_bad_lines="skip"
            )

        if self.need_footprint_variables:
            variable_info = pd.read_csv(
                self.data_path.replace('FLUXMET', 'BIFVARINFO'), on_bad_lines='skip'
            )

            if measurement_height is None:
                group_ids_nee_entries = variable_info.loc[
                    variable_info['DATAVALUE'] == 'NEE_VUT_REF', 'GROUP_ID'
                ].values
                indices_group_id = variable_info.index[variable_info['GROUP_ID'].isin(group_ids_nee_entries)]
                indices_height_parameter = variable_info.index[variable_info['VARIABLE'] == 'VAR_INFO_HEIGHT']
                indices_heights = indices_group_id.intersection(indices_height_parameter)
                heights = variable_info['DATAVALUE'][indices_heights].values
    
                indices_parameter_date = variable_info.index[variable_info['VARIABLE'] == 'VAR_INFO_DATE']
                indices_dates = indices_group_id.intersection(indices_parameter_date)
                dates_height_measurement = variable_info['DATAVALUE'][indices_dates].values
                if not (dates_height_measurement == np.sort(dates_height_measurement)).all():
                    sorting_indices = np.argsort(dates_height_measurement)
                    dates_height_measurement = dates_height_measurement[sorting_indices]
                    heights = heights[sorting_indices]
    
                idata['z_measurement'] = float(heights[0])
                for idx_height, height in enumerate(heights):
                    mask = idata['TIMESTAMP_END'] >= float(dates_height_measurement[idx_height])
                    idata.loc[mask, 'z_measurement'] = float(heights[idx_height])
            else:
                idata['z_measurement'] = float(measurement_height)
            self.mean_z_measurement = idata['z_measurement'].mean()

        idata.rename({self.ssrd_key: "ssrd", self.t2m_key: "t2m"}, inplace=True, axis=1)

        # single vectorized local-standard-time -> UTC conversion, no per-row loop
        idata["datetime_utc"] = self.local_to_utc(idata["TIMESTAMP_END"])

        if self.need_footprint_variables:
            canopy_height = get_eth_canopy_height(self.lat, self.lon, basepath=canopy_height_path)
            print('Land Cover Type', self.land_cover_type)
            if self.land_cover_type in ['ENF', 'EBF', 'DBF', 'DNF', 'MF']:
                idata['z_footprint'] = 0.68 * canopy_height
            else:
                idata['z_footprint'] = idata['z_measurement']
            idata['z_displacement'] = idata['z_measurement'] - idata['z_footprint']
            self.mean_z_footprint = idata['z_footprint'].mean()
            self.mean_z_displacement = idata['z_displacement'].mean()
            print('sensor height above ground:', self.mean_z_measurement)
            print('displacement height above ground:', self.mean_z_displacement)

            rho = idata['PA_F'] * 1000 / (287.05 * (idata['TA_F_MDS'] + 273.15))
            MO_LENGTH = -(rho * 1004 * (idata['TA_F_MDS'] + 273.15) * idata['USTAR'] ** 3) / (
                0.4 * 9.81 * idata['H_F_MDS']
            )
            MO_LENGTH = MO_LENGTH.where((MO_LENGTH > -1000) & (MO_LENGTH < 1000))
            idata['MO_LENGTH'] = MO_LENGTH
            idata['ZL'] = idata['z_displacement'] / idata['MO_LENGTH']  

        mask = np.ones(len(idata), dtype=bool)
        if self.tstart is not None:
            mask &= (idata['datetime_utc'] >= self.tstart).values
        if self.tstop is not None:
            mask &= (idata['datetime_utc'] < self.tstop).values
        flux_data= idata[mask]

        print(flux_data['datetime_utc'])
        
        # if (self.tstart is not None) and (self.tstop is not None):
        #     mask = (idata['datetime_utc'] >= self.tstart) & (idata['datetime_utc'] <= self.tstop)
        #     flux_data = idata[mask]
        # else:
        #     flux_data = idata

        if self.met_inst is not None:
            print(self.lat, self.lon)
            add_mets = self.met_inst.get_data(lonlat=(self.lon, self.lat),
                                   times=flux_data['datetime_utc'].values)
            flux_data = flux_data.reset_index(drop=True)
            for var in add_mets.data_vars:
                print(add_mets[var].values)
                flux_data[var] = add_mets[var].values

        if len(flux_data) < 2:
            print("No data for {} in given time range".format(self.site_name))
            years = pd.DatetimeIndex(idata['datetime_utc']).year.unique()
            print("Data only available for the following years {}".format(years.tolist()))
            return False

        flux_data.rename(columns={'blh': 'PBLH'}, inplace=True)
        self.flux_data = flux_data
        print('PBLS', np.nanmax(flux_data['PBLH']))    
        return True

