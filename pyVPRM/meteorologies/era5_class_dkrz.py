import xarray as xr
import glob
import os
import time
import numpy as np
from dateutil import parser
from scipy.interpolate import interp2d
import pygrib
import copy
import uuid
import datetime
from pyVPRM.meteorologies.met_base_class import met_data_handler_base
from loguru import logger

map_function = lambda lon: (lon - 360) if (lon > 180) else lon

bpaths = {
    "sf00": "/pool/data/ERA5/E*/sf/an/{}",
    "sf12": "/pool/data/ERA5/E*/sf/fc/{}",
    "pl00": "/pool/data/ERA5/E*/pl/an/{}",
    "ml00": "/pool/data/ERA5/E*/ml/an/{}",
}
# Check documentation under
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Spatialgrid

keys_dict = {
    "ssrd": [169, "sf12"],  # surface solar radiation downwards(J/m**2)
    "t2m": [167, "sf00"],  # temperature 2 m
    #             'slt': [43, 'sf00'], #soil type, not sure it's useful...MODIS should do better
    "sp": [134, "sf00"],  # surface pressure
    "tcc": [164, "sf00"],  # total cloud cover
    "stl1": [139, "sf00"],  # soil temperature level1
    "stl2": [170, "sf00"],  # soil temperature level2
    "stl3": [183, "sf00"],  # soil temperature level3
    "swvl1": [39, "sf00"],  # soil water level 1
    "swvl2": [40, "sf00"],  # soil water level 2
    "swvl3": [41, "sf00"],  # soil water level 3
    "swvl4": [42, "sf00"],  # soil water level 3
    "2d": [168, "sf00"],  # 2-m dewpoint temperature
    "tp": [228, "sf12"],  # total precipitation over given time
    #             'ssr': [176, 'sf12'], #net surface solar radiation (J/m**2)
    #             'str': [177, 'sf12'],#net surface thermal radiation (J/m**2)
    "src": [198, "sf00"],
    "q": [133, "ml00"],  # specific humidity (%)
    "e": [182, "sf12"],
    "sst": [34, "sf00"],
}


class met_data_handler(met_data_handler_base):
    """
    Class for using ERA5 data available on Levante's DKRZ cluster.
    """

    def __init__(self, year, month, day=None, hour=None, keys=[], timesteps="hourly"):
        super().__init__()
        if timesteps == "hourly":
            self.add_time_str = "1H"
        elif timesteps == "daily":
            self.add_time_str = "1D"
        elif timesteps == "monthly":
            self.add_time_str = "1M"
        self.file_handlers = dict()
        self.in_era5_grid = True
        self.regridder = None
        self.ds_in_t = None
        self.reg_lats = []
        self.reg_lons = []
        if len(keys) == 0:
            self.keys = keys_dict.keys()
        else:
            self.keys = keys
        self.change_date(year, month, day, hour)

    def _init_data_for_day(self):
        file_dict = dict()
        for c, key in enumerate(self.keys):
            t_dict = dict()
            bpath = bpaths[keys_dict[key][1]].format(self.add_time_str)
            e_id = keys_dict[key][0]
            if self.add_time_str == "1H":
                fname = os.path.join(
                    bpath,
                    "{:03d}/E*{}_1H_{}-{:02d}-{:02d}_{:03d}.grb".format(
                        e_id, keys_dict[key][1], self.year, self.month, self.day, e_id
                    ),
                )
            elif self.add_time_str == "1D":
                fname = os.path.join(
                    bpath,
                    "{:03d}/E*{}_1D_{}-{:02d}_{:03d}.grb".format(
                        e_id, keys_dict[key][1], self.year, self.month, e_id
                    ),
                )
            elif self.add_time_str == "1M":
                fname = os.path.join(
                    bpath,
                    "{:03d}/E*{}_1M_{}_{:03d}.grb".format(
                        e_id, keys_dict[key][1], self.year, e_id
                    ),
                )
            fname = glob.glob(fname)
            fname = [i for i in fname if "EL" not in i]
            if key in self.file_handlers.keys():
                self.file_handlers[key]["current"].close()
            t_dict["current"] = pygrib.open(fname[0])
            t_dict["current_name"] = list(fname)
            t_dict["current_ind"] = 0
            t_dict["type"] = keys_dict[key][1]
            self.file_handlers[key] = t_dict
            if not "sf" in keys_dict[key][1]:  # Not a surface value
                self.select_from_pressure_levels(key)
            if self.ds_in_t is None:
                lats, lons = t_dict["current"][1].latlons()
                self.ds_in_t = xr.Dataset(
                    {
                        "lat": (["lat"], lats[:, 0], {"units": "degrees_north"}),
                        "lon": (
                            ["lon"],
                            [map_function(i) for i in lons[0]],
                            {"units": "degrees_east"},
                        ),
                    }
                )
                self.ds_in_t = self.ds_in_t.set_coords(["lon", "lat"])

    def _load_data_for_hour(self):
        # Caution: The date as argument corresponds to the END of the ERA5 integration time.
        data_dict = dict()
        if self.add_time_str == "1H":
            for key in self.keys:
                data_dict[key] = (
                    ["lat", "lon"],
                    self.file_handlers[key]["current"][int(self.hour + 1)].values,
                )

        elif self.add_time_str == "1D":
            for key in self.keys:
                data_dict[key] = (
                    ["lat", "lon"],
                    self.file_handlers[key]["current"][int(self.day)].values,
                )

        elif self.add_time_str == "1M":
            for key in self.keys:
                if self.month not in [
                    self.file_handlers[key]["current"][t + 1].month
                    for t in range(self.file_handlers[key]["current"].messages)
                ]:
                    for i in self.file_handlers[key]["current_name"]:
                        tmp = pygrib.open(i)
                        if self.month in [
                            tmp[t + 1].month for t in range(tmp.messages)
                        ]:
                            self.file_handlers[key]["current"] = tmp
                            break

                if self.month not in [
                    self.file_handlers[key]["current"][t + 1].month
                    for t in range(self.file_handlers[key]["current"].messages)
                ]:
                    logger.info("No data for given month")
                    return

                for c in range(self.file_handlers[key]["current"].messages):
                    if self.file_handlers[key]["current"][c + 1].month == self.month:
                        break
                data_dict[key] = (
                    ["lat", "lon"],
                    self.file_handlers[key]["current"][c + 1].values,
                )

        self.ds_out = copy.deepcopy(self.ds_in_t)
        self.ds_out = self.ds_out.assign(data_dict)
        self.ds_out = self.ds_out.sortby("lon")
        self.in_era5_grid = True

    def get_all_interpolators(self, day, hour):
        ret_dict = dict()
        for key in self.keys:
            ret_dict[key] = self.get_interpolator()
        return ret_dict

    def regrid(
        self,
        lats=None,
        lons=None,
        dataset=None,
        n_cpus=1,
        weights=None,
        overwrite_regridder=False,
    ):

        import xesmf as xe

        if self.in_era5_grid is False:
            return

        if (self.regridder is None) | (overwrite_regridder):
            if (lats is not None) and (lons is not None):
                t_ds_out = xr.Dataset(
                    {
                        "lat": (["lat"], lats, {"units": "degrees_north"}),
                        "lon": (["lon"], lons, {"units": "degrees_east"}),
                    }
                )
                t_ds_out = t_ds_out.set_coords(["lon", "lat"])
                self.reg_lats = lats
                self.reg_lons = lons
            else:
                t_ds_out = dataset

            if (weights is not None) & os.path.exists(str(weights)):
                logger.info("Load weights from {}".format(weights))
            else:
                bfolder = os.path.dirname(weights)
                src_temp_path = os.path.join(bfolder, "{}.nc".format(str(uuid.uuid4())))
                dest_temp_path = os.path.join(
                    bfolder, "{}.nc".format(str(uuid.uuid4()))
                )
                self.ds_out.to_netcdf(src_temp_path)
                t_ds_out.to_netcdf(dest_temp_path)
                cmd = "mpirun -np {}  ESMF_RegridWeightGen --source {} --destination {} --weight {} -m bilinear --64bit_offset  --extrap_method nearestd  --no_log".format(
                    n_cpus, src_temp_path, dest_temp_path, weights
                )
                logger.info(cmd)
                os.system(cmd)
                os.remove(src_temp_path)
                os.remove(dest_temp_path)

            self.regridder = xe.Regridder(
                self.ds_in_t, t_ds_out, "bilinear", weights=weights, reuse_weights=True
            )
        self.ds_out = self.regridder(self.ds_out)
        self.in_era5_grid = False
        return

    def select_from_pressure_levels(self, key):
        selection_args = {}
        if key == "r":
            selection_args["level"] = 975
            selection_args["typeOfLevel"] = "isobaricInhPa"
        elif key == "q":
            selection_args["level"] = 137
            selection_args["typeOfLevel"] = "hybrid"
        try:
            self.file_handlers[key]["current"] = self.file_handlers[key][
                "current"
            ].select(**selection_args)
        except Exception as e:
            logger.info("No era5 data found for {}".format(selection_args))
            return None

    def get_data(self, lonlat=None, key=None):
        if key is not None:
            tmp = self.ds_out[key]
        else:
            tmp = self.ds_out
        if lonlat is None:
            return tmp
        else:
            lon = lonlat[0]
            if isinstance(lon, list) | isinstance(lon, np.ndarray):
                return tmp.interp(lon=("z", lon), lat=("z", lonlat[1]), method="linear")
            else:
                lon = lonlat[0]
                return tmp.interp(lon=lon, lat=lonlat[1])


if __name__ == "__main__":
    year = "2000"
    month = 2
    day = 20
    hour = 5  # UTC hour
    position = {"lat": 50.30493, "long": 5.99812}
    era5_handler = met_data_handler(year, month, day)
    era5_handler.change_date(hour)
    ret = era5_handler.get_data()
    logger.info(ret)
