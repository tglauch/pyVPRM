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

    def __init__(self, year, month, day, hour, bpath, mpi=False, keys=[]):
        super().__init__()
        # Init with year, month, day, hour and the required era5 keys as given in the
        #  keys_dict above
        self.in_era5_grid = True
        self.this_month = 0
        self.this_year = 0
        self.bpath = bpath
        self.ds_in_t = None
        self.regridder = None
        self.mpi = mpi
        self.change_date(year, month, day, hour)

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
                self.ds_in_t.to_netcdf(src_temp_path)
                t_ds_out.to_netcdf(dest_temp_path)
                cmd = "ESMF_RegridWeightGen --source {} --destination {} --weight {} -m bilinear --64bit_offset  --extrap_method nearestd  --no_log".format(
                    src_temp_path, dest_temp_path, weights
                )
                if self.mpi:
                    cmd = "mpirun -np {} ".format(n_cpus) + cmd
                logger.info(cmd)
                os.system(cmd)
                os.remove(src_temp_path)
                os.remove(dest_temp_path)

            self.regridder = xe.Regridder(
                self.data, t_ds_out, "bilinear", weights=weights, reuse_weights=True
            )
        self.data = self.regridder(self.data)
        self.in_era5_grid = False

    def get_data(self, lonlat=None, key=None):
        # Return ERA5 data for lonlat if lonlat is not None else return all data.
        # Pick a specific key if key is not None. Return as xarray dataset
        dt = np.datetime64(
            "{}-{:02d}-{:02d}T{:02d}:00:00.000000".format(
                self.year, self.month, self.day, self.hour
            )
        )
        if key is not None:
            tmp = self.data.sel({"time": dt})[key]
        else:
            tmp = self.data.sel({"time": dt})
        if lonlat is None:
            return tmp
        else:
            lon = lonlat[0]
            if isinstance(lon, list) | isinstance(lon, np.ndarray):
                return tmp.interp(lon=("z", lon), lat=("z", lonlat[1]), method="linear")
            else:
                lon = lonlat[0]
                return tmp.interp(lon=lon, lat=lonlat[1])
        return self.data

    def _init_data_for_day(self):
        if (self.this_month != self.month) | (self.this_year != self.year):
            self.data = xr.open_dataset(
                os.path.join(self.bpath, "{}_{}.nc".format(self.year, self.month))
            )
            self.this_month = self.month
            self.this_year = self.year
            self.in_era5_grid = True

        if self.ds_in_t is None:
            self.ds_in_t = xr.Dataset(
                {
                    "lat": (
                        ["lat"],
                        self.data["lat"].values,
                        {"units": "degrees_north"},
                    ),
                    "lon": (
                        ["lon"],
                        self.data["lon"].values,
                        {"units": "degrees_east"},
                    ),
                }
            )
            self.ds_in_t = self.ds_in_t.set_coords(["lon", "lat"])
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
