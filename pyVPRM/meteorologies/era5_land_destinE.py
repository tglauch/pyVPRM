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

map_function_inv = lambda lon: (lon + 360) if (lon < 0) else lon

class met_data_handler(met_data_handler_base):
    """
    Class for using ERA5 data available on Levante's DKRZ cluster.
    """

    def __init__(self, year, month, day=None, hour=None,
                 PAT=None, keys=[], lat_slice=None, lon_slice=None,
                 mpi=False):
        if PAT is None:
            print('Need to set the access token. Check https://platform.destine.eu/.')
            return
        super().__init__()
        self.PAT = PAT
        self.in_era5_grid = True
        self.regridder = None
        self.rearranged = False
        self.lat_slice = lat_slice
        self.lon_slice = lon_slice
        self.mpi = mpi
        if self.lon_slice is not None:
            self.lon_slice[0] = map_function_inv(self.lon_slice[0])
            self.lon_slice[1] = map_function_inv(self.lon_slice[1])
        self.keys = keys
        self.load_ds()
        self.change_date(year, month, day, hour)

    def load_ds(self):
        self.ds = xr.open_dataset(
            "https://edh:{}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr".format(self.PAT),
            chunks={},
            engine="zarr",
        ).astype("float32").rename({'longitude':'lon',
                                    'latitude':'lat'})
        if self.keys != []:
            self.ds = self.ds[self.keys]
        return
    
    def _init_data_for_day(self):
      return

    def _load_data_for_hour(self):
        if self.ds is None:
            print('No dataset loaded from destination Earth')
            return
        # Caution: The date as argument corresponds to the END of the ERA5 integration time.
        sel_dict = {"valid_time": '{}-{}-{} {}:00:00'.format(self.year, self.month,
                                                             self.day, self.hour)}
        if self.lat_slice is not None:
            sel_dict['lat'] = slice(self.lat_slice[1], self.lat_slice[0])
        if self.lon_slice is not None:
            sel_dict['lon'] = slice(self.lon_slice[0], self.lon_slice[1])
        self.ds_out = self.ds.sel(sel_dict).compute()
        self.rearranged = False
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

        self.rearrange_lons()

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
                cmd = "ESMF_RegridWeightGen --source {} --destination {} --weight {} -m bilinear --64bit_offset  --extrap_method nearestd  --no_log".format(
                    src_temp_path, dest_temp_path, weights
                )
                if self.mpi:
                    cmd = 'mpirun -np {} '.format(n_cpus) + cmd
                logger.info(cmd)
                os.system(cmd)
                os.remove(src_temp_path)
                os.remove(dest_temp_path)

            self.regridder = xe.Regridder(
                self.ds_out, t_ds_out, "bilinear", weights=weights, reuse_weights=True
            )
        self.ds_out = self.regridder(self.ds_out)
        self.in_era5_grid = False
        return

    def reduce_time(self, t0, t1):
        sel_dict = {"valid_time": slice(t0, t1)}
        if self.lat_slice is not None:
            sel_dict['lat'] = slice(self.lat_slice[1], self.lat_slice[0])
        if self.lon_slice is not None:
            sel_dict['lon'] = slice(self.lon_slice[0], self.lon_slice[1])
        self.ds = self.ds.sel(sel_dict).compute()
        return
        
    def reduce_along_lonlat(self, lon, lat, interp_method='nearest'):
        if self.rearranged is False:
            lon = [map_function_inv(i) for i in lon]
        self.ds = self.ds.interp(lon=("lon", lon), lat=("lat", lat),
                                       method=interp_method)
        return
    
    def rearrange_lons(self):
      if self.ds_out['lon'].values.max() > 180:
          self.ds_out = self.ds_out.assign_coords({'lon': [map_function(i) for i in
                                                            self.ds_out.coords['lon'].values]})
          self.ds_out = self.ds_out.sortby('lon')  
          self.rearranged = True
      return

    def load(self):
        self.ds = self.ds.compute()
        return

    def get_data(self, lonlat=None, key=None, interp_method='nearest'):
        if lonlat is None:
            self.rearrange_lons()
        if key is not None:
            tmp = self.ds_out[key]
        else:
            tmp = self.ds_out
        if lonlat is None:
            return tmp
        else:
            lon = lonlat[0]
            if isinstance(lon, list) | isinstance(lon, np.ndarray):
                if self.rearranged is False:
                    lon = [map_function_inv(i) for i in lon]
                return tmp.interp(lon=("lon", lon), lat=("lat", lonlat[1]),
                                  method=interp_method)
            else:
                lon = lonlat[0]
                if self.rearranged is False:
                    lon = map_function_inv(lon)
                return tmp.interp(lon=lon, lat=lonlat[1],
                                 method=interp_method)



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
