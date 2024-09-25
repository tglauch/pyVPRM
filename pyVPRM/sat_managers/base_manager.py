# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import sys
import os
import time
from shapely.geometry import Point, Polygon, box
import rioxarray as rxr
import zipfile
import glob
from pyproj import Transformer
import geopandas as gpd
from pyVPRM.lib import downmodis
import math
from pyproj import Proj
from affine import Affine
from rioxarray.rioxarray import affine_to_coords
import requests
from requests.auth import HTTPDigestAuth
from rioxarray import merge
import yaml
import warnings

warnings.filterwarnings("ignore")
from matplotlib.colors import LinearSegmentedColormap
from lxml import etree
from datetime import datetime
from rasterio.warp import calculate_default_transform
import h5py
from dateutil import parser
import xarray as xr
from datetime import datetime, timedelta, date
import numpy as np
from loguru import logger

def geodesic_point_buffer(lat, lon, km):
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    t = Transformer.from_crs(
        "+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0".format(lat=lat, lon=lon),
        "+proj=longlat +datum=WGS84",
    )
    ret = [t.transform(i[0], i[1]) for i in buf.exterior.coords[:]]
    return ret


def make_cmap(vmin, vmax, nbins, cmap_name="Reds"):
    if isinstance(cmap_name, list):
        cmap = LinearSegmentedColormap.from_list("some_name", cmap_name, N=nbins)
    else:
        cmap = mpl.cm.get_cmap(cmap_name)  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]

        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N
        )
    #    cmap.set_under('blue') #'#d4ebf2')

    # define the bins and normalize
    bounds = np.linspace(vmin, vmax, nbins)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


class satellite_data_manager:
    # Base class for all satellite images

    def __init__(self, datapath=None, sat_image_path=None, sat_img=None):
        self.bands = None
        self.outpath = datapath
        self.sat_image_path = sat_image_path
        self.sat_img = sat_img
        if isinstance(self.sat_img, str):
            self.sat_img = xr.open_dataset(self.sat_img)
            self.bands = list(self.sat_img.keys())
        self.t = None

        return

    def value_at_lonlat(
        self, lon, lat, as_array=True, key=None, isel={}, interp_method="nearest"
    ):
        """
        Get the value at a specific latitude (lat) and longitude (lon)

            Parameters:
                    lon(float or list of floats) : longitude
                    lat(float or list of floats) : latitude
                    as_array (bool): If true, return numpy array
                    key (str): Only get values for a specific key. All keys if None
                    interp_method (str): which method to use for interpolating. 'linear' by default.
                                         For nearest neighbour set to 'nearest'.

            Returns:
                    None
        """

        if self.t is None:
            self.t = Transformer.from_crs(
                "+proj=longlat +datum=WGS84", self.sat_img.rio.crs
            )
        x_a, y_a = self.t.transform(lon, lat)
        # ret = self.sat_img.sel(x=x_a, y=y_a, method="nearest")
        if key is not None:
            if isinstance(x_a, list):
                ret = (
                    self.sat_img[key]
                    .isel(isel)
                    .interp(x=("z", x_a), y=("z", y_a), method=interp_method)
                )
            else:
                ret = (
                    self.sat_img[key]
                    .isel(isel)
                    .interp(x=x_a, y=y_a, method=interp_method)
                )
        else:
            if isinstance(x_a, list) | isinstance(x_a, np.ndarray):
                ret = self.sat_img.isel(isel).interp(
                    x=("z", x_a), y=("z", y_a), method=interp_method
                )
            else:
                ret = self.sat_img.isel(isel).interp(x=x_a, y=y_a, method=interp_method)
        if as_array:
            return ret.to_array().values
        else:
            return ret

    def drop_bands(self, band_names=None):
        if band_names is not None:
            self.sat_img = self.sat_img.drop(band_names)
        else:
            if self.bands is None:
                self.set_band_names()
            self.sat_img = self.sat_img.drop(self.bands)

    def reduce_along_lon_lat(
        self, lon, lat, interp_method="nearest", new_dim_name="z", inplace=True
    ):
        if self.t is None:
            self.t = Transformer.from_crs(
                "+proj=longlat +datum=WGS84", self.sat_img.rio.crs
            )
        x_a, y_a = self.t.transform(lon, lat)
        ret = self.sat_img.interp(
            x=(new_dim_name, x_a), y=(new_dim_name, y_a), method=interp_method
        )
        if inplace:
            self.sat_img = ret
            return
        else:
            return ret

    def load(self, proj=None, **kwargs):
        # loading using the individual loading functions
        # of the derived classes

        self.individual_loading(**kwargs)
        if proj is not None:
            self.reproject(proj=proj)
        return

    def get_band_names(self):
        return self.keys

    def reproject(self, proj, **kwargs):
        # reproject using a pre-defined projection or by passing a
        # projection string

        if proj == "WGS84":
            self.sat_img = self.sat_img.rio.reproject(
                "+proj=longlat +datum=WGS84", **kwargs
            )  #
        elif proj == "CRS":
            self.sat_img = self.sat_img.rio.reproject(
                self.sat_img.rio.estimate_utm_crs(), **kwargs
            )
        else:
            self.sat_img = self.sat_img.rio.reproject(proj, **kwargs)
        try:
            self.proj_dict = self.sat_img.rio.crs.to_dict()
        except Exception as e:
            self.proj_dict = {}
            logger.info(e)
        self.t = Transformer.from_crs(
            "+proj=longlat +datum=WGS84", self.sat_img.rio.crs
        )
        return

    def transform_to_grid(self, sat_manager_inst):

        import xesmf as xe

        src_x = self.sat_img.coords["x"].values
        src_y = self.sat_img.coords["y"].values
        X, Y = np.meshgrid(src_x, src_y)
        t = Transformer.from_crs(self.sat_img.rio.crs, "+proj=longlat +datum=WGS84")
        x_long, y_lat = t.transform(X, Y)
        src_grid = xr.Dataset(
            {
                "lon": (["y", "x"], x_long, {"units": "degrees_east"}),
                "lat": (["y", "x"], y_lat, {"units": "degrees_north"}),
            }
        )
        src_grid = src_grid.set_coords(["lon", "lat"])

        src_x = sat_manager_inst.sat_img.coords["x"].values
        src_y = sat_manager_inst.sat_img.coords["y"].values
        X, Y = np.meshgrid(src_x, src_y)
        t = Transformer.from_crs(
            sat_manager_inst.sat_img.rio.crs, "+proj=longlat +datum=WGS84"
        )
        x_long, y_lat = t.transform(X, Y)
        dest_grid = xr.Dataset(
            {
                "lon": (["y", "x"], x_long, {"units": "degrees_east"}),
                "lat": (["y", "x"], y_lat, {"units": "degrees_north"}),
            }
        )
        dest_grid = dest_grid.set_coords(["lon", "lat"])

        regridder = xe.Regridder(src_grid, dest_grid, "bilinear")
        self.sat_img = regridder(self.sat_img)
        self.sat_img = self.sat_img.assign_coords(
            {
                "x": sat_manager_inst.sat_img.coords["x"].values,
                "y": sat_manager_inst.sat_img.coords["y"].values,
            }
        )
        self.sat_img.rio.set_crs(sat_manager_inst.sat_img.rio.crs)
        return

    def plot_bands(self, cmaps="Reds", titles=None, save=None):
        # plot the bands of the satellite image
        pass

    def plot_rgb(self, which_bands=[2, 1, 0], str_clip=2, figsize=0.9, save=None):
        # plot an rgb image with the bands given in which_bands
        pass

    def plot_ndvi(
        self, band1, band2, figsize=0.9, save=None, n_colors=9, vmin=None, vmax=1.0
    ):
        # plot the normalized difference vegetation index
        pass

    def add_tile(self, new_tiles, reproject=False):
        # merge tiles together using the projection of the current satellite image

        if not isinstance(new_tiles, list):
            new_tiles = [new_tiles]
        if not np.all([isinstance(i, satellite_data_manager) for i in new_tiles]):
            logger.info("Can only merge with another instance of a satellite_data_manger")
        if reproject:
            logger.info("Do reprojections")
            to_merge = [
                i.sat_img.rio.reproject(self.sat_img.rio.crs) for i in new_tiles
            ]
        else:
            to_merge = [i.sat_img for i in new_tiles]
        to_merge.append(self.sat_img)
        logger.info("Merge")
        self.sat_img = merge.merge_datasets(to_merge)
        return

    def crop_box(self, box, from_disk=False):
        if not box.crs == self.sat_img.rio.crs:
            # If the crs is not equal reproject the data
            box = box.to_crs(self.sat_img.rio.crs)
        self.sat_img = self.sat_img.rio.clip(
            box, all_touched=True, from_disk=from_disk
        ).squeeze()

    def crop(self, lonlat, radius, from_disk=False):
        # crop the satellite images in place using a given radius around a given
        # longitude and latitude

        circle_poly = gpd.GeoSeries(
            Polygon(geodesic_point_buffer(lonlat[1], lonlat[0], radius)), crs="WGS 84"
        )
        if not circle_poly.crs == self.sat_img.rio.crs:
            # If the crs is not equal reproject the data
            circle_poly = circle_poly.to_crs(self.sat_img.rio.crs)

        crop_bound_box = [box(*circle_poly.total_bounds)]

        self.sat_img = self.sat_img.rio.clip(
            [box(*circle_poly.total_bounds)], all_touched=True, from_disk=from_disk
        ).squeeze()
        return

    def crop_to_polygon(self, polygon, from_disk=False):
        if not polygon.crs == self.sat_img.rio.crs:
            polygon = polygon.to_crs(self.sat_img.rio.crs)

        self.sat_img = self.sat_img.rio.clip(
            polygon.geometry, all_touched=True, from_disk=from_disk
        ).squeeze()
        return

    def crop_to_number_of_pixels(self, lonlat, num_pixels, key, reproject=False):
        # crop the satellite images in place to a certain number of pixels around
        # the given longitude and latitude

        if (self.sat_img.rio.crs != self.sat_img.rio.estimate_utm_crs()) & reproject:
            self.sat_img = self.sat_img.rio.reproject(
                self_sat.img.rio.estimate_utm_crs()
            )
        t = Transformer.from_crs("+proj=longlat +datum=WGS84", self.sat_img.rio.crs)
        x_a, y_a = t.transform(lonlat[0], lonlat[1])
        x_ind = np.argmin(np.abs(self.sat_img.x.values - x_a))
        y_ind = np.argmin(np.abs(self.sat_img.y.values - y_a))
        shift = int(np.floor(num_pixels / 2.0))
        return self.sat_img[key].values[
            y_ind - shift : y_ind + shift + 1, x_ind - shift : x_ind + shift + 1
        ]

    def individual_loading(self):
        return

    def save(self, save_path):
        # save satellite image to save_path

        self.sat_img.to_netcdf(save_path)
        return


class earthdata(satellite_data_manager):
    # Classes for everything that can be downloaded from the NASA
    # earthdata server, especially MODIS and VIIRS

    def __init__(self, datapath=None, sat_image_path=None, sat_img=None):
        super().__init__(datapath, sat_image_path, sat_img)
        return

    def _init_downloader(
        self,
        dest,
        date,
        delta,
        username,
        lonlat=None,
        pwd=None,
        token=None,
        jpg=False,
        enddate=None,
        hv=None,
    ):
        # Downloader with interface to the Earthdata server

        if hv is not None:
            h = hv[0]
            v = hv[1]
        else:
            h, v = self.lat_lon_to_modis(lonlat[1], lonlat[0])
        day = "{}-{:02d}-{:02d}".format(date.year, date.month, date.day)
        tiles = "h{:02d}v{:02d}".format(h, v)

        downloader = downmodis.downModis(
            destinationFolder=dest,
            tiles=tiles,
            today=day,
            product=self.product,
            path=self.path,
            delta=delta,
            user=username,
            enddate=enddate,
            password=pwd,
            token=token,
            jpg=jpg,
        )

        return downloader

    def download(
        self,
        date,
        savepath,
        username,
        lonlat=None,
        pwd=None,
        token=None,
        delta=1,
        jpg=False,
        enddate=None,
        hv=None,
        rmv_downloads=False,
    ):

        modisDown = self._init_downloader(
            savepath, date, delta, username, lonlat, pwd, token, jpg, enddate, hv
        )

        # try:
        modisDown.connect()
        ds = modisDown.getListDays()
        for d in ds:
            fs = modisDown.getFilesList(d)
            cde = modisDown.checkDataExist(fs)
            logger.info("Download {}: {}".format(d, cde))
            for c in cde:
                os.system(
                    "wget --user {} --password {} --directory-prefix {} {} ".format(
                        modisDown.user,
                        modisDown.password,
                        modisDown.writeFilePath,
                        os.path.join(modisDown.url, modisDown.path, d, c),
                    )
                )
                time.sleep(5)
            # modisDown.dayDownload(d, cde)
        return

    def _to_standard_format(self):
        return

    def lat_lon_to_modis(self, lat, lon):
        CELLS = 2400
        VERTICAL_TILES = 18
        HORIZONTAL_TILES = 36
        EARTH_RADIUS = 6371007.181
        EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

        TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
        TILE_HEIGHT = TILE_WIDTH
        CELL_SIZE = TILE_WIDTH / CELLS
        MODIS_GRID = Proj(f"+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext")
        x, y = MODIS_GRID(lon, lat)
        h = (EARTH_WIDTH * 0.5 + x) / TILE_WIDTH
        v = -(EARTH_WIDTH * 0.25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
        return int(h), int(v)
