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
from pyVPRM.sat_managers.base_manager import satellite_data_manager
from loguru import logger

class sentinel2(satellite_data_manager):
    # Class to download an load sentinel 2 data
    # Note: Available data on the copernicus hub is
    # limited

    def __init__(self, datapath=None, sat_image_path=None, sat_img=None):
        super().__init__(datapath, sat_image_path)
        self.api = False
        self.load_kwargs = {}
        if sat_img is not None:
            self.sat_img = sat_img
        return

    def individual_loading(self):
        return

    def set_band_names(self):
        logger.info("Trying to set reflectance bands assuming standard naming for Sentinel-2")
        bands = []
        for i in list(self.sat_img.data_vars):
            if (
                ("red" in i)
                | ("blue" in i)
                | ("green" in i)
                | ("swir" in i)
                | ("nir" in i)
            ):
                bands.append(i)
        self.bands = bands

    def download(
        self,
        date0,
        date1,
        savepath,
        username,
        pwd,
        lonlat=False,
        shape=False,
        cloudcoverpercentage=(0, 100),
        server="https://apihub.copernicus.eu/apihub",
        processinglevel="Level-2A",
        platformname="Sentinel-2",
    ):

        # Requires an copernicus account

        # if self.api == False:
        #     self.api = SentinelAPI(username,
        #                            pwd,
        #                            'https://apihub.copernicus.eu/apihub')
        if lonlat is not False:
            footprint = "POINT({} {})".format(lonlat[0], lonlat[1])
        else:
            footprint = shape

        # products = self.api.query(footprint,
        #                      date = (date0, date1), # (date(2022, 8, 10), date(2022, 10, 7)),
        #                      platformname = platformname,
        #                      processinglevel = processinglevel,
        #                      cloudcoverpercentage = cloudcoverpercentage)
        # self.api.download(list(products.keys())[0],
        #                   directory_path=savepath)
        # meta_data = self.api.to_geodataframe(products)
        # self.outpath = self._unzip(savepath, meta_data['title'][0])

    def _unzip(self, folder_path, file_name):
        outpath = os.path.join(folder_path, file_name)
        with zipfile.ZipFile(
            os.path.join(folder_path, file_name + ".zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(outpath)
        return outpath

    def individual_loading(self, bands="all", resolution="20m"):  # [1,2,3]
        ifiles = glob.glob(
            os.path.join(self.outpath, "**/*_B*_{}*".format(resolution)), recursive=True
        )
        if bands != "all":
            ifiles = [i for i in ifiles if i.split("_B")[1].split("_")[0] in bands]
        for j, f in enumerate(ifiles):
            band_name = "B{}".format(f.split("_B")[1].split("_")[0])
            if j == 0:
                t = rxr.open_rasterio(
                    f,
                    masked=False,
                    parse_coordinates=True,
                    band_as_variable=True,
                    cache=False,
                ).squeeze()
                t = t.rename({"band_1": band_name})
                crs = t.rio.crs
            else:
                add_data = rxr.open_rasterio(
                    f, masked=False, cache=False, band_as_variable=True
                ).squeeze()["band_1"]
                t[band_name] = add_data
        t = t.rio.write_crs(crs)  # , inplace=True)
        self.sat_img = t
        self.sat_image_path = self.outpath + ".hdf"
        self.keys = np.array(list(self.sat_img.data_vars))
        self.meta_data = dict()
        meta_data_file = etree.parse(
            glob.glob(os.path.join(self.outpath, "**/MTD_MSIL*.xml"))[0]
        )
        for i in meta_data_file.iter():
            self.meta_data[i.tag] = i.text
        return

    def get_cloud_coverage(self):
        for i in [
            "CLOUDY_PIXEL_OVER_LAND_PERCENTAGE",
            "MEDIUM_PROBA_CLOUDS_PERCENTAGE",
            "HIGH_PROBA_CLOUDS_PERCENTAGE",
        ]:
            logger.info(i, self.meta_data[i])
        return

    def mask_bad_pixels(self, bands=None):
        if bands is None:
            bands = self.bands
        self.sat_img[bands] = xr.where(
            (self.sat_img["scl"] == 0)
            | (self.sat_img["scl"] == 1)
            | (self.sat_img["scl"] == 2)
            | (self.sat_img["scl"] == 3),
            np.nan,
            self.sat_img[bands],
        )
        return

    def mask_clouds(self, bands=None):
        if bands is None:
            bands = self.bands
        self.sat_img[bands] = xr.where(
            (self.sat_img["scl"] == 9)
            | (self.sat_img["scl"] == 8)
            | (self.sat_img["scl"] == 10),
            np.nan,
            self.sat_img[bands],
        )
        return

    def mask_snow(self, bands=None):
        if bands is None:
            bands = self.bands
        self.sat_img[bands] = xr.where(
            (self.sat_img["scl"] == 11), np.inf, self.sat_img[bands]
        )
        return

    def get_recording_time(self):
        if "time" in list(self.sat_img.coords):
            return self.sat_img.coords["time"].values
        else:
            return datetime.strptime(
                self.meta_data["PRODUCT_START_TIME"], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
