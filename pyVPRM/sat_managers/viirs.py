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
from pyVPRM.sat_managers.base_manager import earthdata
from loguru import logger

class VIIRS(earthdata):
    # Class to download and load VIIRS data

    def __init__(self, datapath=None, sat_image_path=None, sat_img=None):
        super().__init__(datapath, sat_image_path, sat_img)
        self.use_keys = []
        self.load_kwargs = {"variable": self.use_keys}
        self.sat = "VIIRS"
        self.path = "VIIRS"
        self.product = "VNP09H1.001"  #'VNP09GA.001' #
        self.pixel_size = 463.31271652777775

    def get_files(self, dest):
        return glob.glob(os.path.join(dest, "*.h5"))

    def set_sat_img(self, ind):
        # implements ones M and L bands are used. Currently only M bands implemented.
        return

    def set_band_names(self):
        logger.info("Trying to set reflectance bands assuming standard naming for VIIRS")
        bands = []
        for k in list(self.sat_img.data_vars):
            if ("SurfReflect_I" not in k) & ("SurfReflect_M" not in k):
                continue
            bands.append(k)
        self.bands = bands

    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(self.sat_image_path, masked=True, cache=False)
        f = h5py.File(self.sat_image_path, "r")

        fileMetadata = f["HDFEOS INFORMATION"]["StructMetadata.0"][()].split()
        fileMetadata = [m.decode("utf-8") for m in fileMetadata]
        ulc = [i for i in fileMetadata if "UpperLeftPointMtrs" in i][0]
        west = float(ulc.split("=(")[-1].replace(")", "").split(",")[0])
        north = float(ulc.split("=(")[-1].replace(")", "").split(",")[1])
        del f

        if isinstance(self.sat_img, list):
            self.sat_img = self.sat_img[0].squeeze()
            self.keys = [i for i in list(self.sat_img.data_vars) if "_M" in i]
            self.sat_img = self.sat_img[self.keys]
        else:
            self.sat_img = self.sat_img.squeeze()
            self.keys = self.sat_img.data_vars

        crs_str = 'PROJCS["unnamed",\
                    GEOGCS["Unknown datum based upon the custom spheroid", \
                    DATUM["Not specified (based on custom spheroid)", \
                    SPHEROID["Custom spheroid",6371007.181,0]], \
                    PRIMEM["Greenwich",0],\
                    UNIT["degree",0.0174532925199433]],\
                    PROJECTION["Sinusoidal"], \
                    PARAMETER["longitude_of_center",0], \
                    PARAMETER["false_easting",0], \
                    PARAMETER["false_northing",0], \
                    UNIT["Meter",1]]'

        rename_dict = dict()
        for i in self.keys:
            rename_dict[i] = i.split("Data_Fields_")[1]
        self.sat_img = self.sat_img.rename(rename_dict)
        self.keys = np.array(list(self.sat_img.data_vars))
        bands = []
        for k in self.keys:
            if ("SurfReflect_I" not in k) & ("SurfReflect_M" not in k):
                continue
            sf = self.sat_img.attrs[
                [
                    i
                    for i in self.sat_img.attrs
                    if ("scale_factor" in i) & ("err" not in i) & (k in i)
                ][0]
            ]
            self.sat_img[k] = self.sat_img[k] * sf
            bands.append(k)
        self.bands = bands
        transform = Affine(self.pixel_size, 0, west, 0, -self.pixel_size, north)
        coords = affine_to_coords(
            transform, self.sat_img.rio.width, self.sat_img.rio.height
        )
        self.sat_img.coords["x"] = coords["x"]
        self.sat_img.coords["y"] = coords["y"]
        self.sat_img.rio.write_crs(crs_str, inplace=True)
        self.meta_data = self.sat_img.attrs
        return

    def mask_bad_pixels(self, bands=None):
        if bands is None:
            bands = self.bands

        band_nums = [int(band[-1]) for band in bands]
        masks = dict()

        for b in band_nums:
            if b > 2:  ## VIIRS only has quality mask for band1 and band2
                continue
            start_bit = b * 4  # Start Bit
            end_bit = b * 4 + 3  # End Bit  (inclusive)
            num_bits_to_extract = end_bit - start_bit + 1
            bit_mask = (1 << num_bits_to_extract) - 1
            masks[b] = (
                np.array(self.sat_img["SurfReflect_QC_500m"].values, dtype=np.uint32)
                >> start_bit
            ) & bit_mask

        for mask_int in masks.keys():
            masks[mask_int] = masks[mask_int] != int("0000", 2)
            self.sat_img["SurfReflect_I{}".format(mask_int)] = xr.where(
                masks[mask_int],
                np.nan,
                self.sat_img["SurfReflect_I{}".format(mask_int)],
            )
            # self.sat_img['SurfReflect_I{}'.format(mask_int)].values[masks[mask_int]] = np.nan
        return

    def mask_clouds(self, bands=None):
        if bands is None:
            bands = self.bands

        start_bit = 0  # Start Bit
        end_bit = 1  # End Bit  (inclusive)
        num_bits_to_extract = end_bit - start_bit + 1
        bit_mask = (1 << num_bits_to_extract) - 1
        mask1 = (
            (
                np.array(self.sat_img["SurfReflect_State_500m"].values, dtype=np.uint32)
                >> start_bit
            )
            & bit_mask
        ) != int("00", 2)

        start_bit = 2  # Start Bit
        end_bit = 2  # End Bit  (inclusive)
        num_bits_to_extract = end_bit - start_bit + 1
        bit_mask = (1 << num_bits_to_extract) - 1
        mask2 = (
            (
                np.array(self.sat_img["SurfReflect_State_500m"].values, dtype=np.uint32)
                >> start_bit
            )
            & bit_mask
        ) != int("0", 2)

        start_bit = 8  # Start Bit
        end_bit = 9  # End Bit  (inclusive)
        num_bits_to_extract = end_bit - start_bit + 1
        bit_mask = (1 << num_bits_to_extract) - 1
        mask3 = (
            (
                np.array(self.sat_img["SurfReflect_State_500m"].values, dtype=np.uint32)
                >> start_bit
            )
            & bit_mask
        ) != int("00", 2)

        for b in bands:
            self.sat_img[b] = xr.where((mask1 | mask2 | mask3), np.nan, self.sat_img[b])
            # self.sat_img[b].values[mask != int('00', 2)] = np.nan
        return

    def mask_snow(self, bands=None):
        if bands is None:
            bands = self.bands
        start_bit = 12  # Start Bit
        end_bit = 12  # End Bit  (inclusive)
        num_bits_to_extract = end_bit - start_bit + 1
        bit_mask = (1 << num_bits_to_extract) - 1
        mask = (
            np.array(self.sat_img["SurfReflect_State_500m"].values, dtype=np.uint32)
            >> start_bit
        ) & bit_mask
        for b in bands:
            self.sat_img[b] = xr.where(mask == int("1", 2), np.inf, self.sat_img[b])
            # self.sat_img[b].values[mask == int('1', 2)] = np.inf
        return

    def get_cloud_coverage(self):
        logger.info("PercentCloudy", self.meta_data["PercentCloudy"])
        return self.meta_data["PercentCloudy"]

    def get_recording_time(self):
        date0 = datetime.strptime(
            self.meta_data["RangeBeginningDate"]
            + "T"
            + self.meta_data["RangeBeginningTime"]
            + "Z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        date1 = datetime.strptime(
            self.meta_data["RangeEndingDate"]
            + "T"
            + self.meta_data["RangeEndingTime"]
            + "Z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        return date0 + (date1 - date0) / 2
