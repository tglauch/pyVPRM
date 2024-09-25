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

class modis(earthdata):
    # Class to download and load MODIS data

    def __init__(
        self, datapath=None, sat_image_path=None, sat_img=None, product="MOD09A1.061"
    ):

        super().__init__(datapath, sat_image_path, sat_img)
        if product == "MOD09A1.061":
            self.use_keys = [
                "sur_refl_b01",
                "sur_refl_b02",
                "sur_refl_b03",
                "sur_refl_b04",
                "sur_refl_b05",
                "sur_refl_b06",
                "sur_refl_b07",
                "sur_refl_qc_500m",
                "sur_refl_day_of_year",
                "sur_refl_state_500m",
            ]
        else:
            self.use_keys = None
        self.load_kwargs = {"variable": self.use_keys}
        self.sat = "MODIS"
        self.product = product
        self.path = "MOLT"
        self.resolution = None
        if self.bands is not None:
            self.bands = [i for i in list(self.sat_img.keys()) if "sur_refl_b" in i]

    def start_date(self):
        return parser.parse(
            self.sat_img.attrs["GRANULEBEGINNINGDATETIME"].split(",")[0]
        ).replace(tzinfo=None)

    def stop_date(self):
        return parser.parse(
            self.sat_img.attrs["GRANULEENDINGDATETIME"].split(",")[-1]
        ).replace(tzinfo=None)

    def set_band_names(self):
        logger.info("Trying to set reflectance bands assuming standard naming for MODIS")
        bands = []
        for i in list(self.sat_img.data_vars):
            if ("sur_refl" in i) & ("_b" in i):
                bands.append(i)
        self.bands = bands

    def mask_bad_pixels(self, bands=None):
        # For technical details: https://compscistudies.quora.com/Python-Implementing-a-bitmasking-procedure-to-extract-bits

        if bands is None:
            bands = self.bands
            if self.bands is not None:
                self.bands = [i for i in list(self.sat_img.keys()) if "sur_refl_b" in i]

        band_nums = [(band, int(band.split("_b")[1])) for band in bands]
        masks = dict()

        for b in band_nums:
            start_bit = (b[1] - 1) * 4 + 2  # Start Bit
            end_bit = b[1] * 4 + 1  # End Bit  (inclusive)
            num_bits_to_extract = end_bit - start_bit + 1
            bit_mask = (1 << num_bits_to_extract) - 1
            masks[b[1]] = (
                np.array(self.sat_img["sur_refl_qc_500m"].values, dtype=np.uint32)
                >> start_bit
            ) & bit_mask

        for b in band_nums:
            masks[b[1]] = masks[b[1]] != int(
                "0000", 2
            )  # & (masks[mask_int] != '0111')  &\
            # (masks[mask_int] != '1000') & (masks[mask_int] != '1010') &\
            # (masks[mask_int] != '1100')
            self.sat_img[b[0]] = xr.where(masks[b[1]], np.nan, self.sat_img[b[0]])
            # self.sat_img[b[0]].values[masks[b[1]]] = np.nan
        return

    def mask_clouds(self, bands=None):
        if bands is None:
            bands = self.bands
            if self.bands is not None:
                self.bands = [i for i in list(self.sat_img.keys()) if "sur_refl_b" in i]

        start_bit = 0  # Start Bit
        end_bit = 1  # End Bit  (inclusive)
        num_bits_to_extract = end_bit - start_bit + 1
        bit_mask = (1 << num_bits_to_extract) - 1
        mask1 = (
            (
                np.array(self.sat_img["sur_refl_state_500m"].values, dtype=np.uint32)
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
                np.array(self.sat_img["sur_refl_state_500m"].values, dtype=np.uint32)
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
                np.array(self.sat_img["sur_refl_state_500m"].values, dtype=np.uint32)
                >> start_bit
            )
            & bit_mask
        ) != int("00", 2)

        # start_bit = 0 # Start Bit
        # end_bit = 1 # End Bit  (inclusive)
        # num_bits_to_extract = end_bit - start_bit + 1
        # bit_mask = (1 << num_bits_to_extract) - 1
        # mask = (np.array(self.sat_img['sur_refl_state_500m'].values, dtype=np.uint32) >> start_bit) & bit_mask

        for b in bands:
            self.sat_img[b] = xr.where((mask1 | mask2 | mask3), np.nan, self.sat_img[b])
        # self.sat_img[b].values[(mask1 | mask2 | mask3)] = np.nan
        return

    def get_resolution(self):
        return self.sat_img.rio.resolution()

    def adjust_obs_timestamps(self, key="sur_refl_day_of_year"):
        # make the observation date a utc timestamp. Quite slow.
        start_date = self.start_date()
        stop_date = self.stop_date()
        start_day = start_date.timetuple().tm_yday
        stop_day = stop_date.timetuple().tm_yday
        new = np.full(self.sat_img[key].shape, start_date)
        for this_ts in np.unique(
            self.sat_img[key].values
        ):  # this_ts is the day of the year
            if np.isnan(this_ts):
                this_ts = start_day
            if np.abs(this_ts - start_day) < np.abs(this_ts - stop_day):
                test = start_date + timedelta(days=float(np.abs(this_ts - start_day)))
                new[self.sat_img[key].values == this_ts] = test
            else:
                test = stop_date - timedelta(days=float(np.abs(this_ts - stop_day)))
                new[self.sat_img[key].values == this_ts] = test
        self.sat_img = self.sat_img.assign(
            {key: (list(self.sat_img.dims.mapping.keys()), new)}
        )
        return

    def individual_loading(self, adjust_timestamps=True):
        if ("MOD09GA" in self.product) or ("MOD21" in self.product):
            self.sat_img = rxr.open_rasterio(
                self.sat_image_path, masked=True, cache=False
            )[1].squeeze()
        else:
            self.sat_img = rxr.open_rasterio(
                self.sat_image_path, masked=True, cache=False
            ).squeeze()
        if self.use_keys is None:
            self.use_keys = list(self.sat_img.keys())
        self.sat_img = self.sat_img[self.use_keys]
        # rename_dict = dict()
        bands = []
        for i in self.use_keys:
            if ("sur_refl" in i) & ("_b" in i):
                bands.append(i)
            #  rename_dict[i] = i.replace('_1', '').split('_')[-1].replace('b', 'B')
            #  bands.append(rename_dict[i])
            # else:
            #     rename_dict[i] = i
        # self.sat_img = self.sat_img.rename(rename_dict)
        self.bands = bands
        self.keys = np.array(list(self.sat_img.data_vars))
        for key in self.keys:
            self.sat_img[key] = self.sat_img[key] * self.sat_img[key].scale_factor
        self.meta_data = self.sat_img.attrs
        self.sat_img = self.sat_img.assign_coords({"time": self.get_recording_time()})
        if adjust_timestamps:
            self.adjust_obs_timestamps()
        return

    def get_files(self, dest):
        return glob.glob(os.path.join(dest, "*.hdf"))

    def get_cloud_coverage(self):
        logger.info("PERCENTCLOUDY", self.meta_data["PERCENTCLOUDY"])
        return self.meta_data["PERCENTCLOUDY"]

    def get_recording_time(self):
        date0 = datetime.strptime(
            self.meta_data["RANGEBEGINNINGDATE"]
            + "T"
            + self.meta_data["RANGEBEGINNINGTIME"]
            + "Z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        date1 = datetime.strptime(
            self.meta_data["RANGEENDINGDATE"]
            + "T"
            + self.meta_data["RANGEENDINGTIME"]
            + "Z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        return date0 + (date1 - date0) / 2

    def mask_snow(self, bands=None):
        if bands is None:
            bands = self.bands
            if self.bands is not None:
                self.bands = [i for i in list(self.sat_img.keys()) if "sur_refl_b" in i]
        start_bit = 12  # Start Bit
        end_bit = 12  # End Bit  (inclusive)
        num_bits_to_extract = end_bit - start_bit + 1
        bit_mask = (1 << num_bits_to_extract) - 1
        mask = (
            np.array(self.sat_img["sur_refl_state_500m"].values, dtype=np.uint32)
            >> start_bit
        ) & bit_mask
        for b in bands:
            self.sat_img[b] = xr.where(mask == int("1", 2), np.inf, self.sat_img[b])
            # self.sat_img[b].values[mask == int('1', 2)] = np.inf
        return
