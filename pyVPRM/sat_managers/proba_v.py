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


# Proba-V is not yet functional
class proba_v(satellite_data_manager):
    # Class to download and load Proba V data

    def __init__(self, datapath=None, sat_image_path=None):
        super().__init__(datapath, sat_image_path)
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        self.path = "PROBA-V_100m"
        self.product = "S5_TOC_100_m_C1"
        self.server = "https://www.vito-eodata.be/PDF/datapool/Free_Data/"
        self.pixel_size = 100

    # PROBA-V_100m/S5_TOC_100_m_C1/2021/05/06/PV_S5_TOC-20210506_100M_V101/PROBAV_S5_TOC_X14Y00_20210506_100M_V101.HDF5

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
    ):

        session = requests.Session()
        session.auth = ("theo_g", "%FeRiEn%07")
        url = "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/S5_TOC_100_m_C1/2021/09/11/PV_S5_TOC-20210911_100M_V101/PROBAV_S5_TOC_X15Y04_20210911_100M_V101.HDF5"
        auth = session.post(url)
        data = session.get(url)
        with open(
            "/home/b/b309233/temp/PROBAV_S5_TOC_X15Y04_20210911_100M_V101.HDF5", "wb"
        ) as f:
            f.write(data.content)
        return

    def coord_to_tile(self):
        return
        # https://proba-v.vgt.vito.be/sites/probavvgt/files/Products_User_Manual.pdft

    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(self.sat_image_path, masked=True, cache=False)
        self.keys = np.array(list(self.sat_img.data_vars))
        self.sat_img.rio.write_crs(
            self.sat_img.attrs["MAP_PROJECTION_WKT"], inplace=True
        )

    def get_recording_time(self):
        return
