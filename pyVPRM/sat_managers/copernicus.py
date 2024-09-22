from pyVPRM.sat_managers.base_manager import satellite_data_manager
import rioxarray as rxr
import numpy as np


class copernicus_land_cover_map(satellite_data_manager):
    # Class to load the copernicus land cover map
    # To get the data, download for example from
    # here: https://lcviewer.vito.be/download

    def __init__(self, sat_image_path):
        super().__init__()
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        self.resolution = 92.2256325412261

    def get_resolution(self):
        return self.resolution

    def individual_loading(self):
        try:
            self.sat_img = rxr.open_rasterio(
                self.sat_image_path, masked=True, band_as_variable=True, cache=False
            ).squeeze()
        except:
            self.sat_img = rxr.open_rasterio(
                self.sat_image_path, masked=True, cache=False
            ).squeeze()
        self.keys = np.array(list(self.sat_img.data_vars))
