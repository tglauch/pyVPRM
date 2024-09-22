from pyVPRM.sat_managers.base_manager import satellite_data_manager
import rioxarray as rxr
import numpy as np


class city_land_cover_map(satellite_data_manager):

    def __init__(self, sat_image_path):
        super().__init__()
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        self.resolution = 10.0

    def get_resolution(self):
        return self.resolution

    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(
            self.sat_image_path, band_as_variable=True
        ).squeeze()
        self.keys = np.array(list(self.sat_img.data_vars))
