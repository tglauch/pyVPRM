from pyVPRM.sat_managers.base_manager import satellite_data_manager
import rioxarray as rxr
import numpy as np
import xarray as xr


class mapbiomas(satellite_data_manager):
    # Class to load the copernicus land cover map
    # To get the data, download for example from
    # here: https://lcviewer.vito.be/download

    def __init__(self, sat_image_path):
        super().__init__()
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        self.resolution = 30

    def get_resolution(self):
        return self.resolution

    def individual_loading(self):
        self.sat_img = xr.open_dataset(self.sat_image_path)
        self.sat_img = self.sat_img.rio.write_crs("WGS84")
        self.sat_img = self.sat_img.rename({"band_data": "band_1"})
        self.sat_img["band_1"] = xr.where(
            np.isnan(self.sat_img["band_1"]), 8, self.sat_img["band_1"]
        )
        # try:
        #     self.sat_img = rxr.open_rasterio(self.sat_image_path,
        #                      masked=True, band_as_variable=True,
        #                                     cache=False).squeeze()
        # except:
        #     self.sat_img = rxr.open_rasterio(self.sat_image_path,
        #                      masked=True, cache=False).squeeze()
        self.keys = np.array(list(self.sat_img.data_vars))
