from pyVPRM.sat_managers.base_manager import satellite_data_manager
import rioxarray as rxr
import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import distance_transform_edt

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

    def map_veg_classes(self, cfg_path, var_name='band_1'):
        map_to_vprm_class = dict()
        with open(cfg_path, "r") as stream:
            try:
                vprm_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.info(exc)
        for key in vprm_cfg:
            for c in vprm_cfg[key]["class_numbers"]:
                map_to_vprm_class[c] = vprm_cfg[key]["vprm_class"]
        for key in map_to_vprm_class.keys():
                self.sat_img[var_name] = xr.where(
                    self.sat_img[var_name] == key,
                    map_to_vprm_class[key],
                    self.sat_img[var_name],
                )
        return

    def map_class_to_nearest_valid_class(self, to_map=3, target=[1,2],
                                         band_name = 'band_1'):
        da = self.sat_img[band_name].load()
        arr = da.values
        bad_mask = arr == to_map
        good_mask = np.isin(arr, target)
        _, (idx_y, idx_x) = distance_transform_edt(
            ~good_mask,
            return_indices=True)
        arr[bad_mask] = arr[idx_y[bad_mask], idx_x[bad_mask]]
        return


    def individual_loading(self):
        try:
            self.sat_img = rxr.open_rasterio(
                self.sat_image_path,
                masked=True,
                band_as_variable=True,
                chunks=True,
                cache=False
            ).squeeze()
        except:
            self.sat_img = rxr.open_rasterio(
                self.sat_image_path, masked=True, cache=False
            ).squeeze()
        self.keys = np.array(list(self.sat_img.data_vars))
