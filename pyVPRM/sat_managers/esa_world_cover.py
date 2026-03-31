from pyVPRM.sat_managers.base_manager import satellite_data_manager
import rioxarray as rxr
import numpy as np
import yaml
import xarray as xr

class esa_world_cover(satellite_data_manager):

    def __init__(self, sat_image_path):
        super().__init__()
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        self.resolution = 10.0

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


    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(
            self.sat_image_path,
            band_as_variable=True,
            chunks=True, 
        ).squeeze()
        self.keys = np.array(list(self.sat_img.data_vars))
