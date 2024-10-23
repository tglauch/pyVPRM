import warnings

warnings.filterwarnings("ignore")
import sys
import os
import pathlib
import numpy as np
import pyVPRM
from pyVPRM.sat_managers.base_manager import satellite_data_manager
from pyVPRM.lib.functions import (
    add_corners_to_1d_grid,
    do_lowess_smoothing,
    make_xesmf_grid,
    to_esmf_grid,
)
from scipy.ndimage import uniform_filter
from pyproj import Transformer
import copy
from joblib import Parallel, delayed
import xarray as xr
import scipy
import uuid
import time
from scipy.optimize import curve_fit
import pandas as pd
import datetime
from dateutil import parser
from multiprocessing import Process
import rasterio
from astropy.convolution import convolve
from datetime import datetime, timedelta
import yaml
from loguru import logger

regridder_options = dict()
regridder_options["conservative"] = "conserve"


class vprm:
    """
    Class for the  Vegetation Photosynthesis and Respiration Model
    """

    def __init__(
        self, vprm_config_path, land_cover_map=None, verbose=False, n_cpus=1, sites=None
    ):
        """
        Initialize a class instance

        Parameters:
                land_cover_map (xarray): A pre calculated map with the land cover types
                verbose (bool): Set true for additional output when debugging
                n_cpus: Number of CPUs
                sites: For fitting. Provide a list of sites.

        Returns:
                The lowess smoothed array
        """

        logger.info("Running with pyVPRM version {}".format(pyVPRM.__version__))
        self.sat_imgs = []

        self.sites = sites
        if self.sites is not None:
            self.lonlats = [i.get_lonlat() for i in sites]
        self.n_cpus = n_cpus
        self.counter = 0
        self.fit_params_dict = None
        self.res = None

        self.new = True
        self.timestamps = []
        self.t2m = None

        # self.target_shape = None

        self.sat_img_buffer = dict()
        self.buffer = dict()
        self.buffer["cur_lat"] = None
        self.buffer["cur_lon"] = None
        self.prototype_lat_lon = None

        self.land_cover_type = land_cover_map
        # land_cover_type: tmin, topt, tmax

        with open(vprm_config_path, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.info(exc)

        self.temp_coefficients = dict()
        self.map_to_vprm_class = dict()
        for key in cfg:
            if "tmin" in cfg[key].keys():
                self.temp_coefficients[cfg[key]["vprm_class"]] = [
                    cfg[key]["tmin"],
                    cfg[key]["topt"],
                    cfg[key]["tmax"],
                    cfg[key]["tlow"],
                ]
            for c in cfg[key]["class_numbers"]:
                self.map_to_vprm_class[c] = cfg[key]["vprm_class"]
        return

    def to_wrf_output(
        self,
        out_grid,
        weights_for_regridder=None,
        regridder_save_path=None,
        driver="xEMSF",
        interp_method="conservative",
        n_cpus=None,
        mpi=True,
        logs=False,
    ):
        """
        Generate output in the format that can be used as an input for WRF

            Parameters:
                    out_grid (dict or xarray): Can be either a dictionary with 1D lats and lons
                                               or an xarray dataset
                    weights_for_regridder (str): Weights to be used for regridding to the WRF grid
                    regridder_save_path (str): Save path when generating a new regridder
                    driver (str): Either ESMF_RegridWeightGen or xESMF. When setting to ESMF_RegridWeightGen
                                  the ESMF library is called directly

            Returns:
                    Dictionary with a dictinoary of the WRF input arrays
        """

        import xesmf as xe

        if n_cpus is None:
            n_cpus = self.n_cpus

        src_grid = make_xesmf_grid(self.sat_imgs.sat_img)
        if isinstance(out_grid, dict):
            ds_out = make_xesmf_grid(out_grid)
        else:
            ds_out = out_grid

        if weights_for_regridder is None:
            logger.info(
                "Need to generate the weights for the regridder. This can be very slow and memory intensive"
            )
            if driver == "xEMSF":
                regridder = xe.Regridder(src_grid, ds_out, interp_method)
                if regridder_save_path is not None:
                    regridder.to_netcdf(regridder_save_path)
            elif driver == "ESMF_RegridWeightGen":
                if regridder_save_path is None:
                    logger.info(
                        "If you use ESMF_RegridWeightGen, a regridder_save_path needs to be given"
                    )
                    return
                src_temp_path = os.path.join(
                    os.path.dirname(regridder_save_path),
                    "{}.nc".format(str(uuid.uuid4())),
                )
                dest_temp_path = os.path.join(
                    os.path.dirname(regridder_save_path),
                    "{}.nc".format(str(uuid.uuid4())),
                )
                src_grid_esmf = to_esmf_grid(self.sat_imgs.sat_img)
                ds_out_esmf = to_esmf_grid(out_grid)
                src_grid_esmf.to_netcdf(src_temp_path)
                ds_out_esmf.to_netcdf(dest_temp_path)
                exec_str = "ESMF_RegridWeightGen --source {} --destination {} --weight {} -m {} -r --netcdf4 –src_regional –dest_regional ".format(
                    src_temp_path,
                    dest_temp_path,
                    regridder_save_path,
                    regridder_options[interp_method],
                )
                if mpi is True:
                    exec_str = "mpirun -np {} ".format(n_cpus) + exec_str
                if not logs:
                    exec_str += " --no_log "
                logger.info(exec_str)
                os.system(exec_str)  # --no_log
                # os.remove(src_temp_path)
                # os.remove(dest_temp_path)
                weights_for_regridder = regridder_save_path
            else:
                logger.info("Driver needs to be xEMSF or ESMF_RegridWeightGen")
        if weights_for_regridder is not None:
            regridder = xe.Regridder(
                src_grid,
                ds_out,
                interp_method,
                weights=weights_for_regridder,
                reuse_weights=True,
            )
        veg_inds = np.unique(
            [self.map_to_vprm_class[i] for i in self.map_to_vprm_class.keys()]
        )
        veg_inds = np.array(veg_inds, dtype=np.int32)
        dims = [i for i in list(ds_out.dims.mapping.keys()) if "_b" not in i]
        lcm = regridder(self.land_cover_type.sat_img)
        lcm = lcm.to_dataset(name="vegetation_fraction_map")
        lcm = lcm.rename({"y": "south_north", "x": "west_east"})
        day_of_the_year = np.array(
            self.sat_imgs.sat_img[self.time_key].values, dtype=np.int32
        )
        day_of_the_year += 1 - day_of_the_year[0]
        kys = len(self.sat_imgs.sat_img[self.time_key].values)
        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({self.time_key: ky})["evi"].where(
                    self.land_cover_type.sat_img.sel({"vprm_classes": v}) > 0, np.nan
                )
                sub_array.append(regridder(tres.values, skipna=True))
            final_array.append(sub_array)
        out_dims = ["vprm_classes", "time"]
        out_dims.extend(dims)
        ds_t_evi = copy.deepcopy(ds_out)
        ds_t_evi = ds_t_evi.assign({"evi": (out_dims, np.moveaxis(final_array, 0, 1))})
        ds_t_evi = ds_t_evi.assign_coords({"time": day_of_the_year})
        ds_t_evi = ds_t_evi.assign_coords({"vprm_classes": veg_inds})
        ds_t_evi = ds_t_evi.rename({"y": "south_north", "x": "west_east"})

        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({self.time_key: ky})["lswi"].where(
                    self.land_cover_type.sat_img.sel({"vprm_classes": v}) > 0, np.nan
                )
                sub_array.append(regridder(tres.values, skipna=True))
            final_array.append(sub_array)
        ds_t_lswi = copy.deepcopy(ds_out)
        ds_t_lswi = ds_t_lswi.assign(
            {"lswi": (out_dims, np.moveaxis(final_array, 0, 1))}
        )
        ds_t_lswi = ds_t_lswi.assign_coords({"time": day_of_the_year})
        ds_t_lswi = ds_t_lswi.assign_coords({"vprm_classes": veg_inds})
        ds_t_lswi = ds_t_lswi.rename({"y": "south_north", "x": "west_east"})

        out_dims = ["vprm_classes"]
        out_dims.extend(dims)
        ds_t_max_evi = copy.deepcopy(ds_out)
        ds_t_max_evi = ds_t_max_evi.assign(
            {"evi_max": (out_dims, np.nanmax(ds_t_evi["evi"], axis=1))}
        )
        ds_t_max_evi = ds_t_max_evi.assign_coords({"vprm_classes": veg_inds})
        ds_t_max_evi = ds_t_max_evi.rename({"y": "south_north", "x": "west_east"})

        ds_t_min_evi = copy.deepcopy(ds_out)
        ds_t_min_evi = ds_t_min_evi.assign(
            {"evi_min": (out_dims, np.nanmin(ds_t_evi["evi"], axis=1))}
        )
        ds_t_min_evi = ds_t_min_evi.assign_coords({"vprm_classes": veg_inds})
        ds_t_min_evi = ds_t_min_evi.rename({"y": "south_north", "x": "west_east"})

        ds_t_max_lswi = copy.deepcopy(ds_out)
        ds_t_max_lswi = ds_t_max_lswi.assign(
            {"lswi_max": (out_dims, np.nanmax(ds_t_lswi["lswi"], axis=1))}
        )
        ds_t_max_lswi = ds_t_max_lswi.assign_coords({"vprm_classes": veg_inds})
        ds_t_max_lswi = ds_t_max_lswi.rename({"y": "south_north", "x": "west_east"})

        ds_t_min_lswi = copy.deepcopy(ds_out)
        ds_t_min_lswi = ds_t_min_lswi.assign(
            {"lswi_min": (out_dims, np.nanmin(ds_t_lswi["lswi"], axis=1))}
        )
        ds_t_min_lswi = ds_t_min_lswi.assign_coords({"vprm_classes": veg_inds})
        ds_t_min_lswi = ds_t_min_lswi.rename({"y": "south_north", "x": "west_east"})

        ret_dict = {
            "lswi": ds_t_lswi,
            "evi": ds_t_evi,
            "veg_fraction": lcm,
            "lswi_max": ds_t_max_lswi,
            "lswi_min": ds_t_min_lswi,
            "evi_max": ds_t_max_evi,
            "evi_min": ds_t_min_evi,
        }

        for key in ret_dict.keys():
            ret_dict[key] = ret_dict[key].assign_attrs(
                title="VPRM input data for WRF: {}".format(key),
                # MODIS_version = '061',
                software_version=pyVPRM.__version__,
                software_github="https://github.com/tglauch/pyVPRM",
                author="Dr. Theo Glauch",
                institution1="Heidelberg University",
                institution2="Deutsches Zentrum für Luft- und Raumfahrt (DLR)",
                contact="theo.glauch@dlr.de",
                date_created=str(datetime.now()),
                comment="Used VPRM classes: 1 Evergreen forest, 2 Deciduous forest, 3 Mixed forest, 4 Shrubland, 5 Trees and grasses, 6 Cropland, 7 Grassland, 8 Barren, Urban and built-up, water, permanent snow and ice",
            )
        return ret_dict

    def add_sat_img(
        self,
        handler,
        b_nir=None,
        b_red=None,
        b_blue=None,
        b_swir=None,
        drop_bands=False,
        which_evi=None,
        timestamp_key=None,
        mask_bad_pixels=True,
        mask_clouds=True,
        mask_snow=True,
    ):
        """
        Add a new satellite image and calculate EVI and LSWI if desired

            Parameters:
                    handler (satellite_data_manager): The satellite image
                    b_nir (str): Name of the near-infrared band
                    b_red (str): Name of the red band
                    b_blue (str): Name of the blue band
                    b_swir (str): Name of the short-wave infrared band
                    drop_bands (bool): If True drop the raw band information after
                                       calculation of EVI and LSWI. Saves memory.
                                       Can also be a list of keys to drop.
                    which_evi (str): Either evi or evi2. evi2 does not need a blue band.
                    timestamp_key (float): satellite data key containing a timestamp for each
                                           single pixel - to be used with lowess

            Returns:
                    None
        """

        evi_params = {"g": 2.5, "c1": 6.0, "c2": 7.5, "l": 1}
        evi2_params = {"g": 2.5, "l": 1, "c": 2.4}

        if not isinstance(handler, satellite_data_manager):
            logger.info(
                "Satellite image needs to be an object of the satellite_data_manager class"
            )
            return
        bands_to_mask = []
        bands = [b_nir, b_red, b_blue, b_swir]
        if which_evi == 'evi2':
            bands = [b_nir, b_red, b_swir]    
        for btm in bands:
            if btm is not None:
                bands_to_mask.append(btm)
        if mask_bad_pixels:
            if bands_to_mask == []:
                handler.mask_bad_pixels()
            else:
                handler.mask_bad_pixels(bands_to_mask)

        if which_evi in ["evi", "evi2"]:
            nir = handler.sat_img[b_nir]
            red = handler.sat_img[b_red]
            swir = handler.sat_img[b_swir]
            if which_evi == "evi":
                blue = handler.sat_img[b_blue]
                temp_evi = (
                    evi_params["g"]
                    * (nir - red)
                    / (
                        nir
                        + evi_params["c1"] * red
                        - evi_params["c2"] * blue
                        + evi_params["l"]
                    )
                )
            elif which_evi == "evi2":
                temp_evi = (
                    evi2_params["g"]
                    * (nir - red)
                    / (nir + evi2_params["c"] * red + evi2_params["l"])
                )
            temp_evi = xr.where((temp_evi <= 0) | (temp_evi > 1), np.nan, temp_evi)
            temp_lswi = (nir - swir) / (nir + swir)
            temp_lswi = xr.where((temp_lswi < -1) | (temp_lswi > 1), np.nan, temp_lswi)
            handler.sat_img["evi"] = temp_evi
            handler.sat_img["lswi"] = temp_lswi
        if timestamp_key is not None:
            handler.sat_img = handler.sat_img.rename({timestamp_key: "timestamps"})

        bands_to_mask = []
        if which_evi in ["evi", "evi2"]:
            bands_to_mask = ["evi", "lswi"]
        else:
            for btm in [b_nir, b_red, b_blue, b_swir]:
                if btm is not None:
                    bands_to_mask.append(btm)
        if mask_snow:
            if bands_to_mask == []:
                handler.mask_snow()
            else:
                handler.mask_snow(bands_to_mask)
        if mask_clouds:
            if bands_to_mask == []:
                handler.mask_clouds()
            else:
                handler.mask_clouds(bands_to_mask)
        if drop_bands:
            if isinstance(drop_bands, list):
                drop_keys = drop_bands
                handler.sat_img = handler.sat_img.drop(drop_keys)
            else:
                handler.drop_bands()
        self.sat_imgs.append(handler)
        return

    def smearing(self, keys, kernel, sat_img=None, lonlats=None):
        """
        By default performs a spatial smearing on the list of pre-loaded satellite images.
        If sat_img is given the smearing is performed on that specific image.

            Parameters:
                    kernel (tuple): The extension of the spatial smoothing
                    lonlats (str): If given the smearing is only performed at the
                                   given lats and lons
                    keys (list): keys for the smoothign of the satellite images
            Returns:
                    None
        """

        if isinstance(kernel, tuple):
            arsz = int(3 * np.max(kernel))
            kernel = np.expand_dims(
                np.ones(shape=kernel) / np.sum(np.ones(shape=kernel)), 0
            )
        else:
            kernel = np.expand_dims(kernel.array, 0)
            arsz = int(3 * np.max(np.shape(kernel)))
        if lonlats is None:
            for key in keys:
                self.sat_imgs.sat_img[key][:, :] = convolve(
                    self.sat_imgs.sat_img[key].values[:, :, :],
                    kernel=kernel,
                    preserve_nan=True,
                )
        else:
            t = Transformer.from_crs(
                "+proj=longlat +datum=WGS84", self.sat_imgs.sat_img.rio.crs
            )
            xs = self.sat_imgs.sat_img.coords["x"].values
            ys = self.sat_imgs.sat_img.coords["y"].values
            for ll in lonlats:
                x, y = t.transform(ll[0], ll[1])
                x_ind = np.argmin(np.abs(x - xs))
                y_ind = np.argmin(np.abs(y - ys))
                for key in keys:
                    logger.info(key)
                    self.sat_imgs.sat_img[key][
                        :, y_ind - arsz : y_ind + arsz, x_ind - arsz : x_ind + arsz
                    ] = convolve(
                        self.sat_imgs.sat_img[key][
                            :, y_ind - arsz : y_ind + arsz, x_ind - arsz : x_ind + arsz
                        ],
                        kernel=kernel,
                        preserve_nan=True,
                    )
        return

    def reduce_along_lat_lon(self):
        self.sat_imgs.reduce_along_lon_lat(
            lon=[i[0] for i in self.lonlats],
            lat=[i[1] for i in self.lonlats],
            new_dim_name="site_names",
            interp_method="nearest",
        )
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign_coords(
            {"site_names": [i.get_site_name() for i in self.sites]}
        )

    def sort_and_merge_by_timestamp(self):
        """
        Called after adding the satellite images with 'add_sat_img'. Sorts the satellite
        images by timestamp and merges everything to one satellite_data_manager.

            Parameters:

            Returns:
                    None
        """
        if self.sites is None:
            x_time_y = 0
            for h in self.sat_imgs:
                size_dict = dict(h.sat_img.sizes)
                prod = np.prod([size_dict[i] for i in size_dict.keys()])
                if prod > x_time_y:
                    biggest = h
                    x_time_y = prod
            self.prototype = copy.deepcopy(biggest)
            keys = list(self.prototype.sat_img.keys())
            self.prototype.sat_img = self.prototype.sat_img.drop(keys)
        #  for h in self.sat_imgs:
        #      h.sat_img = h.sat_img.rio.reproject_match(self.prototype.sat_img, nodata=np.nan)
        else:
            self.prototype = copy.deepcopy(self.sat_imgs[0])
            keys = list(self.prototype.sat_img.keys())
            self.prototype.sat_img = self.prototype.sat_img.drop(keys)
        self.sat_imgs = satellite_data_manager(
            sat_img=xr.concat([k.sat_img for k in self.sat_imgs], "time")
        )
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.sortby(self.sat_imgs.sat_img.time)
        self.timestamps = self.sat_imgs.sat_img.time
        self.timestamps = np.array(
            [pd.Timestamp(i).to_pydatetime() for i in self.timestamps.values]
        )
        self.timestamp_start = self.timestamps[0]
        self.timestamp_end = self.timestamps[-1]
        self.tot_num_days = (self.timestamp_end - self.timestamp_start).days
        logger.info(
            "Loaded data from {} to {}".format(self.timestamp_start, self.timestamp_end)
        )
        day_steps = [i.days for i in (self.timestamps - self.timestamp_start)]
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign_coords({"time": day_steps})
        self.prototype.sat_img = self.prototype.sat_img.assign_coords(
            {"time": day_steps}
        )

        if "timestamps" in list(self.sat_imgs.sat_img.keys()):
            tismp = np.round(
                np.array(
                    (
                        self.sat_imgs.sat_img["timestamps"].values
                        - np.datetime64(self.timestamp_start)
                    )
                    / 1e9
                    / (24 * 60 * 60),
                    dtype=float,
                )
            )
            dims = list(self.sat_imgs.sat_img.data_vars["timestamps"].dims)
            self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign(
                {"timestamps": (dims, tismp)}
            )
        self.time_key = "time"
        if "evi" in list(self.sat_imgs.sat_img.data_vars):
            self.sat_imgs.sat_img["evi"] = xr.where(
                (self.sat_imgs.sat_img["evi"] == np.inf),
                self.sat_imgs.sat_img["evi"].min(dim=self.time_key),
                self.sat_imgs.sat_img["evi"],
            )

        if "lswi" in list(self.sat_imgs.sat_img.data_vars):
            self.sat_imgs.sat_img["lswi"] = xr.where(
                (self.sat_imgs.sat_img["lswi"] == np.inf),
                self.sat_imgs.sat_img["lswi"].min(dim=self.time_key),
                self.sat_imgs.sat_img["lswi"],
            )
        return

    def clip_to_box(self, sat_to_crop):
        bounds = sat_to_crop.sat_img.rio.bounds()
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.rio.clip_box(
            bounds[0], bounds[1], bounds[2], bounds[3]
        )
        keys = list(self.sat_imgs.sat_img.keys())
        self.prototype = satellite_data_manager(
            sat_img=self.sat_imgs.sat_img.drop(keys)
        )
        return

    def add_land_cover_map(
        self,
        land_cover_map,
        var_name="band_1",
        save_path=None,
        filter_size=None,
        mode="fractional",
        regridder_save_path=None,
        n_cpus=None,
        mpi=True,
        logs=False,
    ):
        """
        Add the land cover map. Either use a pre-calculated one or do the calculation on the fly.

            Parameters:
                    land_cover_map (str or satimg instance): The input land cover map.
                                                             If string, assume it's a pre-generated map
                    var_name (str): Name of the land_cover_band in the xarray dataset
                    save_path (str): Path to save the map. Can be useful for re-using
                    filter_size (int): Number of pixels from which the land cover type is aggregated.
            Returns:
                    None
        """

        if n_cpus is None:
            n_cpus = self.n_cpus
        if isinstance(land_cover_map, str):
            logger.info("Load pre-generated land cover map: {}".format(land_cover_map))
            self.land_cover_type = satellite_data_manager(sat_img=land_cover_map)
        else:
            logger.info("Generating satellite data compatible land cover map")

            for key in self.map_to_vprm_class.keys():
                land_cover_map.sat_img[var_name] = xr.where(
                    land_cover_map.sat_img[var_name] == key,
                    self.map_to_vprm_class[key],
                    land_cover_map.sat_img[var_name],
                )
            # land_cover_map.sat_img[var_name].values[land_cover_map.sat_img[var_name].values==key] = self.map_to_vprm_class[key]

            if mode == "fractional":
                import xesmf as xe

                veg_inds = np.unique(
                    [self.map_to_vprm_class[i] for i in self.map_to_vprm_class.keys()]
                )
                if not os.path.exists(regridder_save_path):
                    src_grid = to_esmf_grid(land_cover_map.sat_img)
                    ds_out = to_esmf_grid(self.sat_imgs.sat_img)
                    src_temp_path = os.path.join(
                        os.path.dirname(regridder_save_path),
                        "{}.nc".format(str(uuid.uuid4())),
                    )
                    dest_temp_path = os.path.join(
                        os.path.dirname(regridder_save_path),
                        "{}.nc".format(str(uuid.uuid4())),
                    )
                    src_grid.to_netcdf(src_temp_path)
                    ds_out.to_netcdf(dest_temp_path)
                    exec_str = "ESMF_RegridWeightGen --source {} --destination {} --weight {} -m conserve -r --netcdf4 –src_regional –dest_regional ".format(
                        src_temp_path, dest_temp_path, regridder_save_path
                    )
                    if mpi is True:
                        exec_str = "mpirun -np {} ".format(n_cpus) + exec_str
                    if not logs:
                        exec_str += " --no_log "
                    logger.info("Run: {}".format(exec_str))
                    os.system(exec_str)
                    os.remove(src_temp_path)
                    os.remove(dest_temp_path)
                grid1_xesmf = make_xesmf_grid(land_cover_map.sat_img)
                grid2_xesmf = make_xesmf_grid(self.sat_imgs.sat_img)
                for i in veg_inds:
                    land_cover_map.sat_img["veg_{}".format(i)] = (
                        ["y", "x"],
                        xr.where(
                            land_cover_map.sat_img[var_name].values == i, 1.0, 0.0
                        ),
                    )
                regridder = xe.Regridder(
                    grid1_xesmf,
                    grid2_xesmf,
                    "conservative",
                    weights=regridder_save_path,
                    reuse_weights=True,
                )
                handler = regridder(land_cover_map.sat_img)
                handler = handler.assign_coords(
                    {
                        "x": self.sat_imgs.sat_img.coords["x"].values,
                        "y": self.sat_imgs.sat_img.coords["y"].values,
                    }
                )
                self.land_cover_type = satellite_data_manager(sat_img=handler)

            else:
                if (
                    land_cover_map.sat_img.rio.crs.to_proj4()
                    != self.sat_imgs.sat_img.rio.crs.to_proj4()
                ):
                    logger.info(
                        "Projection of land cover map and satellite images need to match. Reproject first."
                    )
                    return False
                f_array = np.zeros(
                    np.shape(land_cover_map.sat_img[var_name].values), dtype=np.int16
                )
                count_array = np.zeros(
                    np.shape(land_cover_map.sat_img[var_name].values), dtype=np.int16
                )
                if filter_size is None:
                    filter_size = int(
                        np.ceil(
                            self.sat_imgs.sat_img.rio.resolution()[0]
                            / land_cover_map.get_resolution()
                        )
                    )
                    logger.info("Filter size {}:".format(filter_size))
                if filter_size <= 1:
                    filter_size = 1
                for i in veg_inds:
                    mask = np.array(
                        land_cover_map.sat_img[var_name].values == i, dtype=np.float64
                    )
                    ta = scipy.ndimage.uniform_filter(
                        mask, size=(filter_size, filter_size)
                    ) * (filter_size**2)
                    f_array[ta > count_array] = i
                    count_array[ta > count_array] = ta[ta > count_array]
                f_array[f_array == 0] = (
                    8  # 8 is Category for nothing | alternatively np.nan?
                )
                land_cover_map.sat_img[var_name].values = f_array
                del ta
                del count_array
                del f_array
                del mask
                t = (
                    land_cover_map.sat_img.sel(
                        x=self.sat_imgs.sat_img.x.values,
                        y=self.sat_imgs.sat_img.y.values,
                        method="nearest",
                    )
                    .to_array()
                    .values[0]
                )
                self.land_cover_type = copy.deepcopy(self.prototype)
                self.land_cover_type.sat_img = self.land_cover_type.sat_img.assign(
                    {var_name: (["y", "x"], t)}
                )
                for i in veg_inds:
                    self.land_cover_type.sat_img["veg_{}".format(i)] = (
                        ["y", "x"],
                        xr.where(mm.sat_img[var_name].values == i, 1.0, 0.0),
                    )
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.drop_vars(
                [var_name]
            )
            var_list = list(dict(self.land_cover_type.sat_img.data_vars.dtypes).keys())
            self.land_cover_type.sat_img = xr.concat(
                [self.land_cover_type.sat_img[var] for var in var_list],
                dim="vprm_classes",
            )
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.assign_coords(
                {"vprm_classes": [int(c.split("_")[1]) for c in list(var_list)]}
            )
            if save_path is not None:
                self.land_cover_type.save(save_path)
        return

    def calc_min_max_evi_lswi(self):
        """
        Calculate the minimim and maximum EVI and LSWI
            Parameters:
                    None
            Returns:
                    None
        """
        self.max_lswi = copy.deepcopy(self.prototype)
        self.min_lswi = copy.deepcopy(self.prototype)
        self.min_max_evi = copy.deepcopy(self.prototype)
        shortcut = self.sat_imgs.sat_img
        # if self.sites is None:
        self.min_lswi.sat_img["min_lswi"] = shortcut["lswi"].min(
            self.time_key, skipna=True
        )
        self.min_max_evi.sat_img["min_evi"] = shortcut["evi"].min(
            self.time_key, skipna=True
        )
        self.min_max_evi.sat_img["max_evi"] = shortcut["evi"].max(
            self.time_key, skipna=True
        )
        # Set growing season threshold to 20% of the difference between max and min value. This should be studied in more detail
        # self.max_lswi.sat_img['growing_season_th'] = shortcut['evi'].min(self.time_key, skipna=True)  + 0.3 * ( shortcut['evi'].max(self.time_key, skipna=True) - shortcut['evi'].min(self.time_key, skipna=True))
        self.min_max_evi.sat_img["th"] = shortcut["evi"].min(
            self.time_key, skipna=True
        ) + 0.55 * (
            shortcut["evi"].max(self.time_key, skipna=True)
            - shortcut["evi"].min(self.time_key, skipna=True)
        )
        return

    def lowess(self, keys, lonlats=None, times=False, frac=0.25, it=3, n_cpus=None):
        """
        Performs the lowess smoothing

            Parameters:
                    lonlats (str): If given the smearing is only performed at the
                                   given lats and lons
            Returns:
                    None
        """
        self.sat_imgs.sat_img.load()

        if n_cpus is None:
            n_cpus = self.n_cpus
        if isinstance(times, pd.core.indexes.datetimes.DatetimeIndex):
            times = list(times)
        if isinstance(times, list):
            times = np.array(sorted(times))
            if (times[-1] > self.timestamp_end) | (times[0] < self.timestamp_start):
                logger.info(
                    "You have provied some timestamps that are not covered from satellite images.\
                They will be ignored in the following, to avoid unreliable results"
                )
            times = times[
                (times <= self.timestamp_end) & (times >= self.timestamp_start)
            ]
            xvals = [
                int(
                    np.round(
                        (i - self.timestamp_start).total_seconds() / (24 * 60 * 60)
                    )
                )
                for i in times
            ]
        elif isinstance(times, str):
            if times == "daily":
                xvals = np.arange(self.tot_num_days)
            else:
                logger.info("{} is not a valid str for times".format(times))
                return
        else:
            xvals = self.sat_imgs.sat_img["time"]
        logger.info("Lowess timestamps {}".format(xvals))

        if self.sites is not None:  # Is flux tower sites are given
            if "timestamps" in list(self.sat_imgs.sat_img.data_vars):
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign(
                        {
                            key: (
                                ["time_gap_filled", "site_names"],
                                np.array(
                                    [
                                        do_lowess_smoothing(
                                            self.sat_imgs.sat_img.sel(site_names=i)[
                                                key
                                            ].values,
                                            timestamps=self.sat_imgs.sat_img.sel(
                                                site_names=i
                                            )["timestamps"].values,
                                            xvals=xvals,
                                            frac=frac,
                                            it=it,
                                        )
                                        for i in self.sat_imgs.sat_img.site_names.values
                                    ]
                                ).T,
                            )
                        }
                    )
            else:
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign(
                        {
                            key: (
                                ["time_gap_filled", "site_names"],
                                np.array(
                                    [
                                        do_lowess_smoothing(
                                            self.sat_imgs.sat_img.sel(site_names=i)[
                                                key
                                            ].values,
                                            timestamps=self.sat_imgs.sat_img[
                                                "time"
                                            ].values,
                                            xvals=xvals,
                                            frac=frac,
                                            it=it,
                                        )
                                        for i in self.sat_imgs.sat_img.site_names.values
                                    ]
                                ).T,
                            )
                        }
                    )

        elif lonlats is None:  # If smoothing the entire array
            if "timestamps" in list(self.sat_imgs.sat_img.data_vars):
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign(
                        {
                            key: (
                                ["time_gap_filled", "y", "x"],
                                np.array(
                                    Parallel(n_jobs=n_cpus, max_nbytes=None)(
                                        delayed(do_lowess_smoothing)(
                                            self.sat_imgs.sat_img[key][:, :, i].values,
                                            timestamps=self.sat_imgs.sat_img[
                                                "timestamps"
                                            ][:, :, i].values,
                                            xvals=xvals,
                                            frac=frac,
                                            it=it,
                                        )
                                        for i, x_coord in enumerate(
                                            self.sat_imgs.sat_img.x.values
                                        )
                                    )
                                ).T,
                            )
                        }
                    )
            else:
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign(
                        {
                            key: (
                                ["time_gap_filled", "y", "x"],
                                np.array(
                                    Parallel(n_jobs=n_cpus, max_nbytes=None)(
                                        delayed(do_lowess_smoothing)(
                                            self.sat_imgs.sat_img[key][:, :, i].values,
                                            timestamps=self.sat_imgs.sat_img[
                                                "time"
                                            ].values,
                                            xvals=xvals,
                                            frac=frac,
                                            it=it,
                                        )
                                        for i, x_coord in enumerate(
                                            self.sat_imgs.sat_img.x.values
                                        )
                                    )
                                ).T,
                            )
                        }
                    )

        else:
            logger.info("Not implemented")
            # Originally had a function to smooth only at specific lat/long.
            # That doesn't make sense anymore, because the time dimension will change through lowess smoothing.
            # If this is your plan then try to crop the sat image first and then do the lowess filtering.

        self.time_key = "time_gap_filled"
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign_coords(
            {"time_gap_filled": list(xvals)}
        )
        return

    def clip_values(self, key, min_val, max_val):
        self.sat_imgs.sat_img[key].values[
            self.sat_imgs.sat_img[key].values < min_val
        ] = min_val
        self.sat_imgs.sat_img[key].values[
            self.sat_imgs.sat_img[key].values > max_val
        ] = max_val
        return

    def clip_non_finite(self, data_var, val, sel):
        t = self.sat_imgs.sat_img[data_var].loc[sel].values
        t[~np.isfinite(t)] = val
        self.sat_imgs.sat_img[data_var].loc[sel] = t
        return

    def get_current_timestamp(self):
        return self.timestamps[self.counter]

    def get_evi(self, lon=None, lat=None, site_name=None):
        """
        Get EVI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    EVI array
        """

        # if (self.new is False) & ('evi' in self.buffer.keys()):
        #     return self.buffer['evi']
        if site_name is not None:
            self.buffer["evi"] = float(
                self.sat_imgs.sat_img.sel(site_names=site_name).isel(
                    {self.time_key: self.counter}
                )["evi"]
            )
        elif lon is not None:
            self.buffer["evi"] = self.sat_imgs.value_at_lonlat(
                lon, lat, as_array=False, key="evi", isel={self.time_key: self.counter}
            )
        else:
            self.buffer["evi"] = self.sat_imgs.sat_img["evi"].isel(
                {self.time_key: self.counter}
            )
        return self.buffer["evi"]

    def get_lswi(self, lon=None, lat=None, site_name=None):
        """
        Get LSWI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    LSWI array
        """

        # if (self.new is False) & ('lswi' in self.buffer.keys()):
        #     return self.buffer['lswi']
        if site_name is not None:
            self.buffer["lswi"] = float(
                self.sat_imgs.sat_img.sel(site_names=site_name).isel(
                    {self.time_key: self.counter}
                )["lswi"]
            )
        elif lon is not None:
            self.buffer["lswi"] = self.sat_imgs.value_at_lonlat(
                lon, lat, as_array=False, key="lswi", isel={self.time_key: self.counter}
            )
        else:
            self.buffer["lswi"] = self.sat_imgs.sat_img["lswi"].isel(
                {self.time_key: self.counter}
            )
        return self.buffer["lswi"]

    def get_sat_img_values_from_key(self, key, lon=None, lat=None, counter_range=None):
        """
        Get EVI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    EVI array
        """

        if self.new is False:
            return self.buffer[key]
        if counter_range is None:
            select_dict = {self.time_key: self.counter}
        else:
            select_dict = {self.time_key: counter_range}

        if lon is not None:
            self.buffer[key] = self.sat_imgs.value_at_lonlat(
                lon, lat, as_array=False, key=key, isel=select_dict
            ).values.flatten()
        else:
            self.buffer[key] = self.sat_imgs.sat_img[key].isel(select_dict)
        return self.sat_img_buffer[key]

    def get_sat_img_values_for_all_keys(self, lon=None, lat=None, counter_range=None):
        """
        Get EVI for the current satellite image (see self.counter)

            Parameters:
                    lon (float): longitude
                    lat (float): latitude
            Returns:
                    EVI array
        """

        if self.new is False:
            return self.buffer["all_sat_keys"]

        if counter_range is None:
            select_dict = {self.time_key: self.counter}
        else:
            select_dict = {self.time_key: counter_range}

        if lon is not None:
            self.buffer["all_sat_keys"] = self.sat_imgs.value_at_lonlat(
                lon, lat, as_array=True, isel=select_dict
            )
        else:
            self.buffer["all_sat_keys"] = self.sat_imgs.sat_img.isel(select_dict)
        return self.buffer["all_sat_keys"]

    def _set_sat_img_counter(self, datetime_utc):
        days_after_first_image = (
            datetime_utc - self.timestamp_start
        ).total_seconds() / (24 * 60 * 60)
        counter_new = np.argmin(
            np.abs(self.sat_imgs.sat_img[self.time_key].values - days_after_first_image)
        )
        if (days_after_first_image < 0) | (
            days_after_first_image > self.sat_imgs.sat_img[self.time_key][-1]
        ):
            # logger.info('No data for {}'.format(datetime_utc))
            self.counter = 0
            return False
        elif counter_new != self.counter:
            self.new = True
            self.counter = counter_new
            return
        else:
            self.new = False
            return  # Still same satellite image

    def _set_prototype_lat_lon(self):
        src_x = self.prototype.sat_img.coords["x"].values
        src_y = self.prototype.sat_img.coords["y"].values
        X, Y = np.meshgrid(src_x, src_y)
        t = Transformer.from_crs(
            self.prototype.sat_img.rio.crs, "+proj=longlat +datum=WGS84"
        )
        x_long, y_lat = t.transform(X, Y)
        self.prototype_lat_lon = xr.Dataset(
            {
                "lon": (["y", "x"], x_long, {"units": "degrees_east"}),
                "lat": (["y", "x"], y_lat, {"units": "degrees_north"}),
            }
        )
        self.prototype_lat_lon = self.prototype_lat_lon.set_coords(["lon", "lat"])
        return

    def save(self, save_path):
        """
        Save the LSWI and EVI satellite image. ToDo
        """
        self.sat_imgs.save(save_path)
        return

    def is_disjoint(self, this_sat_img):
        bounds = self.prototype.sat_img.rio.transform_bounds(
            this_sat_img.sat_img.rio.crs
        )
        dj = rasterio.coords.disjoint_bounds(bounds, this_sat_img.sat_img.rio.bounds())
        return dj

    def add_vprm_insts(self, vprm_insts, allow_reproject=True):
        # Add Check that timestamps align before merging
        if isinstance(self.sat_imgs, satellite_data_manager):
            self.sat_imgs.add_tile(
                [v.sat_imgs for v in vprm_insts], reproject=allow_reproject
            )
            keys = list(self.sat_imgs.sat_img.keys())
            self.prototype = satellite_data_manager(
                sat_img=self.sat_imgs.sat_img.drop(keys)
            )

        if self.land_cover_type is not None:
            self.land_cover_type.add_tile(
                [v.land_cover_type for v in vprm_insts], reproject=False
            )
