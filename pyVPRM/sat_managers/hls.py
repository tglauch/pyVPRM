import xarray as xr
import numpy as np
from datetime import datetime
from pyVPRM.sat_managers.base_manager import satellite_data_manager
from loguru import logger


class hls(satellite_data_manager):
    """
    Class to load/manage Harmonized Landsat Sentinel-2 (HLS) data. Mirrors
    the sentinel2 manager's interface, but masking is done via bits in the
    single Fmask QA band instead of Sentinel-2's categorical SCL layer.

    Fmask bit layout (HLS v2.0 product spec) - bits 6-7 are a 2-bit aerosol
    level (00 climatology, 01 low, 10 moderate, 11 high), not a mask flag:
        bit 0: cirrus (reserved in v2.0, not actually populated)
        bit 1: cloud
        bit 2: adjacent to cloud/shadow
        bit 3: cloud shadow
        bit 4: snow/ice
        bit 5: water

    Actual downloading/stacking (STAC search across the hls2-l30/hls2-s30
    collections, band harmonization, reflectance scaling) happens upstream
    in satellite_sources.fetch_hls_stack() - an hls instance here is
    constructed directly from that already-assembled cube
    (hls(sat_img=cube)), so download()/individual_loading() aren't needed
    the way they are for the legacy Sentinel-2 Copernicus-Hub flow.
    """

    FMASK_BITS = {
        "cirrus": 0,
        "cloud": 1,
        "adjacent_cloud_shadow": 2,
        "cloud_shadow": 3,
        "snow_ice": 4,
        "water": 5,
    }
    HIGH_AEROSOL_LEVEL = 0b11  # bits 6-7 == 11 -> high aerosol; LP DAAC recommends excluding

    def __init__(self, datapath=None, sat_image_path=None, sat_img=None):
        super().__init__(datapath, sat_image_path)
        self.load_kwargs = {}
        if sat_img is not None:
            self.sat_img = sat_img
        return

    def set_band_names(self):
        logger.info("Trying to set reflectance bands assuming harmonized HLS naming")
        bands = []
        for i in list(self.sat_img.data_vars):
            if any(name in i for name in ("red", "blue", "green", "swir", "nir", "rededge")):
                bands.append(i)
        self.bands = bands

    def _fmask_bit_set(self, bit_name):
        """Boolean array: True wherever the given Fmask bit is set."""
        fmask = self.sat_img["Fmask"].astype("uint16")
        bit = self.FMASK_BITS[bit_name]
        return ((fmask >> bit) & 1).astype(bool)

    def mask_bad_pixels(self, bands=None):
        """
        Broad QA mask: cloud, adjacent-to-cloud/shadow, cloud shadow, and
        snow/ice - the Fmask equivalent of sentinel2.mask_bad_pixels()'s
        no-data/saturated/cloud-shadow/cloud-medium/cirrus SCL sweep.
        """
        if bands is None:
            bands = self.bands
        is_bad = (
            self._fmask_bit_set("cloud")
            | self._fmask_bit_set("adjacent_cloud_shadow")
            | self._fmask_bit_set("cloud_shadow")
            | self._fmask_bit_set("snow_ice")
        )
        self.sat_img[bands] = xr.where(is_bad, np.nan, self.sat_img[bands])
        return

    def mask_water(self, bands=None):
        # Fmask's water bit is a per-scene spectral detection, not a stable
        # land-cover label like SCL's water class - fine for masking a
        # single scene, but for a *permanent* water body mask across an
        # entire time series, take a majority vote instead (see
        # mask_hls()'s dominant-water logic in satellite_sources.py).
        if bands is None:
            bands = self.bands
        self.sat_img[bands] = xr.where(
            self._fmask_bit_set("water"), np.nan, self.sat_img[bands]
        )
        return

    def mask_clouds(self, bands=None):
        if bands is None:
            bands = self.bands
        is_cloud = (
            self._fmask_bit_set("cloud")
            | self._fmask_bit_set("cirrus")
            | self._fmask_bit_set("adjacent_cloud_shadow")
            | self._fmask_bit_set("cloud_shadow")
        )
        self.sat_img[bands] = xr.where(is_cloud, np.nan, self.sat_img[bands])
        self.mask_high_aerosol(bands=bands)
        return

    def mask_snow(self, bands=None):
        if bands is None:
            bands = self.bands
        # Matches sentinel2.mask_snow()'s convention of +inf rather than NaN,
        # so snow pixels stay distinguishable/filterable downstream instead
        # of being indistinguishable from other missing-data NaNs.
        self.sat_img[bands] = xr.where(
            self._fmask_bit_set("snow_ice"), np.inf, self.sat_img[bands]
        )
        return

    def mask_high_aerosol(self, bands=None):
        """
        HLS-specific - no Sentinel-2 SCL equivalent. Bits 6-7 encode aerosol
        level; LP DAAC recommends excluding pixels flagged high aerosol
        (both bits set), since reflectance is unreliable there.
        """
        if bands is None:
            bands = self.bands
        fmask = self.sat_img["Fmask"].astype("uint16")
        aerosol_level = (fmask >> 6) & 0b11
        is_high_aerosol = aerosol_level == self.HIGH_AEROSOL_LEVEL
        self.sat_img[bands] = xr.where(is_high_aerosol, np.nan, self.sat_img[bands])
        return

    def qa_mask(self, bit_names):
        """
        Read-only combined boolean mask (dims match Fmask: time, y, x) -
        True wherever ANY of the given Fmask bits is set. Unlike
        mask_clouds()/mask_snow() above (which mutate self.sat_img in
        place), this just returns the boolean array so callers - e.g.
        satellite_sources.mask_hls(), applying the mask to a *different*
        dataset (vprm_inst.sat_imgs.sat_img's computed indices) - can
        combine it however they need without this class knowing about
        that downstream dataset.
        """
        mask = None
        for name in bit_names:
            bit_mask = self._fmask_bit_set(name)
            mask = bit_mask if mask is None else (mask | bit_mask)
        return mask

    def water_mask(self):
        """Read-only per-scene water-bit boolean mask (see qa_mask() docstring)."""
        return self._fmask_bit_set("water")

    # mask_non_vegetated() has no Fmask equivalent - Fmask has no land-cover
    # classes beyond water, unlike SCL's not_vegetated/vegetation/etc, so
    # it's intentionally not implemented here.

    def get_recording_time(self):
        if "time" in list(self.sat_img.coords):
            return self.sat_img.coords["time"].values
        else:
            raise NotImplementedError(
                "hls sat_img has no 'time' coordinate - construct it from "
                "the cube returned by fetch_hls_stack(), which always keeps "
                "a time dimension/coordinate."
            )