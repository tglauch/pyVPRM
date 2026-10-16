"""Regression tests for satellite timestamp normalization."""

import numpy as np
import xarray as xr

from pyVPRM.VPRM import vprm_preprocessor
from pyVPRM.sat_managers.base_manager import satellite_data_manager


def make_satellite_scene(scene_time, pixel_time):
    """Build a one-pixel scene with microsecond-resolution acquisition time.

    Parameters
    ----------
    scene_time : str
        Scene timestamp interpreted as a NumPy datetime value.
    pixel_time : str
        Per-pixel acquisition timestamp interpreted at microsecond precision.

    Returns
    -------
    pyVPRM.sat_managers.base_manager.satellite_data_manager
        Scene containing one EVI value and its per-pixel timestamp.
    """

    dataset = xr.Dataset(
        data_vars={
            "evi": (("time", "y", "x"), np.ones((1, 1, 1))),
            "timestamps": (
                ("time", "y", "x"),
                np.array([[[np.datetime64(pixel_time, "us")]]]),
            ),
        },
        coords={
            "time": [np.datetime64(scene_time, "us")],
            "y": [0],
            "x": [0],
        },
    )
    return satellite_data_manager(sat_img=dataset)


def test_sort_and_merge_normalizes_microsecond_timestamps_to_days():
    """Convert microsecond timestamps to correct elapsed-day values.

    Returns
    -------
    None
        The test verifies that LOWESS receives elapsed days rather than all
        timestamps being rounded to zero under ``datetime64[us]`` input.
    """

    preprocessor = object.__new__(vprm_preprocessor)
    preprocessor.flux_tower_instances = None
    preprocessor.satellite_indices = ["evi"]
    preprocessor.sat_imgs = [
        make_satellite_scene("2022-01-01", "2022-01-01"),
        make_satellite_scene("2022-01-09", "2022-01-09"),
    ]

    preprocessor.sort_and_merge_by_timestamp(min_length_snow_period=None)

    np.testing.assert_array_equal(
        preprocessor.sat_imgs.sat_img["timestamps"].values[:, 0, 0], [0.0, 8.0]
    )
