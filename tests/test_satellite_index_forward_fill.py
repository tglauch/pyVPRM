"""Tests for bounded forward filling of satellite indices."""

import numpy as np
import xarray as xr

from pyVPRM.VPRM import vprm_preprocessor
from pyVPRM.sat_managers.base_manager import satellite_data_manager


def test_forward_fill_satellite_indices_respects_maximum_age():
    """Fill only gaps supported by a sufficiently recent observation.

    Returns
    -------
    None
        The test verifies per-pixel EVI and LSWI filling, preservation of
        leading gaps, and rejection of observations older than the limit.
    """
    preprocessor = object.__new__(vprm_preprocessor)
    preprocessor.time_key = "time"
    satellite_indices = xr.Dataset(
        {
            "evi": (
                ("time", "y", "x"),
                np.array(
                    [
                        [[0.2, np.nan], [np.nan, np.nan]],
                        [[np.nan, 0.4], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[0.8, np.nan], [np.nan, np.nan]],
                    ]
                ),
            ),
            "lswi": (
                ("time", "y", "x"),
                np.array(
                    [
                        [[0.1, np.nan], [np.nan, np.nan]],
                        [[np.nan, 0.3], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[0.7, np.nan], [np.nan, np.nan]],
                    ]
                ),
            ),
        },
        coords={
            "time": [0.0, 8.0, 16.0, 32.0],
            "y": [200.0, 100.0],
            "x": [100.0, 200.0],
        },
    )
    preprocessor.sat_imgs = satellite_data_manager(sat_img=satellite_indices)

    preprocessor.forward_fill_satellite_indices(max_age_days=12)

    filled_evi = preprocessor.sat_imgs.sat_img["evi"]
    filled_lswi = preprocessor.sat_imgs.sat_img["lswi"]
    assert filled_evi.sel(time=8.0, y=200.0, x=100.0).item() == 0.2
    assert np.isnan(filled_evi.sel(time=16.0, y=200.0, x=100.0).item())
    assert filled_evi.sel(time=16.0, y=200.0, x=200.0).item() == 0.4
    assert np.isnan(filled_evi.sel(time=32.0, y=200.0, x=200.0).item())
    assert np.isnan(filled_evi.sel(time=0.0, y=200.0, x=200.0).item())
    assert filled_lswi.sel(time=8.0, y=200.0, x=100.0).item() == 0.1
