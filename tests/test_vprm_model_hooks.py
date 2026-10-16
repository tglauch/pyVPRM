"""Tests for overridable base-VPRM flux calculation hooks."""

import numpy as np
import xarray as xr

from pyVPRM.vprm_models.vprm_base_model import vprm_base_model


def test_base_model_flux_hooks_preserve_current_equations():
    """Match the original base-VPRM GPP and respiration calculations.

    Returns
    -------
    None
        The test confirms that extracted flux hooks preserve the former inline
        equations for a two-cell xarray input.
    """
    model = object.__new__(vprm_base_model)
    model.fit_params_dict = {
        3: {"lamb": 0.5, "par0": 1000.0, "alpha": 0.2, "beta": -1.0}
    }
    land_cover_fraction = xr.DataArray([0.25, 1.0], dims="x")
    inputs = {
        "Ps": xr.DataArray([0.5, 1.0], dims="x"),
        "Ws": xr.DataArray([0.8, 0.6], dims="x"),
        "Ts": xr.DataArray([0.9, 0.7], dims="x"),
        "evi": xr.DataArray([0.3, 0.4], dims="x"),
        "par": xr.DataArray([500.0, 1000.0], dims="x"),
        "tcorr": xr.DataArray([2.0, 10.0], dims="x"),
    }

    gpp = model._calculate_gpp(land_cover_fraction, 3, inputs)
    respiration = model._calculate_respiration(land_cover_fraction, 3, inputs)

    expected_gpp = land_cover_fraction * (
        0.5
        * inputs["Ps"]
        * inputs["Ws"]
        * inputs["Ts"]
        * inputs["evi"]
        * inputs["par"]
        / (1 + inputs["par"] / 1000.0)
    )
    expected_respiration = np.maximum(
        land_cover_fraction * (0.2 * inputs["tcorr"] - 1.0), 0
    )
    xr.testing.assert_identical(gpp, expected_gpp)
    xr.testing.assert_identical(respiration, expected_respiration)
