"""Tests for Hardiman-style UrbanVPRM respiration adjustments."""

import numpy as np
import xarray as xr

from pyVPRM.vprm_models.vprm_urban_model import vprm_urban_model


def test_urban_respiration_uses_isa_and_reference_evi():
    """Apply Hardiman Supplement Eqs. 6--8 to an urban class.

    Returns
    -------
    None
        The test verifies urban heterotrophic and autotrophic respiration and
        confirms non-urban classes retain base-VPRM respiration.
    """
    model = object.__new__(vprm_urban_model)
    model.urban_vprm_classes = frozenset({10, 11})
    model.fit_params_dict = {
        3: {"alpha": 0.2, "beta": -1.0},
        10: {"alpha": 0.2, "beta": -1.0},
    }
    inputs = {
        "tcorr": xr.DataArray([10.0]),
        "evi": xr.DataArray([0.3]),
        "ISA": xr.DataArray([0.4]),
        "evi_ref": xr.DataArray([0.5]),
        "min_evi_ref": xr.DataArray([0.1]),
    }

    urban_respiration = model._calculate_respiration(1.0, 10, inputs)
    non_urban_respiration = model._calculate_respiration(1.0, 3, inputs)

    re_initial = 1.0
    expected_urban = (1.0 - 0.4) * re_initial / 2.0 + (
        (0.3 + 0.1 * 0.4) / 0.5 * re_initial / 2.0
    )
    xr.testing.assert_allclose(urban_respiration, xr.DataArray([expected_urban]))
    xr.testing.assert_allclose(non_urban_respiration, xr.DataArray([re_initial]))
