"""Tests for the VPRM water-scalar calculation."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from pyVPRM.vprm_models.vprm_base_model import vprm_base_model


def test_get_w_scale_uses_published_formula_for_every_class():
    """Use the stable published water scalar for a former special-case class.

    Returns
    -------
    None
        The test verifies the formula remains finite even when annual LSWI
        extrema differ by only a tiny amount.
    """
    model = object.__new__(vprm_base_model)
    model.buffer = {}
    model.get_lswi = lambda lon, lat, site_name: xr.DataArray([[0.15]])
    model.vprm_pre = SimpleNamespace(
        max_lswi=SimpleNamespace(sat_img={"max_lswi": xr.DataArray([[0.150001]])}),
        min_lswi=SimpleNamespace(sat_img={"min_lswi": xr.DataArray([[0.15]])}),
    )

    result = model.get_w_scale(land_cover_type=4)

    expected = (1.0 + 0.15) / (1.0 + 0.150001)
    np.testing.assert_allclose(result, expected)
    assert np.isfinite(result).all()
