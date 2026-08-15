"""Regression tests for missing classwise VPRM flux aggregation."""

from types import MethodType, SimpleNamespace

import numpy as np
import xarray as xr

from pyVPRM.vprm_models.vprm_base_model import vprm_base_model


def make_model(inputs):
    """Build a two-class model returning fixed VPRM input fields.

    Parameters
    ----------
    inputs : dict[str, xarray.DataArray]
        VPRM driver fields returned for each flux-capable class.

    Returns
    -------
    pyVPRM.vprm_models.vprm_base_model.vprm_base_model
        Minimally initialized model suitable for aggregation tests.
    """

    model = object.__new__(vprm_base_model)
    model.fit_params_dict = {
        3: {"lamb": 0.5, "par0": 1000.0, "alpha": 0.2, "beta": 1.0},
        4: {"lamb": 0.5, "par0": 1000.0, "alpha": 0.2, "beta": 1.0},
    }
    model.vprm_pre = SimpleNamespace(
        land_cover_type=SimpleNamespace(
            sat_img=xr.DataArray(
                [[0.75], [0.25]],
                dims=("vprm_classes", "x"),
                coords={"vprm_classes": [3, 4], "x": [0]},
            )
        )
    )

    def get_inputs(self, *args, **kwargs):
        """Return fixed drivers for each requested VPRM class.

        Parameters
        ----------
        *args
            Positional arguments accepted by the model hook.
        **kwargs
            Keyword arguments accepted by the model hook.

        Returns
        -------
        dict[str, xarray.DataArray]
            Fixed input fields supplied to the VPRM equations.
        """

        del self, args, kwargs
        return inputs

    model._get_vprm_variables = MethodType(get_inputs, model)
    return model


def make_inputs(evi, tcorr):
    """Build otherwise-valid VPRM inputs with selected missing drivers.

    Parameters
    ----------
    evi : float
        Enhanced Vegetation Index input value.
    tcorr : float
        Respiration temperature input value in degrees Celsius.

    Returns
    -------
    dict[str, xarray.DataArray]
        One-cell VPRM input fields.
    """

    def field(value):
        """Wrap a scalar as a one-cell xarray field.

        Parameters
        ----------
        value : float
            Scalar value to store.

        Returns
        -------
        xarray.DataArray
            One-dimensional field over the test cell.
        """

        return xr.DataArray([value], dims="x")

    return {
        "Ps": field(1.0),
        "Ws": field(1.0),
        "Ts": field(1.0),
        "evi": field(evi),
        "par": field(1000.0),
        "tcorr": field(tcorr),
    }


def test_all_missing_gpp_contributions_remain_missing():
    """Keep GPP and NEE missing when every classwise GPP is missing.

    Returns
    -------
    None
        The test verifies that an unresolved EVI is not silently converted to
        zero GPP during cross-class aggregation.
    """

    model = make_model(make_inputs(evi=np.nan, tcorr=20.0))
    fluxes = model.make_vprm_predictions(date=object())

    assert bool(fluxes["gpp"].isnull().all())
    assert bool(fluxes["nee"].isnull().all())


def test_all_missing_respiration_contributions_remain_missing():
    """Keep NEE missing when every classwise respiration is missing.

    Returns
    -------
    None
        The test verifies that a missing respiration temperature input is not
        silently converted to zero respiration during aggregation.
    """

    model = make_model(make_inputs(evi=0.5, tcorr=np.nan))
    fluxes = model.make_vprm_predictions(date=object())

    assert bool(fluxes["gpp"].notnull().all())
    assert bool(fluxes["nee"].isnull().all())
