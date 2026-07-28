"""Tests for Hardiman-style UrbanVPRM respiration adjustments."""

import numpy as np
import pytest
import xarray as xr

from pyVPRM.sat_managers.base_manager import satellite_data_manager
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


def test_isa_fraction_clips_numerical_bound_overshoots():
    """Clip negligible percentage-bound overshoots before ISA conversion.

    Returns
    -------
    None
        The test verifies that conservative-regridding roundoff at zero and
        100 percent is clipped, while a material out-of-range value fails.
    """
    model = object.__new__(vprm_urban_model)
    model.vprm_pre = type("Preprocessor", (), {})()
    model.vprm_pre.impervious_surface_area = satellite_data_manager(
        sat_img=xr.Dataset(
            {
                "impervious_surface_percentage": xr.DataArray(
                    [[-1e-9, 100.0 + 1e-9]], dims=("y", "x")
                )
            }
        )
    )

    isa_fraction = model.get_isa_fraction()
    xr.testing.assert_allclose(
        isa_fraction, xr.DataArray([[0.0, 1.0]], dims=("y", "x"))
    )

    model.vprm_pre.impervious_surface_area.sat_img[
        "impervious_surface_percentage"
    ].data[0, 1] = 100.1
    model._isa_fraction_cache = None
    with pytest.raises(ValueError, match="maximum 100.1"):
        model.get_isa_fraction()


def test_reference_evi_lookup_uses_target_grid_coordinates():
    """Gather reference EVI without retaining source-grid index coordinates.

    Returns
    -------
    None
        The test verifies that two-dimensional reference indexers carrying
        target-grid coordinates produce target-grid reference EVI fields.
    """
    source_y = [10.0, 20.0]
    source_x = [30.0, 40.0]
    target_y = [100.0, 200.0]
    target_x = [300.0, 400.0]
    evi = xr.DataArray(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": source_y, "x": source_x},
    )
    min_evi = xr.DataArray(
        [[0.01, 0.02], [0.03, 0.04]],
        dims=("y", "x"),
        coords={"y": source_y, "x": source_x},
    )
    lookup = xr.Dataset(
        {
            "reference_y_index": xr.DataArray(
                [[1, 0], [0, 1]],
                dims=("y", "x"),
                coords={"y": target_y, "x": target_x},
            ),
            "reference_x_index": xr.DataArray(
                [[1, 0], [1, 0]],
                dims=("y", "x"),
                coords={"y": target_y, "x": target_x},
            ),
        }
    )
    model = object.__new__(vprm_urban_model)
    model.vprm_pre = type("Preprocessor", (), {})()
    model.vprm_pre.time_key = "time"
    model.vprm_pre.counter = 0
    model.vprm_pre.sat_imgs = satellite_data_manager(
        sat_img=evi.to_dataset(name="evi")
    )
    model.vprm_pre.min_max_evi = satellite_data_manager(
        sat_img=min_evi.to_dataset(name="min_evi")
    )
    model.vprm_pre.urban_reference_evi = satellite_data_manager(sat_img=lookup)

    reference_evi, minimum_reference_evi = model._get_reference_evi()

    xr.testing.assert_allclose(
        reference_evi,
        xr.DataArray(
            [[0.4, 0.1], [0.2, 0.3]],
            dims=("y", "x"),
            coords={"time": 0, "y": target_y, "x": target_x},
        ),
    )
    xr.testing.assert_allclose(
        minimum_reference_evi,
        xr.DataArray(
            [[0.04, 0.01], [0.02, 0.03]],
            dims=("y", "x"),
            coords={"y": target_y, "x": target_x},
        ),
    )

    model.vprm_pre.sat_imgs.sat_img["evi"].data[0, :, :] = 99.0
    cached_reference_evi, cached_minimum_reference_evi = model._get_reference_evi()
    xr.testing.assert_allclose(cached_reference_evi, reference_evi)
    xr.testing.assert_allclose(cached_minimum_reference_evi, minimum_reference_evi)

    model.vprm_pre.counter = 1
    refreshed_reference_evi, refreshed_minimum_reference_evi = (
        model._get_reference_evi()
    )
    xr.testing.assert_allclose(
        refreshed_reference_evi,
        xr.DataArray(
            [[0.8, 0.5], [0.6, 0.7]],
            dims=("y", "x"),
            coords={"time": 1, "y": target_y, "x": target_x},
        ),
    )
    xr.testing.assert_allclose(refreshed_minimum_reference_evi, minimum_reference_evi)
