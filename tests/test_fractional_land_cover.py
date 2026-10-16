"""Tests for ingesting pre-apportioned fractional land cover."""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from pyVPRM.VPRM import vprm_preprocessor
from pyVPRM.sat_managers.base_manager import satellite_data_manager


def make_preprocessor_with_grid():
    """Build a minimal preprocessor with an EPSG:2193 satellite grid.

    Returns
    -------
    vprm_preprocessor
        Preprocessor instance with a two-by-two satellite grid.
    """
    grid = xr.Dataset(coords={"y": [200.0, 100.0], "x": [100.0, 200.0]})
    grid = grid.rio.set_spatial_dims(x_dim="x", y_dim="y")
    grid = grid.rio.write_crs("EPSG:2193")
    grid = grid.rio.write_transform()
    preprocessor = object.__new__(vprm_preprocessor)
    preprocessor.sat_imgs = satellite_data_manager(sat_img=grid)
    preprocessor.land_cover_type = None
    return preprocessor


def make_source_fractions():
    """Build source-class fractions on the test satellite grid.

    Returns
    -------
    xarray.DataArray
        Three source-class fractions with ``land_cover_class``, ``y``, and
        ``x`` dimensions and EPSG:2193 spatial metadata.
    """
    fractions = xr.DataArray(
        np.array(
            [
                [[0.2, 0.0], [0.5, 0.0]],
                [[0.3, 1.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.5, 1.0]],
            ]
        ),
        dims=("land_cover_class", "y", "x"),
        coords={
            "land_cover_class": [10, 20, 30],
            "y": [200.0, 100.0],
            "x": [100.0, 200.0],
        },
    )
    fractions = fractions.rio.set_spatial_dims(x_dim="x", y_dim="y")
    fractions = fractions.rio.write_crs("EPSG:2193")
    return fractions.rio.write_transform()


def test_add_fractional_land_cover_map_aggregates_classes():
    """Aggregate multiple source classes into one VPRM class.

    Returns
    -------
    None
        The test verifies source-class aggregation and retained CRS metadata.
    """
    preprocessor = make_preprocessor_with_grid()
    preprocessor.add_fractional_land_cover_map(
        make_source_fractions(),
        {10: 1, 20: 1, 30: 8},
        source_name="synthetic",
    )

    result = preprocessor.land_cover_type.sat_img
    np.testing.assert_array_equal(result.vprm_classes.values, [1, 8])
    np.testing.assert_allclose(result.sel(vprm_classes=1), [[0.5, 1.0], [0.5, 0.0]])
    np.testing.assert_allclose(result.sel(vprm_classes=8), [[0.5, 0.0], [0.5, 1.0]])
    assert result.rio.crs.to_epsg() == 2193
    assert result.attrs["source_product"] == "synthetic"


def test_add_fractional_land_cover_map_rejects_unmapped_classes():
    """Require every source class to have an explicit VPRM mapping.

    Returns
    -------
    None
        The test verifies an incomplete class mapping raises ``ValueError``.
    """
    preprocessor = make_preprocessor_with_grid()

    try:
        preprocessor.add_fractional_land_cover_map(
            make_source_fractions(), {10: 1, 20: 1}
        )
    except ValueError as error:
        assert "missing source classes" in str(error)
    else:
        raise AssertionError("Expected an incomplete class mapping to fail.")
