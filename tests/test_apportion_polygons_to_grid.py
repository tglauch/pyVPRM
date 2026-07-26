"""Tests for polygon-to-grid fractional coverage apportionment."""

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

from pyVPRM.apportion_polygons_to_grid import apportion


def test_apportion_preserves_fractional_class_coverage():
    """Apportion full and partial polygons onto a two-cell target grid.

    Returns
    -------
    None
        The test asserts expected class fractions and spatial metadata.
    """
    target_grid = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={"y": [500.0, 1500.0], "x": [500.0, 1500.0]},
    )
    target_grid = target_grid.rio.set_spatial_dims(x_dim="x", y_dim="y")
    target_grid = target_grid.rio.write_crs("EPSG:2193")
    target_grid = target_grid.rio.write_transform()
    polygons = gpd.GeoDataFrame(
        {"lcdb_class": [1, 2]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 1500, 1000)],
        crs="EPSG:2193",
    )

    fractions = apportion(
        polygons,
        target_grid,
        "EPSG:2193",
        class_column="lcdb_class",
        chunk_rows=1,
    )

    np.testing.assert_allclose(
        fractions.sel(land_cover_class=1), [[1.0, 0.0], [0.0, 0.0]]
    )
    np.testing.assert_allclose(
        fractions.sel(land_cover_class=2), [[0.0, 0.5], [0.0, 0.0]]
    )
    assert fractions.rio.crs.to_epsg() == 2193
