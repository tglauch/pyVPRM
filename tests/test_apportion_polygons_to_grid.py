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


def test_apportion_queries_polygons_in_their_own_crs():
    """Find polygon candidates when source and target grids use different CRSs.

    Returns
    -------
    None
        The test asserts that a polygon matching one target cell contributes
        full coverage after source-CRS spatial-index selection.
    """
    target_grid = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={"y": [-43.005, -43.015], "x": [172.005, 172.015]},
    )
    target_grid = target_grid.rio.set_spatial_dims(x_dim="x", y_dim="y")
    target_grid = target_grid.rio.write_crs("EPSG:4326")
    target_grid = target_grid.rio.write_transform()
    source_polygon = gpd.GeoDataFrame(
        {"lcdb_class": [1]},
        geometry=[box(172.0, -43.01, 172.01, -43.0)],
        crs="EPSG:4326",
    ).to_crs("EPSG:2193")

    fractions = apportion(
        source_polygon,
        target_grid,
        "EPSG:2193",
        class_column="lcdb_class",
        chunk_rows=1,
    )

    np.testing.assert_allclose(
        fractions.sel(land_cover_class=1).isel(y=0, x=0), 1.0, rtol=1e-6
    )
    np.testing.assert_allclose(fractions.sum(), 1.0, rtol=1e-6)
