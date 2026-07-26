"""Apportion polygon classes onto a regular raster grid.

Notes
-----
Areas are calculated after both input geometries have been transformed to an
equal-area CRS supplied by the caller.  The returned fractions retain the
coordinates and spatial metadata of the target grid; the area CRS is used
only for the intersection calculation.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box


def _coordinate_edges(coordinates, name):
    """Return pixel edges inferred from regularly spaced center coordinates.

    Parameters
    ----------
    coordinates : numpy.ndarray
        One-dimensional pixel-center coordinates.
    name : str
        Coordinate name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional array with one more element than ``coordinates``.

    Raises
    ------
    ValueError
        If fewer than two coordinates are supplied or their spacing is not
        regular.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(f"{name!r} must contain at least two coordinates.")

    steps = np.diff(coordinates)
    if not np.allclose(steps, steps[0]):
        raise ValueError(f"{name!r} coordinates must be regularly spaced.")

    half_step = steps[0] / 2
    return np.concatenate(
        (
            [coordinates[0] - half_step],
            coordinates[:-1] + np.diff(coordinates) / 2,
            [coordinates[-1] + half_step],
        )
    )


def _target_cell_chunk(x_edges, y_edges, row_start, row_stop, crs):
    """Build one row-wise chunk of target-grid cell polygons.

    Parameters
    ----------
    x_edges, y_edges : numpy.ndarray
        Pixel-edge coordinates for the target grid.
    row_start, row_stop : int
        Half-open range of target-grid rows included in the chunk.
    crs : pyproj.CRS or str
        CRS of the target grid.

    Returns
    -------
    geopandas.GeoDataFrame
        Cell polygons with ``_row`` and ``_column`` index columns.
    """
    row_indices = np.repeat(np.arange(row_start, row_stop), len(x_edges) - 1)
    column_indices = np.tile(np.arange(len(x_edges) - 1), row_stop - row_start)
    lower_x = np.tile(x_edges[:-1], row_stop - row_start)
    upper_x = np.tile(x_edges[1:], row_stop - row_start)
    lower_y = np.repeat(
        np.minimum(y_edges[row_start:row_stop], y_edges[row_start + 1 : row_stop + 1]),
        len(x_edges) - 1,
    )
    upper_y = np.repeat(
        np.maximum(y_edges[row_start:row_stop], y_edges[row_start + 1 : row_stop + 1]),
        len(x_edges) - 1,
    )

    return gpd.GeoDataFrame(
        {"_row": row_indices, "_column": column_indices},
        geometry=[
            box(min_x, min_y, max_x, max_y)
            for min_x, min_y, max_x, max_y in zip(lower_x, lower_y, upper_x, upper_y)
        ],
        crs=crs,
    )


def apportion(
    polygons,
    target_grid,
    area_crs,
    class_column="vprm_class",
    *,
    class_dim="land_cover_class",
    chunk_rows=128,
    dtype=np.float32,
):
    """Calculate per-class polygon coverage fractions for a regular grid.

    Parameters
    ----------
    polygons : geopandas.GeoDataFrame
        Polygon features with a defined CRS and a categorical ``class_column``.
        Features with a missing class value are ignored.
    target_grid : xarray.Dataset or xarray.DataArray
        Regular grid with one-dimensional ``x`` and ``y`` coordinates and rio
        CRS metadata.  Fractions are returned on these original coordinates.
    area_crs : str, int, or pyproj.CRS
        Equal-area CRS appropriate to the region.  This must be supplied by
        the caller; it is used only to measure cell and intersection areas.
    class_column : str, default="vprm_class"
        Column in ``polygons`` containing the categorical class values.
    class_dim : str, default="land_cover_class"
        Name of the class dimension in the returned array.
    chunk_rows : int, default=128
        Number of target-grid rows to intersect at a time.  Smaller chunks
        reduce peak memory use at the cost of more spatial-index queries.
    dtype : numpy.dtype, default=numpy.float32
        Data type used for the returned coverage fractions.

    Returns
    -------
    xarray.DataArray
        Fractional class coverage with dimensions ``(class_dim, y, x)``.  Its
        rio CRS and affine transform match ``target_grid``.

    Raises
    ------
    TypeError
        If ``polygons`` is not a GeoDataFrame.
    ValueError
        If required spatial metadata, coordinates, or class values are
        absent, or if ``chunk_rows`` is not positive.

    Notes
    -----
    Fractions are intersection area divided by the complete target-cell area,
    not by the mapped polygon area.  Therefore cells partly outside the input
    polygon coverage retain a fractional sum below one.
    """
    if not isinstance(polygons, gpd.GeoDataFrame):
        raise TypeError("polygons must be a geopandas.GeoDataFrame.")
    if polygons.crs is None:
        raise ValueError("polygons must have a defined CRS.")
    if class_column not in polygons.columns:
        raise ValueError(f"polygons do not contain class column {class_column!r}.")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer.")
    if "x" not in target_grid.coords or "y" not in target_grid.coords:
        raise ValueError(
            "target_grid must have one-dimensional 'x' and 'y' coordinates."
        )
    if target_grid.coords["x"].ndim != 1 or target_grid.coords["y"].ndim != 1:
        raise ValueError("target_grid 'x' and 'y' coordinates must be one-dimensional.")
    if target_grid.rio.crs is None:
        raise ValueError("target_grid must have rio CRS metadata.")

    source = polygons.loc[
        polygons[class_column].notna(), [class_column, "geometry"]
    ].copy()
    if source.empty:
        raise ValueError("polygons contain no non-null class values.")

    class_values = list(pd.unique(source[class_column]))
    class_positions = {value: index for index, value in enumerate(class_values)}
    x_values = target_grid.coords["x"].values
    y_values = target_grid.coords["y"].values
    x_edges = _coordinate_edges(x_values, "x")
    y_edges = _coordinate_edges(y_values, "y")
    fractions = np.zeros((len(class_values), len(y_values), len(x_values)), dtype=dtype)
    source_index = source.sindex

    for row_start in range(0, len(y_values), chunk_rows):
        row_stop = min(row_start + chunk_rows, len(y_values))
        cells = _target_cell_chunk(
            x_edges, y_edges, row_start, row_stop, target_grid.rio.crs
        )
        candidate_indices = source_index.query(
            box(*cells.total_bounds), predicate="intersects"
        )
        if len(candidate_indices) == 0:
            continue

        cells = cells.to_crs(area_crs)
        cells["_cell_area"] = cells.geometry.area
        candidates = source.iloc[candidate_indices].to_crs(area_crs)
        intersections = gpd.overlay(
            cells, candidates, how="intersection", keep_geom_type=False
        )
        if intersections.empty:
            continue

        intersections["_intersection_area"] = intersections.geometry.area
        intersections = intersections.loc[
            intersections["_intersection_area"] > 0
        ].copy()
        if intersections.empty:
            continue
        grouped = (
            intersections.groupby(["_row", "_column", class_column], sort=False)[
                "_intersection_area"
            ]
            .sum()
            .reset_index()
            .merge(
                cells[["_row", "_column", "_cell_area"]],
                on=["_row", "_column"],
                how="left",
                validate="many_to_one",
            )
        )
        for class_value, class_group in grouped.groupby(class_column, sort=False):
            fractions[
                class_positions[class_value],
                class_group["_row"].to_numpy(),
                class_group["_column"].to_numpy(),
            ] = (
                class_group["_intersection_area"].to_numpy()
                / class_group["_cell_area"].to_numpy()
            )

    result = xr.DataArray(
        fractions,
        dims=(class_dim, "y", "x"),
        coords={class_dim: class_values, "y": y_values, "x": x_values},
        name="fraction",
        attrs={
            "long_name": "fractional polygon-class coverage",
            "units": "1",
            "class_column": class_column,
            "area_crs": str(area_crs),
        },
    )
    result = result.rio.set_spatial_dims(x_dim="x", y_dim="y")
    result = result.rio.write_crs(target_grid.rio.crs)
    return result.rio.write_transform(target_grid.rio.transform(recalc=False))
