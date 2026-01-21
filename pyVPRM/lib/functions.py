from pyproj import Proj
import math
import pandas as pd
import numpy as np
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import xarray as xr
from datetime import datetime
import rioxarray
import warnings

def get_corners_from_pixel_centers_1D(pixel_centers_1D):
    half_pixel_width = np.unique(np.diff(pixel_centers_1D))[0]/2       #get smallest difference between two pixel centers (why not just take first?)
    pixel_corners = pixel_centers_1D - half_pixel_width                #shift from center to top/left corners
    pixel_corners = list(pixel_corners)                                #list for fast appending
    pixel_corners.append(pixel_corners[-1]+2*half_pixel_width)         #add missing lower/right corners
    pixel_corners = np.array(pixel_corners)
    return pixel_corners  

def parse_wrf_grid_file(file_path, n_chunks=1, chunk_x=1, chunk_y=1):

    t = xr.open_dataset(file_path)

    lats = np.linspace(
        0, np.shape(t["XLAT_M"].values.squeeze())[0], n_chunks + 1, dtype=int
    )

    lons = np.linspace(
        0, np.shape(t["XLONG_M"].values.squeeze())[1], n_chunks + 1, dtype=int
    )

    out_grid = xr.Dataset(
        {
            "lon": (
                ["y", "x"],
                t["XLONG_M"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y], lons[chunk_x - 1] : lons[chunk_x]
                ],
                {"units": "degrees_east"},
            ),
            "lon_b": (
                ["y_b", "x_b"],
                t["XLONG_C"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y] + 1,
                    lons[chunk_x - 1] : lons[chunk_x] + 1,
                ],
                {"units": "degrees_east"},
            ),
            "lat": (
                ["y", "x"],
                t["XLAT_M"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y], lons[chunk_x - 1] : lons[chunk_x]
                ],
                {"units": "degrees_north"},
            ),
            "lat_b": (
                ["y_b", "x_b"],
                t["XLAT_C"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y] + 1,
                    lons[chunk_x - 1] : lons[chunk_x] + 1,
                ],
                {"units": "degrees_north"},
            ),
        }
    )

    out_grid = out_grid.set_coords(["lon", "lat", "lat_b", "lon_b"])
    out_grid.rio.write_crs("WGS84", inplace=True)
    return out_grid


def make_xesmf_grid(satellite_image, transformer = None):
    '''
    give satellite_image as dict only if coordinates are aready lat lon, then no transformation necessary
    '''
    
    #get x and y coordinates of pixel centers from the satellite image
    if isinstance(satellite_image, dict):
        x_pixel_centers = satellite_image['lons']
        y_pixel_centers = satellite_image['lats']
    else:
        x_pixel_centers = satellite_image.coords['x'].values
        y_pixel_centers = satellite_image.coords['y'].values

    #get the pixel corners from the pixel centers   
    x_pixel_corners = get_corners_from_pixel_centers_1D(x_pixel_centers)
    y_pixel_corners = get_corners_from_pixel_centers_1D(y_pixel_centers)

    #make meshgrids
    X_center_grid, Y_center_grid = np.meshgrid(x_pixel_centers, y_pixel_centers)
    X_corner_grid, Y_corner_grid = np.meshgrid(x_pixel_corners, y_pixel_corners)

    #if coordinates are not already lat lon dict, define transformer to transform from the images crs to the defined crs of WGS84
    if not isinstance(satellite_image, dict):    
        #define transformer using crs of satellite_image
        if transformer is None:
            transformer = Transformer.from_crs(satellite_image.rio.crs,
                            '+proj=longlat +datum=WGS84',
                            always_xy=True)
        #apply transformer to the grids
        X_center_grid, Y_center_grid = transformer.transform(X_center_grid, Y_center_grid)
        X_corner_grid, Y_corner_grid = transformer.transform(X_corner_grid, Y_corner_grid)

   #put the grids in one xarray
    pixel_grid = xr.Dataset({"lon": (["y", "x"], X_center_grid,
                          {"units": "degrees_east"}),
                         "lon_b": (["y_b", "x_b"], X_corner_grid,
                         {"units": "degrees_east"}),
                          "lat": (["y", "x"], Y_center_grid,
                          {"units": "degrees_north"}),
                         "lat_b": (["y_b", "x_b"], Y_corner_grid,
                         {"units": "degrees_north"})
                          })
    pixel_grid = pixel_grid.set_coords(['lon', 'lat', "lon_b", "lat_b"])
    return pixel_grid


def to_esmf_grid(sat_img):

    if isinstance(sat_img, dict):
        x = sat_img["lons"]
        y = sat_img["lats"]
    else:
        y = sat_img.coords["y"].values
        x = sat_img.coords["x"].values

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    ny = len(y)
    nx = len(x)
    # make 2D
    y_center = np.broadcast_to(y[:, None], (ny, nx))
    x_center = np.broadcast_to(x[None, :], (ny, nx))

    # compute corner points: must be counterclockwise
    y_corner = np.stack(
        (
            y_center - dy / 2.0,  # SW
            y_center - dy / 2.0,  # SE
            y_center + dy / 2.0,  # NE
            y_center + dy / 2.0,
        ),  # NW
        axis=2,
    )

    x_corner = np.stack(
        (
            x_center - dx / 2.0,  # SW
            x_center + dx / 2.0,  # SE
            x_center + dx / 2.0,  # NE
            x_center - dx / 2.0,
        ),  # NW
        axis=2,
    )

    if not isinstance(sat_img, dict):
        t = Transformer.from_crs(
            sat_img.rio.crs, "+proj=longlat +datum=WGS84", always_xy=True
        )
        x_center, y_center = t.transform(x_center, y_center)
        x_corner, y_corner = t.transform(x_corner, y_corner)
    grid_imask = np.ones((ny, nx), dtype=np.int32)

    # generate output dataset
    dso = xr.Dataset()
    dso["grid_dims"] = xr.DataArray(
        np.array([nx, ny], dtype=np.int32), dims=("grid_rank",)
    )
    dso.grid_dims.encoding = {"dtype": np.int32}

    dso["grid_center_lat"] = xr.DataArray(
        y_center.reshape((-1,)), dims=("grid_size"), attrs={"units": "degrees"}
    )

    dso["grid_center_lon"] = xr.DataArray(
        x_center.reshape((-1,)), dims=("grid_size"), attrs={"units": "degrees"}
    )

    dso["grid_corner_lat"] = xr.DataArray(
        y_corner.reshape((-1, 4)),
        dims=("grid_size", "grid_corners"),
        attrs={"units": "degrees"},
    )
    dso["grid_corner_lon"] = xr.DataArray(
        x_corner.reshape((-1, 4)),
        dims=("grid_size", "grid_corners"),
        attrs={"units": "degrees"},
    )
    dso["grid_imask"] = xr.DataArray(
        grid_imask.reshape((-1,)), dims=("grid_size"), attrs={"units": "unitless"}
    )
    dso.grid_imask.encoding = {"dtype": np.int32}

    # force no '_FillValue' if not specified
    for v in dso.variables:
        if "_FillValue" not in dso[v].encoding:
            dso[v].encoding["_FillValue"] = None

    dso.attrs = {
        "title": f"{ny} x {nx} (lat x lon) grid",
        "created_by": "latlon_to_scrip",
        "date_created": f"{datetime.now()}",
        "conventions": "SCRIP",
    }
    return dso


def do_lowess_smoothing(array_to_smooth, xvals=None, timestamps=None, frac=0.25, it=3):
    ### ToDo: Choose frac adaptively from the data.

    """
    Performs lowess smoothing on a 2-D-array, where the first dimension is the time.

        Parameters:
                array_to_smooth (list): The 2-D-array
        Returns:
                The lowess smoothed array
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret = []

        if array_to_smooth.ndim == 1:
            if timestamps is None:
                t_timestamp = np.arange(len(array_to_smooth))
            else:
                t_timestamp = timestamps
            mask = np.isfinite(array_to_smooth)
            if xvals is None:
                xvals = t_timestamp
            ret = [np.nan]
            counter = 0
            while counter < 10:
                ret = lowess(
                    array_to_smooth[mask],
                    t_timestamp[mask],
                    is_sorted=True,
                    frac=frac + 0.05 * counter,
                    it=it,
                    xvals=xvals,
                    return_sorted=False,
                )
                if not np.all(np.isfinite(ret)):
                    #    print('Non finite values for frac: {}. Retry.'.format(frac+0.05*counter))
                    counter += 1
                else:
                    break
            return ret
        else:
            if xvals is not None:
                ret_array = np.zeros((len(xvals), np.shape(array_to_smooth)[1]))
            else:
                ret_array = np.zeros(
                    (len(array_to_smooth[:, 0]), np.shape(array_to_smooth)[1])
                )
            for j in range(np.shape(array_to_smooth)[1]):
                if timestamps is None:
                    t_timestamp = np.arange(len(array_to_smooth[:, j]))
                else:
                    if timestamps.ndim == 1:
                        t_timestamp = timestamps
                    else:
                        t_timestamp = timestamps[:, j]
                mask = np.isfinite(array_to_smooth[:, j])
                if xvals is None:
                    xvals = t_timestamp
                lws_res = [np.nan]
                counter = 0
                while counter < 10:
                    lws_res = lowess(
                        array_to_smooth[:, j][mask],
                        t_timestamp[mask],
                        is_sorted=True,
                        frac=frac + 0.05 * counter,
                        it=it,
                        xvals=xvals,
                        return_sorted=False,
                    )
                    if not np.all(np.isfinite(lws_res)):
                        #  print('Non finite values for frac: {}. Retry.'.format(frac+0.05*counter))
                        counter += 1
                    else:
                        break
                ret_array[:, j] = lws_res
            return ret_array.T


def lat_lon_to_modis(lat, lon):
    CELLS = 2400
    VERTICAL_TILES = 18
    HORIZONTAL_TILES = 36
    EARTH_RADIUS = 6371007.181
    EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

    TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
    TILE_HEIGHT = TILE_WIDTH
    CELL_SIZE = TILE_WIDTH / CELLS
    MODIS_GRID = Proj(f"+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext")
    x, y = MODIS_GRID(lon, lat)
    h = (EARTH_WIDTH * 0.5 + x) / TILE_WIDTH
    v = -(EARTH_WIDTH * 0.25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
    return int(h), int(v)


def add_corners_to_1d_grid(mids):
    diff = np.unique(np.diff(mids))[0] / 2
    mids = mids - diff
    mids = list(mids)
    mids.append(mids[-1] + 2 * diff)
    mids = np.array(mids)
    return mids

def get_specific_chunk(data, dim_chunks, chunk_position):
    """
    Get a specific chunk from an xarray DataArray based on its position in the grid.

    Parameters:
        data (xr.DataArray): The input DataArray to split.
        dim_chunks (dict): A dictionary specifying the number of chunks for each dimension,
                           e.g., {'x': 3, 'y': 2}.
        chunk_position (tuple): The position of the chunk in the grid, e.g., (0, 1).

    Returns:
        xr.DataArray: The specific chunk corresponding to the given position.
    """
    # Validate chunk_position
    if len(chunk_position) != len(dim_chunks):
        raise ValueError("chunk_position must have the same length as dim_chunks")

    dim_slices = {}

    for dim, n_chunks in dim_chunks.items():
        # Get dimension size
        dim_size = data.sizes[dim]
        # Compute chunk sizes, handling remainders
        chunk_sizes = [(dim_size + i) // n_chunks for i in range(n_chunks)]
        # Compute start and stop indices for each chunk
        indices = [sum(chunk_sizes[:i]) for i in range(n_chunks + 1)]
        dim_slices[dim] = [(indices[i], indices[i + 1]) for i in range(n_chunks)]

    # Build slice for the specified chunk position
    slices = {
        dim: slice(*dim_slices[dim][chunk_position[idx]])
        for idx, dim in enumerate(dim_chunks.keys())
    }

    # Return the specific chunk
    return data.isel(**slices)


def get_fully_covered_destinaion_grid_cell(dest_grid, regridder):
    # Currently only works for destination grid in WGS84
    
    dest_lon = dest_grid['lon'].values
    dest_lat = dest_grid['lat'].values
    weights = regridder.weights.data  # Extract sparse matrix from DataArray
    
    # Sum weights for each destination cell
    dest_weights_sum = np.array(weights.sum(axis=1).todense()).flatten()
    
    # Check if destination cells are fully covered (sum of weights == 1)
    is_fully_covered = dest_weights_sum > 0.99
    
    # Create a mask indicating fully covered cells
    coverage_mask = is_fully_covered.reshape((len(dest_lat), len(dest_lon)))
    dest_grid["is_fully_covered"] = (["lat", "lon"], coverage_mask)
    return dest_grid


def get_fractional_coverage_of_destinaion_grid_cell(dest_grid, regridder):
    # Currently only works for destination grid in WGS84
    
    dest_lon = dest_grid['lon'].values
    dest_lat = dest_grid['lat'].values
    weights = regridder.weights.data  # Extract sparse matrix from DataArray
    
    # Sum weights for each destination cell
    dest_weights_sum = np.array(weights.sum(axis=1).todense()).flatten()
    
    # Create a mask indicating fully covered cells
    coverage_mask = dest_weights_sum.reshape((len(dest_lat), len(dest_lon)))
    dest_grid["is_fully_covered"] = (["lat", "lon"], coverage_mask)
    return dest_grid


def merge_chunks_with_open_mfdataset(chunk_files, dim_order=['x', 'y']):
    """
    Merge pre-saved chunk files into a single xarray DataArray using open_mfdataset
    for efficient processing row-by-row or column-by-column.

    Parameters:
        chunk_files (dict): A dictionary where keys are chunk positions (e.g., (0, 0))
                            and values are file paths to the chunk files.
        dim_order (list): List of dimension names in the order they should be concatenated,
                          e.g., ['x', 'y'].

    Returns:
        xr.DataArray: The remerged xarray DataArray.
    """
    # Determine the number of dimensions
    ndim = len(dim_order)

    # Group files by their positions along the last dimension
    grouped_files = {}
    for position, file_path in chunk_files.items():
        group_key = position[:-1]  # All dimensions except the last
        grouped_files.setdefault(group_key, []).append((position[-1], file_path))

    # Sort files within each group by the last dimension
    for key in grouped_files:
        grouped_files[key] = [fp for _, fp in sorted(grouped_files[key])]

    # Merge files within each group (rows or columns)
    partial_datasets = {}
    for group_key, file_list in grouped_files.items():
        partial_datasets[group_key] = xr.open_mfdataset(
            file_list, combine="nested", concat_dim=dim_order[-1], engine="netcdf4"
        )

    # Organize partial datasets into a grid for further concatenation
    grid_shape = tuple(max(pos[i] for pos in chunk_files.keys()) + 1 for i in range(ndim - 1))
    dataset_grid = np.empty(grid_shape, dtype=object)

    for group_key, dataset in partial_datasets.items():
        dataset_grid[group_key] = dataset

    # Merge across the remaining dimensions recursively
    for dim_idx in reversed(range(ndim - 1)):
        dataset_grid = [
            xr.concat(row, dim=dim_order[dim_idx]) if isinstance(row, (list, np.ndarray)) else row
            for row in dataset_grid
        ]
        dataset_grid = xr.concat(dataset_grid, dim=dim_order[dim_idx])

    return dataset_grid


def replace_inf_runs_ignore_nans(
    da: xr.DataArray,
    time_dim: str = "time",
    N: int = 4,
    percentile: float = 5.0
) -> xr.DataArray:
    """
    Replace runs of >= N consecutive inf values (ignoring nans) along time
    with the given percentile (default 5%) of the finite values over time.
    Otherwise replace infs with nan.
    
    Parameters
    ----------
    da : xr.DataArray
        Input array with a time dimension.
    time_dim : str
        Name of the time dimension.
    N : int
        Minimum run length of inf values (ignoring nans) that triggers replacement.
    percentile : float
        Percentile (0–100) of finite values to use for replacement.
    """
    time_axis = da.get_axis_num(time_dim)
    arr = np.moveaxis(da.values, time_axis, 0)   # shape (T, ...)
    T = arr.shape[0]

    if arr.ndim == 1:
        flat = arr.reshape(T, 1)
        out_flat = flat.copy()
    else:
        spatial_size = int(np.prod(arr.shape[1:]))
        flat = arr.reshape(T, spatial_size)      # shape (T, P)
        out_flat = flat.copy()

    for p in range(flat.shape[1]):
        seq = flat[:, p]
        is_inf = np.isinf(seq)
        is_finite = np.isfinite(seq)
        not_break = ~is_finite         # True for inf or nan

        # reference distribution of finite values
        finite_vals = seq[is_finite]
        if finite_vals.size > 0:
            repl_val = np.nanpercentile(finite_vals, percentile)
        else:
            repl_val = np.nan

        t = 0
        while t < T:
            if not_break[t]:
                start = t
                t2 = t + 1
                while t2 < T and not_break[t2]:
                    t2 += 1
                run_slice = slice(start, t2)
                run_is_inf = is_inf[run_slice]
                count_inf = int(run_is_inf.sum())

                if count_inf >= N and finite_vals.size > 0:
                    abs_idxs = start + np.where(run_is_inf)[0]
                    out_flat[abs_idxs, p] = repl_val
                else:
                    if count_inf > 0:
                        abs_idxs = start + np.where(run_is_inf)[0]
                        out_flat[abs_idxs, p] = np.nan

                t = t2
            else:
                t += 1

    out = out_flat.reshape(arr.shape)
    out = np.moveaxis(out, 0, time_axis)

    return xr.DataArray(out, coords=da.coords, dims=da.dims, name=da.name, attrs=da.attrs)


