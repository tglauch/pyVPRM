"""Conservative ESMF regridding utilities for regular xarray grids.

The helpers in this module partition oversized source grids before generating
conservative ESMF weights. Partition contributions are additive when ESMF
uses destination-area normalisation, allowing high-resolution sources to be
regridded without creating one prohibitively large ESMF mesh.
"""

import os
import subprocess
import uuid

import xarray as xr
from loguru import logger

from pyVPRM.lib.functions import make_xesmf_grid, to_esmf_grid


def _source_y_slices(source, max_source_cells):
    """Return disjoint y-axis slices that respect a source-cell limit.

    Parameters
    ----------
    source : xarray.Dataset
        Source dataset with one-dimensional ``x`` and ``y`` coordinates.
    max_source_cells : int or None
        Maximum number of source cells in each returned slice. ``None``
        returns one slice containing the complete source grid.

    Returns
    -------
    list of slice
        Contiguous, non-overlapping slices along the source ``y`` dimension.

    Raises
    ------
    ValueError
        If the source does not have non-empty one-dimensional spatial
        coordinates, or if ``max_source_cells`` cannot contain one row.
    """
    if "x" not in source.coords or "y" not in source.coords:
        raise ValueError("source must provide one-dimensional x and y coordinates.")
    if source.coords["x"].ndim != 1 or source.coords["y"].ndim != 1:
        raise ValueError("source x and y coordinates must be one-dimensional.")

    nx = source.sizes["x"]
    ny = source.sizes["y"]
    if nx == 0 or ny == 0:
        raise ValueError("source x and y dimensions must be non-empty.")
    if max_source_cells is None:
        return [slice(0, ny)]
    if not isinstance(max_source_cells, int) or max_source_cells <= 0:
        raise ValueError("max_source_cells must be a positive integer or None.")
    if nx * ny <= max_source_cells:
        return [slice(0, ny)]

    rows_per_slice = max_source_cells // nx
    if rows_per_slice == 0:
        raise ValueError(
            "max_source_cells must be at least as large as the source x dimension."
        )
    return [
        slice(row_start, min(row_start + rows_per_slice, ny))
        for row_start in range(0, ny, rows_per_slice)
    ]


def _weight_path(weight_path, chunk_number, chunk_count):
    """Return the cache path for one source-grid partition.

    Parameters
    ----------
    weight_path : str or os.PathLike
        Requested weight-cache path.
    chunk_number : int
        Zero-based partition number.
    chunk_count : int
        Total number of source partitions.

    Returns
    -------
    str
        Weight-cache path. The original path is retained for an unsplit grid.
    """
    weight_path = os.fspath(weight_path)
    if chunk_count == 1:
        return weight_path
    stem, extension = os.path.splitext(weight_path)
    return "{}_src_{:03d}{}".format(stem, chunk_number, extension or ".nc")


def _generate_conservative_weights(
    source,
    destination,
    *,
    weight_path,
    n_cpus,
    mpi,
    logs,
):
    """Generate one cached ESMF conservative weight file when needed.

    Parameters
    ----------
    source : xarray.Dataset
        Source dataset for one grid partition.
    destination : xarray.Dataset
        Complete destination dataset.
    weight_path : str or os.PathLike
        Weight-cache file to create.
    n_cpus : int
        MPI process count when ``mpi`` is true.
    mpi : bool
        Whether to launch ESMF through ``mpirun``.
    logs : bool
        Whether ESMF PET log files should be retained.

    Returns
    -------
    None
        The weight file is created at ``weight_path`` when it is absent.

    Raises
    ------
    RuntimeError
        If ESMF weight generation fails. Temporary SCRIP grids are retained
        on failure to support diagnosis.
    """
    weight_path = os.fspath(weight_path)
    if os.path.exists(weight_path):
        return

    weight_directory = os.path.dirname(weight_path)
    if weight_directory:
        os.makedirs(weight_directory, exist_ok=True)
    source_temp_path = os.path.join(
        weight_directory, "{}.nc".format(uuid.uuid4())
    )
    destination_temp_path = os.path.join(
        weight_directory, "{}.nc".format(uuid.uuid4())
    )
    to_esmf_grid(source).to_netcdf(source_temp_path)
    to_esmf_grid(destination).to_netcdf(destination_temp_path)

    command = [
        "ESMF_RegridWeightGen",
        "--source",
        source_temp_path,
        "--destination",
        destination_temp_path,
        "--weight",
        weight_path,
        "-m",
        "conserve",
        "-r",
        "--netcdf4",
        "--ignore_unmapped",
    ]
    if not logs:
        command.append("--no_log")
    if mpi:
        command = ["mpirun", "-np", str(n_cpus)] + command

    logger.info("Run: {}".format(" ".join(command)))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ESMF failed to generate conservative regridding weights.") from exc

    for temporary_path in (source_temp_path, destination_temp_path):
        if os.path.exists(temporary_path):
            logger.info("regrid successful; removing {}".format(temporary_path))
            os.remove(temporary_path)


def conservative_regrid(
    source,
    destination,
    *,
    weight_path,
    max_source_cells=None,
    n_cpus=1,
    mpi=True,
    logs=False,
):
    """Conservatively regrid a regular source dataset onto a destination grid.

    Source partitions are regridded independently and their contributions are
    summed. This is valid because ESMF conservative weights generated here
    use destination-area normalisation and the source partitions contain
    disjoint cells.

    Parameters
    ----------
    source : xarray.Dataset
        Dataset containing variables on a regular grid with one-dimensional
        ``x`` and ``y`` coordinates and an assigned rio CRS.
    destination : xarray.Dataset
        Dataset defining the destination grid with one-dimensional ``x`` and
        ``y`` coordinates and an assigned rio CRS.
    weight_path : str or os.PathLike
        Cache path for an unsplit weight file. Partitioned grids receive
        ``_src_###`` before the file extension.
    max_source_cells : int, optional
        Maximum source cells in one ESMF invocation. Larger source grids are
        split into disjoint contiguous y-axis partitions.
    n_cpus : int, default=1
        MPI process count when generating new weights.
    mpi : bool, default=True
        Launch ESMF through ``mpirun``.
    logs : bool, default=False
        Retain ESMF PET logs instead of passing ``--no_log``.

    Returns
    -------
    xarray.Dataset
        Conservatively regridded source variables on the destination grid.

    Raises
    ------
    ValueError
        If the inputs do not use compatible regular x/y grid coordinates.
    RuntimeError
        If ESMF cannot create a required weight file.
    """
    if not isinstance(source, xr.Dataset) or not isinstance(destination, xr.Dataset):
        raise ValueError("source and destination must both be xarray.Dataset instances.")
    source_slices = _source_y_slices(source, max_source_cells)
    _source_y_slices(destination, None)

    import xesmf as xe

    result = None
    for chunk_number, source_slice in enumerate(source_slices):
        source_chunk = source.isel(y=source_slice)
        chunk_weight_path = _weight_path(
            weight_path, chunk_number, len(source_slices)
        )
        _generate_conservative_weights(
            source_chunk,
            destination,
            weight_path=chunk_weight_path,
            n_cpus=n_cpus,
            mpi=mpi,
            logs=logs,
        )
        regridder = xe.Regridder(
            make_xesmf_grid(source_chunk),
            make_xesmf_grid(destination),
            "conservative",
            weights=chunk_weight_path,
            reuse_weights=True,
        )
        contribution = regridder(source_chunk).fillna(0)
        result = contribution if result is None else result + contribution

    result = result.assign_coords(
        {
            "x": destination.coords["x"].values,
            "y": destination.coords["y"].values,
        }
    )
    for variable_name in source.data_vars:
        result[variable_name].attrs.update(source[variable_name].attrs)
    return result
