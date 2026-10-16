"""Tests for partitioned conservative regridding utilities."""

import sys
from types import SimpleNamespace

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from pyVPRM.lib import regridding


def make_grid(values, *, x, y, crs="EPSG:4326"):
    """Build a georeferenced one-variable dataset for regridding tests.

    Parameters
    ----------
    values : numpy.ndarray
        Two-dimensional source data with ``y, x`` dimensions.
    x : sequence[float]
        Grid x coordinates.
    y : sequence[float]
        Grid y coordinates.
    crs : str, default="EPSG:4326"
        Coordinate reference system assigned through rioxarray.

    Returns
    -------
    xarray.Dataset
        Dataset containing a variable named ``value``.
    """

    return xr.Dataset(
        {"value": (("y", "x"), values)}, coords={"x": x, "y": y}
    ).rio.write_crs(crs)


def test_source_y_slices_partition_without_overlap():
    """Split an oversized source into contiguous non-overlapping strips.

    Returns
    -------
    None
        The test verifies a cell-limited source is partitioned by rows.
    """

    source = make_grid(np.ones((5, 4)), x=[0, 1, 2, 3], y=[0, 1, 2, 3, 4])

    slices = regridding._source_y_slices(source, max_source_cells=8)

    assert slices == [slice(0, 2), slice(2, 4), slice(4, 5)]


def test_partitioned_regridding_sums_contributions_and_keeps_crs(monkeypatch, tmp_path):
    """Sum partition contributions and retain destination spatial metadata.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace ESMF and xESMF execution.
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None
        The test verifies two partitions are generated, their contributions
        are added, and the result retains the destination CRS and transform.
    """

    source = make_grid(
        np.ones((4, 2)), x=[174.0, 174.1], y=[-36.3, -36.2, -36.1, -36.0]
    )
    destination = make_grid(
        np.zeros((2, 2)), x=[174.0, 174.1], y=[-36.1, -36.0]
    )
    generated_partitions = []
    regridder_calls = []

    def capture_weight_generation(source_chunk, *args, **kwargs):
        """Record each partition without invoking ESMF.

        Parameters
        ----------
        source_chunk : xarray.Dataset
            Source partition that would be passed to ESMF.
        *args
            Positional arguments accepted by the helper.
        **kwargs
            Keyword arguments accepted by the helper.

        Returns
        -------
        None
            The source partition is recorded for later assertions.
        """

        del args, kwargs
        generated_partitions.append(source_chunk)

    class FakeRegridder:
        """Return one deterministic destination contribution per partition."""

        def __init__(self, *args, **kwargs):
            """Record construction order for the fake xESMF regridder.

            Parameters
            ----------
            *args
                Positional xESMF constructor arguments.
            **kwargs
                Keyword xESMF constructor arguments.
            """

            del args, kwargs
            regridder_calls.append(len(regridder_calls) + 1)

        def __call__(self, source_chunk):
            """Return a destination-sized contribution for one partition.

            Parameters
            ----------
            source_chunk : xarray.Dataset
                Source partition passed by the regridding helper.

            Returns
            -------
            xarray.Dataset
                Constant contribution on the test destination grid.
            """

            del source_chunk
            return xr.Dataset(
                {"value": (("y", "x"), np.full((2, 2), regridder_calls[-1]))},
                coords={"x": destination.x, "y": destination.y},
            )

    monkeypatch.setattr(
        regridding, "_generate_conservative_weights", capture_weight_generation
    )
    monkeypatch.setattr(regridding, "make_xesmf_grid", lambda dataset: dataset)
    monkeypatch.setitem(sys.modules, "xesmf", SimpleNamespace(Regridder=FakeRegridder))

    result = regridding.conservative_regrid(
        source,
        destination,
        weight_path=tmp_path / "weights.nc",
        max_source_cells=4,
        mpi=False,
    )

    assert [partition.sizes["y"] for partition in generated_partitions] == [2, 2]
    assert regridder_calls == [1, 2]
    np.testing.assert_array_equal(result["value"].values, np.full((2, 2), 3))
    assert result.rio.crs == destination.rio.crs
    assert result.rio.transform(recalc=False) == destination.rio.transform(recalc=False)
