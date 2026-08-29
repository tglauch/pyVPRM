"""Tests for GISA-new ingestion and impervious-surface preprocessing."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
import rioxarray  # noqa: F401
import xarray as xr

from pyVPRM.VPRM import vprm_preprocessor
from pyVPRM.sat_managers.base_manager import satellite_data_manager
from pyVPRM.sat_managers.gisa_new import gisa_new


def write_gisa_raster(path):
    """Write a small GISA-new-like GeoTIFF for manager tests.

    Parameters
    ----------
    path : pathlib.Path
        Output GeoTIFF path.

    Returns
    -------
    None
        The raster contains no-impervious, 1985, and 2021 detection indices.
    """

    values = np.array([[0, 1], [37, 0]], dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(174.0, -36.0, 0.01, 0.01),
    ) as output:
        output.write(values, 1)


def test_gisa_new_derives_binary_percentages_from_detection_year(tmp_path):
    """Convert GISA detection indices into target-year ISA percentages.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None
        The test verifies first-detection years and 0/100 ISA percentages.
    """

    raster_path = tmp_path / "gisa.tif"
    write_gisa_raster(raster_path)
    manager = gisa_new(raster_path, target_year=2021)

    dataset = manager._open_image(raster_path)

    np.testing.assert_allclose(
        dataset["impervious_surface_percentage"].values,
        [[0.0, 100.0], [100.0, 0.0]],
    )
    np.testing.assert_allclose(
        dataset["impervious_surface_first_year"].values,
        [[np.nan, 1985.0], [2021.0, np.nan]],
        equal_nan=True,
    )


def test_add_impervious_surface_masks_zero_source_cells(monkeypatch, tmp_path):
    """Exclude zero ISA cells when invoking conservative regridding.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace conservative regridding with capture logic.
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None
        The test verifies that the preprocessor supplies an ISA-positive mask
        and retains the regridded manager and variable metadata.
    """

    source = xr.Dataset(
        {
            "impervious_surface_percentage": (
                ("y", "x"), np.array([[0.0, 100.0], [25.0, 0.0]])
            )
        },
        coords={"x": [174.0, 174.1], "y": [-36.1, -36.0]},
    ).rio.write_crs("EPSG:4326")
    destination = xr.Dataset(
        coords={"x": [174.05], "y": [-36.05]}
    ).rio.write_crs("EPSG:4326")
    preprocessor = object.__new__(vprm_preprocessor)
    preprocessor.n_cpus = 3
    preprocessor.sat_imgs = satellite_data_manager(sat_img=destination)
    captured = {}

    def capture_regrid(source_dataset, destination_dataset, **kwargs):
        """Record regridding arguments and return a one-cell result.

        Parameters
        ----------
        source_dataset : xarray.Dataset
            Source ISA dataset passed by the preprocessor.
        destination_dataset : xarray.Dataset
            Destination satellite grid passed by the preprocessor.
        **kwargs
            Regridding options supplied by the preprocessor.

        Returns
        -------
        xarray.Dataset
            One-cell percentage dataset on the destination grid.
        """

        captured["source"] = source_dataset
        captured["destination"] = destination_dataset
        captured.update(kwargs)
        return xr.Dataset(
            {"impervious_surface_percentage": (("y", "x"), [[42.0]])},
            coords={"x": destination_dataset.x, "y": destination_dataset.y},
        ).rio.write_crs(destination_dataset.rio.crs)

    monkeypatch.setattr("pyVPRM.VPRM.conservative_regrid", capture_regrid)
    preprocessor.add_impervious_surface_area(
        satellite_data_manager(sat_img=source),
        regridder_save_path=tmp_path / "weights.nc",
        max_source_cells=100,
        mpi=False,
    )

    np.testing.assert_array_equal(
        captured["source_mask"].values,
        [[False, True], [True, False]],
    )
    assert captured["n_cpus"] == 3
    assert captured["max_source_cells"] == 100
    assert captured["mpi"] is False
    assert (
        preprocessor.impervious_surface_area.sat_img[
            "impervious_surface_percentage"
        ].attrs["units"]
        == "percent"
    )
