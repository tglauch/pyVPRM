"""Regression tests for dual-regional ESMF command construction."""

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from pyVPRM.VPRM import vprm_preprocessor
from pyVPRM.meteorologies.era5_monthly_xr import met_data_handler
from pyVPRM.sat_managers.base_manager import satellite_data_manager


class CapturedESMFCommand(RuntimeError):
    """Stop a test immediately after capturing an ESMF command string."""


def make_satellite_grid():
    """Build a minimal projected two-by-two satellite grid.

    Returns
    -------
    xarray.Dataset
        EPSG:2193 grid with one-dimensional ``x`` and ``y`` coordinates.
    """

    grid = xr.Dataset(coords={"y": [200.0, 100.0], "x": [100.0, 200.0]})
    grid = grid.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return grid.rio.write_crs("EPSG:2193").rio.write_transform()


def make_preprocessor():
    """Build a minimally initialized preprocessor for command tests.

    Returns
    -------
    pyVPRM.VPRM.vprm_preprocessor
        Preprocessor with a projected satellite grid and one CPU.
    """

    preprocessor = object.__new__(vprm_preprocessor)
    preprocessor.sat_imgs = satellite_data_manager(sat_img=make_satellite_grid())
    preprocessor.n_cpus = 1
    return preprocessor


def make_era5_handler():
    """Build a minimally initialized ERA5 handler for command tests.

    Returns
    -------
    pyVPRM.meteorologies.era5_monthly_xr.met_data_handler
        Handler with a one-cell source dataset and no existing regridder.
    """

    handler = object.__new__(met_data_handler)
    handler.in_era5_grid = True
    handler.regridder = None
    handler.mpi = False
    handler.ds_in_t = xr.Dataset(
        {"temperature": (("lat", "lon"), np.ones((1, 1)))},
        coords={"lat": [48.0], "lon": [11.0]},
    )
    handler.data = handler.ds_in_t.copy()
    return handler


def capture_esmf_command(monkeypatch):
    """Replace ESMF execution with command capture and an immediate stop.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace ``os.system`` in the VPRM module.

    Returns
    -------
    list[str]
        Mutable single-command capture container.
    """

    commands = []

    def capture(command):
        """Record an ESMF command and stop before external execution.

        Parameters
        ----------
        command : str
            Complete ESMF command constructed by the preprocessor.

        Raises
        ------
        CapturedESMFCommand
            Always, after recording the command.
        """

        commands.append(command)
        raise CapturedESMFCommand(command)

    monkeypatch.setattr("pyVPRM.VPRM.os.system", capture)
    return commands


def assert_dual_regional_flags(command):
    """Assert documented ESMF semantics for two regional grids.

    Parameters
    ----------
    command : str
        Complete ESMF_RegridWeightGen command string.

    Returns
    -------
    None
        The test fails if conflicting global/regional flags are present.
    """

    assert " -r " in command
    assert "--src_regional" not in command
    assert "--dest_regional" not in command


def test_to_wrf_output_uses_dual_regional_esmf_flags(monkeypatch, tmp_path):
    """Use ``-r`` alone when building WRF output regridding weights.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to capture the ESMF command.
    tmp_path : pathlib.Path
        Temporary directory used for SCRIP-grid and weight paths.

    Returns
    -------
    None
        The test captures the command before ESMF or xESMF can execute.
    """

    preprocessor = make_preprocessor()
    commands = capture_esmf_command(monkeypatch)
    output_grid = {
        "lons": np.array([172.0, 172.1]),
        "lats": np.array([-37.1, -37.0]),
    }

    with pytest.raises(CapturedESMFCommand):
        preprocessor.to_wrf_output(
            output_grid,
            driver="ESMF_RegridWeightGen",
            regridder_save_path=tmp_path / "wrf_weights.nc",
            mpi=False,
        )

    assert len(commands) == 1
    assert_dual_regional_flags(commands[0])


def test_land_cover_weights_use_dual_regional_esmf_flags(monkeypatch, tmp_path):
    """Use ``-r`` alone when building fractional land-cover weights.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to capture the ESMF command.
    tmp_path : pathlib.Path
        Temporary directory used for SCRIP-grid and weight paths.

    Returns
    -------
    None
        The test captures the command before ESMF or xESMF can execute.
    """

    preprocessor = make_preprocessor()
    preprocessor.map_to_vprm_class = {1: 3}
    commands = capture_esmf_command(monkeypatch)
    source = make_satellite_grid().assign(
        band_1=(("y", "x"), np.ones((2, 2), dtype=np.int16))
    )
    land_cover_map = satellite_data_manager(sat_img=source)

    with pytest.raises(CapturedESMFCommand):
        preprocessor.add_land_cover_map(
            land_cover_map,
            regridder_save_path=tmp_path / "land_cover_weights.nc",
            mpi=False,
        )

    assert len(commands) == 1
    assert "--ignore_unmapped" in commands[0]
    assert_dual_regional_flags(commands[0])


def test_era5_weights_use_dual_regional_esmf_flags(monkeypatch, tmp_path):
    """Use ``-r`` alone when building ERA5-to-satellite weights.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to capture the ESMF command.
    tmp_path : pathlib.Path
        Temporary directory used for SCRIP-grid and weight paths.

    Returns
    -------
    None
        The test captures the command before ESMF or xESMF can execute.
    """

    handler = make_era5_handler()
    commands = []

    def capture(command):
        """Record an ERA5 ESMF command and stop before execution.

        Parameters
        ----------
        command : str
            Complete ESMF command constructed by the meteorology handler.

        Raises
        ------
        CapturedESMFCommand
            Always, after recording the command.
        """

        commands.append(command)
        raise CapturedESMFCommand(command)

    monkeypatch.setattr("pyVPRM.meteorologies.era5_monthly_xr.os.system", capture)
    destination = xr.Dataset(coords={"lat": [48.1], "lon": [11.1]})

    with pytest.raises(CapturedESMFCommand):
        handler.regrid(
            dataset=destination,
            weights=tmp_path / "era5_weights.nc",
        )

    assert len(commands) == 1
    assert_dual_regional_flags(commands[0])
