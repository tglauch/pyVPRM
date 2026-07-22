import os
import uuid
import numpy as np
import xarray as xr
import pandas as pd
from loguru import logger

from pyVPRM.meteorologies.met_base_class import met_data_handler_base

# --- lon conversion helpers -------------------------------------------------
# ERA5 stores longitude on [0, 360); most callers want [-180, 180).

def map_function(lon):
    """Convert a longitude from [0, 360) to [-180, 180)."""
    return lon - 360 if lon > 180 else lon


def map_function_inv(lon):
    """Convert a longitude from [-180, 180) to [0, 360)."""
    return lon + 360 if lon < 0 else lon


DATA_PRODUCT_PATH = {
    "era5_land": "api.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
    "era5_single_level": "api.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
}


class met_data_handler(met_data_handler_base):
    """
    Class for reading ERA5 data from the DestinE Earth Data Hub (EDH).
    """

    def __init__(
        self,
        PAT=None,
        year=None,
        month=None,
        day=None,
        hour=None,
        keys=None,
        lat_slice=None,
        lon_slice=None,
        t0=None,
        t1=None,
        mpi=False,
        data_product="era5_land",
    ):
        if PAT is None:
            raise ValueError(
                "Need to set the access token via the PAT argument. "
                "Check https://platform.destine.eu/."
            )
        if data_product not in DATA_PRODUCT_PATH:
            raise ValueError(
                f"Unknown data_product '{data_product}'. "
                f"Choose one of {list(DATA_PRODUCT_PATH)}."
            )

        super().__init__()

        self.PAT = PAT
        self.regridded = False
        self.regridder = None
        self.rearranged = False
        self.mpi = mpi
        self.data_product = data_product
        self.instantaneous = False

        # Copy slices instead of mutating the caller's lists in place.
        self.lat_slice = list(lat_slice) if lat_slice is not None else None
        self.lon_slice = list(lon_slice) if lon_slice is not None else None
        if self.lon_slice is not None:
            self.lon_slice[0] = map_function_inv(self.lon_slice[0])
            self.lon_slice[1] = map_function_inv(self.lon_slice[1])

        # Optional overall time window, applied once (lazily) at load
        # time in load_ds() -- narrows self.ds itself, not just
        # self.ds_out per-call, so every later selection (hour lookup,
        # reduce_time, the auto-window in _ensure_data_loaded) is already
        # working within this range.
        if (t0 is None) != (t1 is None):
            raise ValueError("t0 and t1 must be given together.")
        self.time_slice = (pd.Timestamp(t0), pd.Timestamp(t1)) if t0 is not None else None

        self.keys = keys if keys is not None else []
        self.ds = None
        self.ds_out = None
        self.load_ds()

    def load_ds(self):
        url = "https://edh:{}@{}".format(self.PAT, DATA_PRODUCT_PATH[self.data_product])
        self.ds = (
            xr.open_dataset(url, chunks={}, engine="zarr")
            .astype("float32")
            .rename({"longitude": "lon", "latitude": "lat"})
        )
        if self.keys:
            self.ds = self.ds[self.keys]

        # Narrow the base dataset itself, once, using the SAME selection
        # core as every other selection call (_select). Either both
        # spatial and time bounds get applied here, or neither -- there's
        # no separate spatial-only / time-only path.
        if self.lat_slice is not None or self.lon_slice is not None or self.time_slice is not None:
            sel_dict = {}
            if self.time_slice is not None:
                t0, t1 = self.time_slice
                sel_dict["valid_time"] = slice(t0, t1)
            self.ds = self._select(sel_dict, source=self.ds)

    def _init_data_for_day(self):
        # Nothing to precompute per day for this backend; data is fetched
        # lazily/on demand in _load_data_for_hour().
        return

    def change_date(self, year=None, month=None, day=None, hour=None):
        """
        Update the current date/hour and (re)load data for it.

        NOTE: added so that `era5_handler.change_date(...)` from the
        original __main__ example actually does something on this class.
        If met_data_handler_base already implements change_date, prefer
        that implementation and remove this override.
        """
        if year is not None:
            self.year = year
        if month is not None:
            self.month = month
        if day is not None:
            self.day = day
        if hour is not None:
            self.hour = hour
        self._load_data_for_hour()

    def _select(self, sel_dict, source=None):
        """
        Single core selection routine, used by load_ds(),
        _load_data_for_hour(), reduce_time(), and reduce_space(). Applies:
          - whatever keys are already in sel_dict (e.g. an hour lookup or
            a valid_time slice)
          - self.lat_slice / self.lon_slice, including antimeridian
            wraparound handling
        """
        source = self.ds if source is None else source
        sel_dict = dict(sel_dict)

        if self.lat_slice is not None:
            lat_min, lat_max = self.lat_slice
            sel_dict["lat"] = slice(lat_max, lat_min)

        if self.lon_slice is not None:
            lon_min, lon_max = self.lon_slice
            if lon_min < lon_max:
                sel_dict["lon"] = slice(lon_min, lon_max)
                return source.sel(sel_dict)
            else:
                # Requested range wraps around the antimeridian (e.g. 350 -> 10).
                base = source.sel(sel_dict)
                part1 = base.sel(lon=slice(lon_min, 360))
                part2 = base.sel(lon=slice(0, lon_max))
                return xr.concat([part1, part2], dim="lon")

        return source.sel(sel_dict)

    def _load_selection(self, sel_dict):
        """
        Core "load a selection and make it usable" sequence, shared by
        _load_data_for_hour(), reduce_time(), and reduce_space(). Every
        call to _select() REPLACES self.ds_out with a fresh, raw (still
        accumulated, still un-rearranged, still ungridded) lazy selection
        -- so whatever the instantaneous/rearranged/regridded flags said
        about the PREVIOUS self.ds_out no longer applies and must be reset
        before re-deriving them for this one.
        """
        self.ds_out = self._select(sel_dict).compute()

        self.instantaneous = False
        self.rearranged = False
        self.regridded = False

        self.accumulated_to_inst()

    def _load_data_for_hour(self):
        if self.ds is None:
            logger.error("No dataset loaded from Earth Data Hub.")
            return

        # Caution: the date argument corresponds to the END of the ERA5
        # integration time.
        sel_dict = {
            "valid_time": "{}-{}-{} {}:00:00".format(
                self.year, self.month, self.day, self.hour
            )
        }
        self._load_selection(sel_dict)

    def _current_time_sel_dict(self):
        """
        Best-effort reconstruction of "whatever time selection is
        currently in effect", read off self.ds_out's own valid_time
        coordinate rather than re-deriving it from year/month/day/hour
        (which may not be set if the handler was last loaded via
        reduce_time() rather than change_date()). Used by reduce_space()
        so it can change the spatial box without silently dropping or
        resetting whatever time window was already loaded.
        """
        if self.ds_out is None or "valid_time" not in self.ds_out.coords:
            return {}
        vt = self.ds_out["valid_time"].values
        if vt.size == 0:
            return {}
        if vt.size == 1:
            return {"valid_time": vt[0]}
        return {"valid_time": slice(vt.min(), vt.max())}

    def reduce_time(self, t0, t1):
        """Reload the currently-selected spatial box for a new time range."""
        if self.time_slice is not None:
            bound_t0, bound_t1 = self.time_slice
            t0 = max(pd.Timestamp(t0), bound_t0)
            t1 = min(pd.Timestamp(t1), bound_t1)
            if t0 > t1:
                raise ValueError(
                    f"Requested window [{t0}, {t1}] falls outside the "
                    f"handler's configured time_slice {self.time_slice}."
                )
        self._load_selection({"valid_time": slice(t0, t1)})

    def reduce_space(self, lat_slice=None, lon_slice=None):
        """
        Update the spatial bounding box and reload it, keeping whatever
        time selection is currently in effect. Mirrors reduce_time(): one
        of the two axes changes, the other is preserved.

        Pass lat_slice/lon_slice as [min, max] in the same convention as
        the constructor args (lon in [-180, 180)); pass None to leave that
        axis's existing slice unchanged, or an empty list/tuple to clear
        it (select the full extent along that axis).
        """
        if lat_slice is not None:
            self.lat_slice = list(lat_slice) if len(lat_slice) else None

        if lon_slice is not None:
            if len(lon_slice):
                new_lon_slice = list(lon_slice)
                new_lon_slice[0] = map_function_inv(new_lon_slice[0])
                new_lon_slice[1] = map_function_inv(new_lon_slice[1])
                self.lon_slice = new_lon_slice
            else:
                self.lon_slice = None

        self._load_selection(self._current_time_sel_dict())

    def accumulated_to_inst(self):
        """
        Convert ERA5's accumulated (running-total-since-start-of-day)
        variables into per-timestep instantaneous increments.

        Stays lazy: diff/concat/groupby build a dask task graph without
        forcing computation.
        """
        if self.instantaneous:
            return

        acc_keys = [
            k
            for k in self.ds_out.keys()
            if self.ds_out[k].attrs.get("GRIB_stepType") == "accum"
        ]

        for k in acc_keys:
            acc = self.ds_out[k]
            acc = acc.assign_coords(begin_time=acc["valid_time"] - pd.Timedelta(hours=1))

            def _diff_within_day(x):
                if x.sizes.get("valid_time", 0) < 2:
                    return x
                return xr.concat(
                    [x.isel(valid_time=0), x.diff("valid_time")], dim="valid_time"
                )

            diff = acc.groupby("begin_time.date").map(_diff_within_day)
            # concat/groupby drop the original time coordinate; restore it.
            diff = diff.assign_coords(valid_time=acc["valid_time"])
            self.ds_out[k] = diff

        self.ds_out = self.ds_out.isel(valid_time=slice(1, None))
        self.instantaneous = True

    def regrid(self, lats=None, lons=None, dataset=None, n_cpus=1, weights=None,
               overwrite_regridder=False):
        import xesmf as xe

        if self.regridded:
            return

        self.rearrange_lons_lats()

        if self.regridder is None or overwrite_regridder:
            if lats is not None and lons is not None:
                t_ds_out = xr.Dataset(
                    {
                        "lat": (["lat"], lats, {"units": "degrees_north"}),
                        "lon": (["lon"], lons, {"units": "degrees_east"}),
                    }
                )
                t_ds_out = t_ds_out.set_coords(["lon", "lat"])
                self.reg_lats = lats
                self.reg_lons = lons
            elif dataset is not None:
                t_ds_out = dataset
            else:
                raise ValueError("regrid() needs either lats/lons or a target dataset.")

            if weights is None:
                raise ValueError(
                    "A `weights` file path must be provided (existing, to reuse; "
                    "or non-existing, to generate)."
                )

            if os.path.exists(str(weights)):
                logger.info("Loading regridding weights from {}".format(weights))
            else:
                bfolder = os.path.dirname(weights) or "."
                os.makedirs(bfolder, exist_ok=True)
                src_temp_path = os.path.join(bfolder, "{}.nc".format(uuid.uuid4()))
                dest_temp_path = os.path.join(bfolder, "{}.nc".format(uuid.uuid4()))
                try:
                    non_spatial_dims = [d for d in self.ds_out.dims if d not in ("lat", "lon")]
                    grid_src = self.ds_out.isel({d: 0 for d in non_spatial_dims})
                    if grid_src.data_vars:
                        first_var = list(grid_src.data_vars)[0]
                        grid_src = grid_src[[first_var]]
                    grid_src.compute().to_netcdf(src_temp_path)
                    t_ds_out.to_netcdf(dest_temp_path)

                    cmd = (
                        "ESMF_RegridWeightGen --source {} --destination {} "
                        "--weight {} -m bilinear --64bit_offset "
                        "--extrap_method nearestd --no_log".format(
                            src_temp_path, dest_temp_path, weights
                        )
                    )
                    if self.mpi:
                        cmd = "mpirun -np {} ".format(n_cpus) + cmd
                    logger.info(cmd)
                    ret = os.system(cmd)
                    if ret != 0:
                        raise RuntimeError(
                            "ESMF_RegridWeightGen failed with exit code {}".format(ret)
                        )
                finally:
                    for p in (src_temp_path, dest_temp_path):
                        if os.path.exists(p):
                            os.remove(p)

            self.regridder = xe.Regridder(
                self.ds_out, t_ds_out, "bilinear", weights=weights, reuse_weights=True
            )

        # xesmf.Regridder works fine on dask-backed input and stays lazy.
        self.ds_out = self.regridder(self.ds_out)
        self.regridded = True

    def reduce_along_lonlat(self, lon, lat, interp_method="nearest"):
        """
        Interpolate the currently-loaded slice (self.ds_out) onto specific
        lon/lat points.
        """
        target = self.ds_out if self.ds_out is not None else self.ds
        lon_vals = lon
        if not self.rearranged:
            lon_vals = [map_function_inv(x) for x in lon]
        result = target.interp(lon=("lon", lon_vals), lat=("lat", lat), method=interp_method)
        self.ds_out = result
        return result

    def rearrange_lons_lats(self):
        """Normalize longitudes to [-180, 180) and sort lat ascending."""
        if self.ds_out["lon"].values.max() > 180:
            self.ds_out = self.ds_out.assign_coords(
                lon=[map_function(i) for i in self.ds_out.coords["lon"].values]
            )
            self.ds_out = self.ds_out.sortby("lon")
        if float(self.ds_out["lat"][0]) > float(self.ds_out["lat"][-1]):
            self.ds_out = self.ds_out.sortby("lat")
            self.rearranged = True

    @staticmethod
    def _as_time_index(times):
        """Normalize `times` (scalar, list/tuple, or ndarray) into a DatetimeIndex."""
        return pd.DatetimeIndex(np.atleast_1d(np.asarray(times, dtype="datetime64[ns]")))

    def _times_covered(self, times):
        """Check whether self.ds_out already spans the requested time(s)."""
        if self.ds_out is None or "valid_time" not in self.ds_out.coords:
            return False
        vt = self.ds_out["valid_time"].values
        if vt.size == 0:
            return False
        lo, hi = pd.Timestamp(vt.min()), pd.Timestamp(vt.max())
        t_idx = self._as_time_index(times)
        return lo <= t_idx.min() and t_idx.max() <= hi

    def _ensure_data_loaded(self, times=None):
        """
        Make sure self.ds_out contains the data needed to answer a
        get_data() call, loading it if necessary. This is what lets
        get_data() be called directly (e.g. right after construction, or
        for one timestamp or a whole array of them) without first calling
        change_date()/reduce_time() by hand.
        """
        if self.ds_out is not None and (times is None or self._times_covered(times)):
            return  # already have what we need, nothing to do

        if times is not None:
            t_idx = self._as_time_index(times)
            t0 = t_idx.min() - pd.Timedelta(hours=2)
            t1 = t_idx.max() + pd.Timedelta(hours=2)
            self.reduce_time(t0, t1)
        elif all(getattr(self, attr, None) is not None for attr in ("year", "month", "day", "hour")):
            self._load_data_for_hour()
        else:
            raise RuntimeError(
                "get_data() needs either a `times` argument or a full "
                "year/month/day/hour to know what to load. Pass `times=...`, "
                "call change_date(...)/reduce_time(...) first, or set a full "
                "date on the constructor."
            )

    def get_data(self, lonlat=None, key=None, times=None, interp_method="nearest"):
        self._ensure_data_loaded(times)

        if not self.instantaneous:
            self.accumulated_to_inst()

        if lonlat is None and not self.rearranged:
            self.rearrange_lons_lats()

        tmp = self.ds_out if key is None else self.ds_out[key]

        if lonlat is not None or times is not None:
            tmp = tmp

        if lonlat is not None:
            lon = lonlat[0]
            if isinstance(lon, (list, tuple, np.ndarray)):
                if not self.rearranged:
                    lon = [map_function_inv(x) for x in lon]
                tmp = tmp.interp(lon=("lon", lon), lat=("lat", lonlat[1]), method=interp_method)
            else:
                if not self.rearranged:
                    lon = map_function_inv(lon)
                tmp = tmp.interp(lon=lon, lat=lonlat[1], method=interp_method)

        if times is not None:
            tmp = tmp.interp(valid_time=times, method=interp_method)

        return tmp


if __name__ == "__main__":

    PAT = os.environ.get("EDH_PAT")  # set your DestinE token in this env var
    year = "2000"
    month = 2
    day = 20
    hour = 5  # UTC hour
    position = {"lat": 50.30493, "long": 5.99812}

    era5_handler = met_data_handler(PAT=PAT, year=year, month=month, day=day, hour=hour)
    era5_handler.change_date(hour=hour)
    ret = era5_handler.get_data(lonlat=(position["long"], position["lat"]))
    logger.info(ret)