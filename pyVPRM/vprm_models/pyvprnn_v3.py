import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from pyVPRM.lib.functions import sel_nearest_valid
from pyVPRM.vprm_models.pyvprnn_v1 import (
    pyvprnn_v1,
    BatchGenerator,
    TimeLimitCallback,
    BroadcastToImage,
    ExpandLastDim,
    GlobalSumPooling,
    DayMask,
    SelectFeatures,
    GPPPenalty,
    TimeIntegratedRatioPenalty,
    apply_met_scaling,
)


# =============================================================================
# Slow-memory features, split by mechanism instead of by a single source:
#
#   - Pulse onset + fast decay (hours to a few days) needs PRECISE timing -
#     this is exactly where ERA5's precip forcing is least trustworthy
#     (convective timing/intensity error), so this half uses real measured
#     precipitation (self.precip_var, "P_F" at training time).
#
#   - Seasonal drought memory (a week to ~90 days) is a rolling
#     mean/min over a long window - by construction, hour-level timing
#     error gets smoothed out at that scale, so the source's timing
#     accuracy barely matters here. This half uses swvl1_era5 instead,
#     which has an unrelated but genuinely useful property precip doesn't:
#     it's available everywhere, with no upscaling-availability problem at
#     all (only an accuracy one, and one this design deliberately doesn't
#     lean on for anything timing-sensitive).
#
# UPSCALING NOTE: because of this split, only the SHORT-tau precip features
# need a bias-corrected gridded precip product off-tower (precip_var must
# not be the raw tower P_F there). The long-tau swvl1 features need no such
# bridge - swvl1_era5 was never tower-only to begin with.
# =============================================================================

def apply_snow_gating(precip_values, snow_depth_values, melt_threshold=1e-3):
    """
    Cheap leaky-reservoir gate: withholds precipitation's contribution to
    the pulse-onset API while snow is on the ground, then releases it back
    in proportional to how much the snowpack has shrunk since the previous
    step (a coarse melt proxy), with anything still held released in full
    once the snowpack is gone.

    This is NOT a real snowpack water-equivalent model - it doesn't track
    SWE separately from depth, and ignores compaction/sublimation. It's a
    gate against the specific, obvious failure mode of a snowfall event
    registering as an immediate rewetting pulse; not a substitute for a
    real snow hydrology model if melt timing itself needs to be precise.
    """
    n = len(precip_values)
    gated = np.zeros(n, dtype=np.float32)
    reservoir = 0.0
    prev_depth = float(snow_depth_values[0]) if n else 0.0

    for i in range(n):
        depth = float(snow_depth_values[i])
        p = float(precip_values[i])

        if depth > melt_threshold:
            reservoir += p
            if depth < prev_depth and prev_depth > melt_threshold:
                melt_fraction = (prev_depth - depth) / prev_depth
                release = reservoir * melt_fraction
                gated[i] = release
                reservoir -= release
        else:
            gated[i] = p + reservoir
            reservoir = 0.0

        prev_depth = depth

    return gated


def compute_api(precip_values, dt_hours, tau_hours):
    """
    Exponentially-weighted Antecedent Precipitation Index: a decaying
    running sum of precipitation, with tau_hours as the memory timescale.
    Used here for the FAST (pulse onset/decay) branch only - see module
    docstring above for why the slow branch uses swvl1 rolling stats
    instead.
    """
    api = np.empty(len(precip_values), dtype=np.float32)
    decay = float(np.exp(-dt_hours / tau_hours))
    running = 0.0
    for i, p in enumerate(precip_values):
        running = running * decay + float(p)
        api[i] = running
    return api


def compute_rolling_stat(values, window_steps, stat):
    """Rolling mean/min over the trailing window_steps, min_periods=1 (mild
    early-record underestimate, not hard-excluded)."""
    s = pd.Series(values)
    roll = s.rolling(window=window_steps, min_periods=1)
    if stat == "mean":
        return roll.mean().values.astype(np.float32)
    if stat == "min":
        return roll.min().values.astype(np.float32)
    raise ValueError(f"Unknown stat '{stat}'")


class PeakAndTrendPooling1D(layers.Layer):
    """
    Concatenates global average pooling (trend / sustained-warming signal)
    with global max pooling (peak signal) over the time axis of a Conv1D
    output. Average pooling alone smooths out sharp excursions that can
    trigger a respiration peak; max pooling preserves them regardless of
    where in the window they occurred.
    """

    def call(self, x):
        avg = tf.reduce_mean(x, axis=1)
        mx = tf.reduce_max(x, axis=1)
        return tf.concat([avg, mx], axis=-1)

    def get_config(self):
        return super().get_config()


# =============================================================================
# Batch generator
# =============================================================================

class LaggedBatchGenerator(BatchGenerator):
    """
    Extends BatchGenerator's met_vars dispatch with:

      - engineered current-value features for Reco: snow-gated precip API
        (pulse onset/decay), swvl1 rolling mean/min (seasonal drought
        memory), and air/soil temperature gradient -- all computed here,
        so they flow through the ordinary
        met_input path with zero new learned parameters
      - a separate, fixed-time-step lag history array (t2m only, by
        default) for the Reco branch's Conv1D diffusion/peak encoder

    New met_vars names recognized on top of everything BatchGenerator
    already handles:
      - "t_gradient_air_soil"    : t2m - stl2_era5, current timestep.
      - "precip_api_tau<H>h"     : snow-gated Antecedent Precipitation
                                    Index at memory timescale H hours,
                                    from self.precip_var. Intended for
                                    SHORT H (pulse onset/decay) only.
      - "swvl1_roll_mean_<H>h"   : rolling mean of swvl1_era5 over H hours.
      - "swvl1_roll_min_<H>h"    : rolling min of swvl1_era5 over H hours.
                                    Intended for LONG H (seasonal drought
                                    memory) only.

    precip_var (constructor arg, default "P_F") is the single place that
    decides which precipitation source feeds the API family - swap it for
    a bias-corrected gridded product at upscale time.
    swvl1_era5-derived features need no equivalent swap.

    UNITS: dt_hours is inferred from ds_cropped's own datetime_utc axis.
    lag_hours (for the Conv1D branch) are real hours, converted to
    row-steps using this same inferred dt_hours.
    """

    def __init__(self, *args, lagged_met_vars=("t2m",), lag_hours=None,
                 precip_var="P_F", **kwargs):
        self.lagged_met_vars = list(lagged_met_vars)
        self.lag_hours = list(lag_hours) if lag_hours is not None else [1, 2, 4, 8, 16, 32, 64, 128]
        self.precip_var = precip_var
        super().__init__(*args, **kwargs)

        valid_mask = ~np.isnan(self.lagged_met_array).any(axis=(1, 2))
        n_before = len(self.indexes)
        self.indexes = self.indexes[valid_mask[self.indexes]]
        self.n = len(self.indexes)
        n_dropped = n_before - self.n
        if n_dropped:
            import logging
            logging.getLogger("vprm_pipeline").warning(
                "Dropped %d/%d requested samples lacking full lag history "
                "(need >= %.0fh of prior record).",
                n_dropped, n_before, max(self.lag_hours),
            )

    def _fetch_era5_nearest(self, var):
        return sel_nearest_valid(
            self.ds_cropped[[var]].compute(),
            lon=self.ds_cropped.attrs["site_lon"],
            lat=self.ds_cropped.attrs["site_lat"],
        )[var]

    # A couple of feature names used in met_vars don't match their actual
    # ds_cropped variable name: raw ERA5 t2m and snow depth are stored as
    # "t2m_era5"/"sd_era5" (gridded, needing the same nearest-valid-point
    # reduction every other *_era5 variable already gets via
    # _fetch_era5_nearest), not under the bare "t2m"/"sd" the rest of this
    # file's naming otherwise uses. Kept as an explicit table rather than
    # guessed/inferred at runtime, since it's a fact about the dataset
    # schema, not something to detect per call.
    ERA5_KEY_OVERRIDES = {"t2m": "t2m_era5", "sd": "sd_era5"}

    def init_training_cache(self, sat_vars, met_vars, met_dim=1):
        self.time = self.ds_cropped["datetime_utc"].values
        self.time_index = {t: i for i, t in enumerate(self.time)}
        self.sat_time_index = self.ds_cropped["days_since_t0"].values

        dt_hours = float(
            np.median(np.diff(pd.DatetimeIndex(self.time))).astype("timedelta64[s]").astype(float) / 3600.0
        )
        if dt_hours <= 0:
            raise ValueError("Could not infer a positive time step from datetime_utc.")
        self.dt_hours_ = dt_hours

        # Base variables the engineered features below depend on. t2m/sd go
        # through the override table + the same sel_nearest_valid reduction
        # stl2_era5/swvl1_era5 already use. precip_var (P_F) is tower point
        # data, not gridded ERA5, so it's fetched plain.
        t2m_raw = self._fetch_era5_nearest(self.ERA5_KEY_OVERRIDES["t2m"]).sel(datetime_utc=self.time).values.astype(np.float32)
        stl2_raw = self._fetch_era5_nearest("stl2_era5").sel(datetime_utc=self.time).values.astype(np.float32)
        swvl1_raw = self._fetch_era5_nearest("swvl1_era5").sel(datetime_utc=self.time).values.astype(np.float32)
        precip_raw = self.ds_cropped[self.precip_var].sel(datetime_utc=self.time).values.astype(np.float32)
        snow_raw = self._fetch_era5_nearest(self.ERA5_KEY_OVERRIDES["sd"]).sel(datetime_utc=self.time).values.astype(np.float32)

        precip_gated = apply_snow_gating(precip_raw, snow_raw)

        met_list = []
        for v in met_vars:
            if v == "t_gradient_air_soil":
                arr = t2m_raw - stl2_raw
            elif v.startswith("precip_api_tau") and v.endswith("h"):
                tau = float(v[len("precip_api_tau"):-1])
                arr = compute_api(precip_gated, dt_hours, tau)
            elif v.startswith("swvl1_roll_mean_") and v.endswith("h"):
                h = float(v[len("swvl1_roll_mean_"):-1])
                window_steps = max(1, int(round(h / dt_hours)))
                arr = compute_rolling_stat(swvl1_raw, window_steps, "mean")
            elif v.startswith("swvl1_roll_min_") and v.endswith("h"):
                h = float(v[len("swvl1_roll_min_"):-1])
                window_steps = max(1, int(round(h / dt_hours)))
                arr = compute_rolling_stat(swvl1_raw, window_steps, "min")
            elif v == "t2m":
                arr = t2m_raw
            elif v == "sd":
                arr = snow_raw
            elif v == "stl2_era5":
                arr = stl2_raw
            elif v == "swvl1_era5":
                arr = swvl1_raw
            elif v.endswith("_era5"):
                da = self._fetch_era5_nearest(v).sel(datetime_utc=self.time)
                arr = apply_met_scaling(v, da.values)
            else:
                da = self.ds_cropped[v].sel(datetime_utc=self.time)
                arr = apply_met_scaling(v, da.values)

            met_list.append(np.asarray(arr, dtype=np.float32))

        self.met_array = np.stack(met_list, axis=-1).astype(np.float32)
        if met_dim == 1:
            self.met_array = self.met_array[:, None, None, :]

        # Identical to BatchGenerator.init_training_cache from here.
        sw_in_pot = self.ds_cropped["SW_IN_POT"].sel(datetime_utc=self.time).values.astype(np.float32)
        self.sw_in_pot_array = sw_in_pot
        if met_dim == 1:
            self.sw_in_pot_array = self.sw_in_pot_array[:, None, None, None]

        self.sat_array = np.stack(
            [self.ds_cropped[v].values for v in sat_vars], axis=-1
        ).astype(np.float32)

        self.lc = np.moveaxis(
            self.ds_cropped["land_cover_map"].sel(vprm_classes=self.land_cover_classes).values,
            0, -1,
        ).astype(np.float32)

        self.nirv_max = self.ds_cropped["nirv_90pct"].values[..., None].astype(np.float32)
        self.nirv_min = self.ds_cropped["nirv_10pct"].values[..., None].astype(np.float32)

        self.y_array = self.ds_cropped[self._target].sel(datetime_utc=self.time).values.astype(np.float32)
        self.y_unc_array = self.ds_cropped[self._unc].sel(datetime_utc=self.time).values.astype(np.float32)
        self.fp_array = self.ds_cropped["ffp_footprint"].sel(t=self.time).values.astype(np.float32)
        self.mask_static = self.ds_cropped["flux_mask"].values.astype(np.float32)
        self.static_stack = np.concatenate([self.nirv_max, self.nirv_min, self.lc], axis=-1)
        self.y_pack = np.stack([self.y_array, self.y_unc_array], axis=-1).astype(np.float32)

        # ---- fixed-unit lag history for the Conv1D diffusion/peak branch ----
        missing = set(self.lagged_met_vars) - set(met_vars)
        if missing:
            raise ValueError(
                f"lagged_met_vars {sorted(missing)} not present in met_vars {met_vars} - "
                f"every lagged variable must also be fetched as a regular met_var."
            )
        lag_steps = [max(1, int(round(h / dt_hours))) for h in self.lag_hours]
        self.lag_steps_ = lag_steps
        self.lag_window = max(lag_steps)

        n_times = len(self.time)
        n_vars = len(self.lagged_met_vars)
        n_lags = len(lag_steps)
        met_flat = self.met_array[:, 0, 0, :]

        lagged = np.full((n_times, n_lags, n_vars), np.nan, dtype=np.float32)
        for v_idx, var in enumerate(self.lagged_met_vars):
            var_col = met_flat[:, met_vars.index(var)]
            for lag_idx, lag in enumerate(lag_steps):
                lagged[lag:, lag_idx, v_idx] = var_col[: n_times - lag]
        self.lagged_met_array = lagged

        n_nan_rows = np.isnan(lagged).any(axis=(1, 2)).sum()
        if n_nan_rows:
            import logging
            logging.getLogger("vprm_pipeline").warning(
                "%d of %d timesteps have incomplete lag history (NaN) - train/val "
                "time selections should start at least %.0fh into the record.",
                n_nan_rows, n_times, max(self.lag_hours),
            )

    def __getitem__(self, batch_index):
        (sat, static, met, sw_in_pot, fp, mask), ypack = super().__getitem__(batch_index)
        batch_idxs = self.indexes[
            batch_index * self.batch_size: (batch_index + 1) * self.batch_size
        ]
        lagged_met = self.lagged_met_array[batch_idxs]
        return (sat, static, met, sw_in_pot, lagged_met, fp, mask), ypack


# =============================================================================
# Model
# =============================================================================

class pyvprnn_v3(pyvprnn_v1):
    """
    pyvprnn_v1 plus a time-sensitive Reco encoder, built around THREE
    mechanisms now, each matched to the timescale/source combination it's
    actually good at:

      FAST DIFFUSION/PEAK (hours) -- t2m only, uniform-step dilated causal
      Conv1D -> peak+trend pooling. Air->soil temperature diffusion delay
      and short sharp excursions. stl2_era5 stays current-value only;
      t_gradient_air_soil captures the instantaneous diffusion driving
      force directly.

      FAST PULSE ONSET/DECAY (hours to a few days) -- snow-gated
      Antecedent Precipitation Index from precip_var, at API_TAUS_HOURS
      (short only). Uses real measured precipitation because pulse timing
      is exactly where ERA5's precip forcing is least trustworthy.

      SLOW SEASONAL MEMORY (a week to ~90 days) -- swvl1_era5 rolling
      mean/min at SWVL_ROLLING_HOURS (long only). Uses ERA5 soil moisture
      because at this timescale, timing error gets smoothed out by the
      rolling window itself, and swvl1_era5 has no upscaling-availability
      problem the way tower precip does.

    This split means only the pulse-onset features need a bias-corrected
    gridded precip source for upscaling (precip_var); the seasonal-memory
    features need no equivalent bridge.

    GPP branch is unchanged and receives no lagged/engineered inputs.

    The Reco-vs-GPP consistency constraint (TimeIntegratedRatioPenalty) is
    unchanged from pyvprnn_v1/v2: applied to reco_map using gpp_map and
    day_mask, after the Reco branch produces its map.
    """

    DEFAULT_LAG_HOURS = [1, 2, 4, 8, 16, 32, 64, 128]
    DEFAULT_LAGGED_MET_VARS = ["t2m"]
    API_TAUS_HOURS = (6, 24, 72, 168)            # precip-based: pulse onset + fast decay only
    SWVL_ROLLING_HOURS = (168, 720, 2160, 3600)   # swvl1-based: week -> seasonal memory only
    DEFAULT_PRECIP_VAR = "P_F"  # tower measurement; swap for a gridded/bias-corrected var at upscale time

    def __init__(self, lagged_met_vars=None, lag_hours=None, precip_var=None, **kwargs):
        super().__init__(**kwargs)

        self.lagged_met_vars = list(lagged_met_vars) if lagged_met_vars is not None else list(self.DEFAULT_LAGGED_MET_VARS)
        self.lag_hours = list(lag_hours) if lag_hours is not None else list(self.DEFAULT_LAG_HOURS)
        self.precip_var = precip_var if precip_var is not None else self.DEFAULT_PRECIP_VAR

        api_vars = [f"precip_api_tau{h}h" for h in self.API_TAUS_HOURS]
        swvl_roll_vars = []
        for h in self.SWVL_ROLLING_HOURS:
            swvl_roll_vars += [f"swvl1_roll_mean_{h}h", f"swvl1_roll_min_{h}h"]

        self.met_vars = (
            ["t2m", "ssrd", "RH_from_VDP", "swvl1_era5", "swvl2_era5", "sd",
             "stl2_era5", "t_gradient_air_soil"]
            + api_vars + swvl_roll_vars
        )

        self.gpp_met_vars = ["t2m", "ssrd", "RH_from_VDP", "swvl2_era5", "sd"]

        self.reco_met_vars = (
            ["t2m", "RH_from_VDP", "swvl1_era5", "stl2_era5", "sd",
             "t_gradient_air_soil"]
            + api_vars + swvl_roll_vars
        )

        for v in self.lagged_met_vars:
            if v not in self.met_vars:
                self.met_vars.append(v)

        unknown = set(self.reco_met_vars) - set(self.met_vars)
        if unknown:
            raise ValueError(f"reco_met_vars {sorted(unknown)} missing from met_vars.")

    def load_model(self, path):
        self.pixel_model = load_model(path, custom_objects={
            "BroadcastToImage": BroadcastToImage,
            "ExpandLastDim": ExpandLastDim,
            "GlobalSumPooling": GlobalSumPooling,
            "DayMask": DayMask,
            "SelectFeatures": SelectFeatures,
            "GPPPenalty": GPPPenalty,
            "TimeIntegratedRatioPenalty": TimeIntegratedRatioPenalty,
            "PeakAndTrendPooling1D": PeakAndTrendPooling1D
        })
    
    
    def train(self, save_path_model,
              save_path_history=None,
              train_params={"batch_size": 42,
                             "max_runtime_in_seconds": None,
                             "epochs": 1000,
                             "patience": 10,
                             "plateau_patience": 5,
                             "learning rate": 5e-4,
                             "workers": 1,
                             "multiprocessing": False,
                             "max_queue_size": 1,
                             "loss": "nll_loss_from_stacked"},
              target="NEE_VUT_REF",
              target_unc="NEE_VUT_REF_JOINTUNC",
              cv_fold=0,
              random_state=41):

        train_times = self.cv_folds[cv_fold]["train_times"]
        qc_train = self.ds_cropped["NEE_VUT_REF_QC"].sel(datetime_utc=train_times)
        wrong_nigttime_train = ((self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=train_times) < 0) &
                                 (self.ds_cropped["ssrd"].sel(datetime_utc=train_times) == 0))
        train_times_qc0 = train_times[(qc_train == 0) & ~wrong_nigttime_train]

        val_times = self.cv_folds[cv_fold]["val_times"]
        qc_val = self.ds_cropped["NEE_VUT_REF_QC"].sel(datetime_utc=val_times)
        wrong_nigttime_val = ((self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=val_times) < 0) &
                               (self.ds_cropped["ssrd"].sel(datetime_utc=val_times) == 0))
        val_times_qc0 = val_times[(qc_val == 0) & ~wrong_nigttime_val]

        gen_kwargs = dict(
            land_cover_classes=self.land_cover_classes,
            lagged_met_vars=self.lagged_met_vars,
            lag_hours=self.lag_hours,
            precip_var=self.precip_var,
        )

        train_gen = LaggedBatchGenerator(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            times=train_times_qc0,
            workers=train_params["workers"],
            use_multiprocessing=train_params["multiprocessing"],
            max_queue_size=train_params["max_queue_size"],
            target=target, unc=target_unc,
            **gen_kwargs,
        )

        val_gen = LaggedBatchGenerator(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            times=val_times_qc0,
            shuffle=False,
            target=target, unc=target_unc,
            **gen_kwargs,
        )

        (Xsat_batch, Xstatic_batch, Xmet_batch, _, Xmet_lagged_batch, _, _), _ = train_gen[0]

        n_sat_features = Xsat_batch.shape[-1]
        n_static_features = Xstatic_batch.shape[-1]
        n_met_features = Xmet_batch.shape[-1]

        gpp_met_idx = [self.met_vars.index(v) for v in self.gpp_met_vars]
        reco_met_idx = [self.met_vars.index(v) for v in self.reco_met_vars]
        filter_size = 1

        # =========================================================
        # Inputs
        # =========================================================
        self.sat_input = layers.Input(shape=(None, None, n_sat_features), name="sat")
        self.static_input = layers.Input(shape=(None, None, n_static_features), name="static")
        self.met_input = layers.Input(shape=(None, None, n_met_features), name="met")
        self.sw_in_pot_input = layers.Input(shape=(None, None, 1), name="sw_in_pot")
        self.met_input_lagged = layers.Input(
            shape=(Xmet_lagged_batch.shape[1], Xmet_lagged_batch.shape[2]), name="met_lagged"
        )
        self.fp_input = layers.Input(shape=(None, None), name="fp")
        self.flux_mask_input = layers.Input(shape=(None, None), name="flux_mask")

        flux_mask_exp = ExpandLastDim(name="flux_mask_exp")(self.flux_mask_input)
        fp_exp = ExpandLastDim(name="fp_exp")(self.fp_input)

        sat_static = layers.Concatenate(name="sat_static_concat")([self.sat_input, self.static_input])

        met_gpp = SelectFeatures(gpp_met_idx, name="met_gpp")(self.met_input)
        met_bc_gpp = BroadcastToImage(name="met_bc_gpp")([met_gpp, sat_static])

        met_reco = SelectFeatures(reco_met_idx, name="met_reco")(self.met_input)
        met_bc_reco = BroadcastToImage(name="met_bc_reco")([met_reco, sat_static])

        # =========================================================
        # Day/night mask
        # =========================================================
        sw_in_pot_bc = BroadcastToImage(name="sw_in_pot_bc")([self.sw_in_pot_input, sat_static])
        day_mask = DayMask(0, name="day_mask")(sw_in_pot_bc)

        # =========================================================
        # Fast-timescale lag encoder (Reco branch only): t2m history via
        # uniform-dilation causal Conv1D, then peak+trend pooling.
        # =========================================================
        x_met_lagged = self.met_input_lagged
        for dilation in (1, 2, 4, 8):
            x_met_lagged = layers.Conv1D(
                filters=8, kernel_size=2, dilation_rate=dilation,
                padding="causal", activation="softplus",
                kernel_initializer="he_normal",
                name=f"conv1d_lagged_dilation_{dilation}",
            )(x_met_lagged)
        x_met_lagged_summary = PeakAndTrendPooling1D(name="peak_trend_pool")(x_met_lagged)
        x_met_lagged_summary = layers.Dense(
            4, activation="softplus", name="dense_lagged_summary"
        )(x_met_lagged_summary)
        met_bc_lagged = BroadcastToImage(name="met_bc_lagged")([x_met_lagged_summary, sat_static])

        # =========================================================
        # GPP branch (unchanged from pyvprnn_v1)
        # =========================================================
        x_gpp = layers.Concatenate(name="gpp_concat")([self.sat_input, self.static_input, met_bc_gpp])
        for i in range(6):
            x_gpp = layers.Conv2D(32, filter_size, padding="same", activation="softplus",
                                   kernel_initializer="he_normal")(x_gpp)
        x_gpp = layers.Conv2D(
            1, 1, activation="softplus", kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.3), name="x_gpp_map",
        )(x_gpp)
        x_gpp = GPPPenalty(threshold=40.0, weight=1e-4, name="gpp_penalty")(x_gpp)

        gpp_map_day_mask = layers.Multiply(name="gpp_map_masked")([x_gpp, day_mask])
        gpp_map = layers.Multiply(name="gpp_map")([gpp_map_day_mask, flux_mask_exp])
        gpp_weighted = layers.Multiply(name="gpp_weighted")([gpp_map, fp_exp])
        gpp_sum = GlobalSumPooling(name="gpp_sum_raw")(gpp_weighted)

        # =========================================================
        # Reco branch -- current-value met (incl. precip API + swvl1
        # rolling stats + t2m/soil gradient) plus the fast-lag peak/trend
        # summary.
        # =========================================================
        x_reco = layers.Concatenate(name="reco_concat")([
            self.sat_input, self.static_input, met_bc_reco, met_bc_lagged,
        ])
        for i in range(6):
            x_reco = layers.Conv2D(32, filter_size, padding="same", activation="softplus",
                                    kernel_initializer="he_normal")(x_reco)
        x_reco_map = layers.Conv2D(
            1, 1, activation="softplus", kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.7), name="x_reco_map",
        )(x_reco)
        reco_map = layers.Multiply(name="reco_map")([x_reco_map, flux_mask_exp])

        # ---- Reco-to-GPP constraint: unchanged from pyvprnn_v1/v2. ----
        reco_map = TimeIntegratedRatioPenalty(
            max_ratio=1.0,
            weight=1e-4,
            gpp_floor=1.0,
            name="time_integrated_ratio_penalty",
        )([gpp_map, reco_map, day_mask])

        reco_weighted = layers.Multiply(name="reco_weighted")([reco_map, fp_exp])
        reco_sum = GlobalSumPooling(name="reco_sum")(reco_weighted)

        # =========================================================
        # NEE (physics only)
        # =========================================================
        nee = layers.Subtract(name="nee")([reco_sum, gpp_sum])
        nee = layers.Flatten(name="Output")(nee)

        self.model = Model(
            inputs=[self.sat_input, self.static_input, self.met_input, self.sw_in_pot_input,
                    self.met_input_lagged, self.fp_input, self.flux_mask_input],
            outputs=nee,
        )

        def nll_loss_from_stacked(y_with_sigma_true, y_pred):
            y_true = y_with_sigma_true[..., 0][..., None]
            sigma = tf.maximum(y_with_sigma_true[..., 1], 0.5)[..., None]
            return tf.reduce_mean((y_true - y_pred) ** 2 / (2 * sigma ** 2) + 0.5 * tf.math.log(2 * np.pi * sigma ** 2))

        def mse_true_only(y_with_sigma_true, y_pred):
            y_true = y_with_sigma_true[..., 0][..., None]
            return tf.reduce_mean(tf.square(y_true - y_pred))

        def nll_loss_laplace_from_stacked(y_with_sigma_true, y_pred, sigma_floor=0.5):
            y_true = y_with_sigma_true[..., 0][..., None]
            sigma = tf.maximum(y_with_sigma_true[..., 1], sigma_floor)[..., None]
            b = sigma / tf.sqrt(2.0)
            return tf.reduce_mean(tf.abs(y_true - y_pred) / b + tf.math.log(2 * b))

        if train_params["loss"] == "nll_loss_from_stacked":
            loss = nll_loss_from_stacked
        elif train_params["loss"] == "nll_loss_laplace_from_stacked":
            loss = nll_loss_laplace_from_stacked
        elif train_params["loss"] == "mse":
            loss = mse_true_only
        else:
            raise NotImplementedError(f"Unknown loss '{train_params['loss']}'")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=train_params["learning rate"]),
            loss=loss,
            metrics=[mse_true_only],
        )
        self.model.summary()

        early_stop = EarlyStopping(monitor="val_loss", patience=train_params["patience"], restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=train_params["plateau_patience"], min_lr=1e-6, verbose=1,
        )
        callbacks = [early_stop, reduce_lr]
        if train_params["max_runtime_in_seconds"] is not None:
            callbacks.append(TimeLimitCallback(train_params["max_runtime_in_seconds"]))

        history = self.model.fit(
            train_gen, validation_data=val_gen, epochs=train_params["epochs"], callbacks=callbacks,
        )

        best_val_loss = min(history.history["val_loss"])
        print("Best val_loss:", best_val_loss)

        hist_df = pd.DataFrame(history.history)
        hist_df["epoch"] = range(1, len(hist_df) + 1)
        if save_path_history is not None:
            hist_df.to_csv(save_path_history, index=False)

        self.pixel_model = Model(
            inputs=[self.sat_input, self.static_input, self.met_input, self.sw_in_pot_input,
                    self.met_input_lagged, self.flux_mask_input],
            outputs=[self.model.get_layer("gpp_map").output, self.model.get_layer("reco_map").output],
            name="pixel_flux_model",
        )
        self.pixel_model.save(save_path_model)