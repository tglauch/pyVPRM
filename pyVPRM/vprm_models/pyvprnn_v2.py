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
# pyvprnn_v2 -- simplified successor of pyvprnn_v3
#
# Changes w.r.t. pyvprnn_v3:
#   - No temperature lag encoder (Conv1D + peak/trend pooling) and no
#     lagged-met input. Heat diffusion into the soil is represented directly
#     by ERA5 soil temperature at two depths (stl1: 0-7 cm, stl2: 7-28 cm).
#     -> generator and network have the same 6-input signature as pyvprnn_v1.
#   - No air-soil temperature gradient feature (t2m - stl2), which mainly
#     acted as a day/night switch for the Reco branch.
#   - GPP branch additionally receives swvl1_era5 (near-surface root zone),
#     which responds to rain events within hours to days; swvl2_era5 (deeper
#     roots) is kept.
#   - Consistent input scaling for engineered/raw features:
#     temperatures in degC, precipitation APIs log1p-transformed.
#
# Unchanged w.r.t. pyvprnn_v3:
#   - Precipitation API timescales (6, 24, 72, 168 h) from tower precipitation
#     (pulse onset/decay), snow-gated.
#   - swvl1 rolling mean/min drought memory (168, 720, 2160, 3600 h).
#   - GPP penalty, time-integrated Reco/GPP ratio penalty, day mask, loss.
#
# Feature/source logic (as in v3):
#   - Pulse onset + fast decay needs precise timing -> measured precipitation
#     (self.precip_var, "P_F" at training time; swap for a gridded product at
#     upscaling time).
#   - Seasonal drought memory is a rolling statistic over long windows, where
#     timing errors are smoothed out -> swvl1_era5, available everywhere.
# =============================================================================


def apply_snow_gating(precip_values, snow_depth_values, melt_threshold=1e-3):
    """
    Cheap leaky-reservoir gate: withholds precipitation's contribution to
    the pulse-onset API while snow is on the ground, then releases it back
    in proportion to how much the snowpack has shrunk since the previous
    step (a coarse melt proxy), with anything still held released in full
    once the snowpack is gone.

    This is NOT a real snowpack water-equivalent model. It is a gate against
    the failure mode of a snowfall event registering as an immediate
    rewetting pulse.
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
    running sum of precipitation with memory timescale tau_hours.
    API_t = P_t + exp(-dt/tau) * API_{t-1}
    """
    api = np.empty(len(precip_values), dtype=np.float32)
    decay = float(np.exp(-dt_hours / tau_hours))
    running = 0.0
    for i, p in enumerate(precip_values):
        running = running * decay + float(np.nan_to_num(p))
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


def to_celsius(values):
    """ERA5 temperatures are stored in K; convert to degC if needed so that
    all temperature inputs share the same scale."""
    values = np.asarray(values, dtype=np.float32)
    if np.nanmedian(values) > 150.0:
        values = values - 273.15
    return values


# =============================================================================
# Batch generator
# =============================================================================

class EngineeredBatchGenerator(BatchGenerator):
    """
    BatchGenerator plus engineered met features. No lag history, i.e. the
    generator returns the same (sat, static, met, sw_in_pot, fp, mask), y
    tuple as pyvprnn_v1's BatchGenerator.

    Met-variable names recognized on top of BatchGenerator:
      - "t2m", "sd"              : ERA5 2 m temperature / snow depth
                                    (stored as t2m_era5 / sd_era5).
      - "stl1_era5", "stl2_era5" : ERA5 soil temperature, layers 1 and 2.
      - "swvl1_era5"             : ERA5 soil moisture, layer 1 (raw).
      - "precip_api_tau<H>h"     : snow-gated Antecedent Precipitation Index
                                    at timescale H hours from self.precip_var,
                                    log1p-transformed.
      - "swvl1_roll_mean_<H>h"   : rolling mean of swvl1_era5 over H hours.
      - "swvl1_roll_min_<H>h"    : rolling min of swvl1_era5 over H hours.
    Everything else is handled exactly as in v3 (apply_met_scaling).

    precip_var (default "P_F") is the single place that decides which
    precipitation source feeds the API features - swap it at upscale time.
    """

    ERA5_KEY_OVERRIDES = {"t2m": "t2m_era5", "sd": "sd_era5"}
    TEMPERATURE_VARS = ("t2m", "stl1_era5", "stl2_era5")

    def __init__(self, *args, precip_var="P_F", **kwargs):
        self.precip_var = precip_var
        super().__init__(*args, **kwargs)

    def _fetch_era5_nearest(self, var):
        return sel_nearest_valid(
            self.ds_cropped[[var]].compute(),
            lon=self.ds_cropped.attrs["site_lon"],
            lat=self.ds_cropped.attrs["site_lat"],
        )[var]

    def _era5_at_site(self, var):
        key = self.ERA5_KEY_OVERRIDES.get(var, var)
        return (self._fetch_era5_nearest(key)
                .sel(datetime_utc=self.time).values.astype(np.float32))

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

        # Base variables for the engineered features (fetched lazily, once).
        cache = {}

        def base(var):
            if var not in cache:
                cache[var] = self._era5_at_site(var)
            return cache[var]

        precip_gated = None
        if any(v.startswith("precip_api_tau") for v in met_vars):
            precip_raw = self.ds_cropped[self.precip_var].sel(datetime_utc=self.time).values.astype(np.float32)
            precip_gated = apply_snow_gating(precip_raw, base("sd"))

        met_list = []
        for v in met_vars:
            if v in self.TEMPERATURE_VARS:
                arr = to_celsius(base(v))
            elif v in ("sd", "swvl1_era5"):
                arr = base(v)
            elif v.startswith("precip_api_tau") and v.endswith("h"):
                tau = float(v[len("precip_api_tau"):-1])
                arr = compute_api(precip_gated, dt_hours, tau) # np.log1p()
            elif v.startswith("swvl1_roll_mean_") and v.endswith("h"):
                h = float(v[len("swvl1_roll_mean_"):-1])
                window_steps = max(1, int(round(h / dt_hours)))
                arr = compute_rolling_stat(base("swvl1_era5"), window_steps, "mean")
            elif v.startswith("swvl1_roll_min_") and v.endswith("h"):
                h = float(v[len("swvl1_roll_min_"):-1])
                window_steps = max(1, int(round(h / dt_hours)))
                arr = compute_rolling_stat(base("swvl1_era5"), window_steps, "min")
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
        if "ffp_footprint" in self.ds_cropped.variables:
            self.fp_array = self.ds_cropped["ffp_footprint"].sel(t=self.time).values.astype(np.float32)
        else:
            self.fp_array = None
        self.mask_static = self.ds_cropped["flux_mask"].values.astype(np.float32)
        self.static_stack = np.concatenate([self.nirv_max, self.nirv_min, self.lc], axis=-1)
        self.y_pack = np.stack([self.y_array, self.y_unc_array], axis=-1).astype(np.float32)

    # No __getitem__ override: BatchGenerator.__getitem__ returns
    # (sat, static, met, sw_in_pot, fp, mask), ypack -- the v1 signature.


# =============================================================================
# Model
# =============================================================================

class pyvprnn_v2(pyvprnn_v1):
    """
    pyvprnn_v1 plus time-sensitive water inputs, without a temperature lag
    encoder:

      HEAT (soil depth)     -- stl1_era5 (0-7 cm) and stl2_era5 (7-28 cm)
                               for soil respiration, t2m for aboveground
                               respiration. ERA5 soil temperature represents
                               heat diffusion into the soil directly.

      PULSE ONSET/DECAY     -- snow-gated precipitation API from precip_var
                               at API_TAUS_HOURS (Reco branch).

      DROUGHT MEMORY        -- swvl1_era5 rolling mean/min at
                               SWVL_ROLLING_HOURS (Reco branch).

      GPP WATER SUPPLY      -- swvl1_era5 (shallow roots, responds to rain
                               within hours to days) and swvl2_era5 (deeper
                               roots).

    Generator and network use the same 6-input signature as pyvprnn_v1.
    """

    API_TAUS_HOURS = (6, 24, 72, 168)             # as in pyvprnn_v3
    SWVL_ROLLING_HOURS = (168, 720, 2160, 3600)   # as in pyvprnn_v3
    DEFAULT_PRECIP_VAR = "P_F"  # tower measurement; swap for a gridded var at upscale time

    # Used by utils.make_generator (training AND inference) so that the
    # engineered features are always computed the same way.
    generator_cls = EngineeredBatchGenerator

    def __init__(self, precip_var=None, **kwargs):
        super().__init__(**kwargs)

        self.sat_vars = ["lswi", "nirv"]
        self.precip_var = precip_var if precip_var is not None else self.DEFAULT_PRECIP_VAR
        self.generator_kwargs = {"precip_var": self.precip_var}

        api_vars = [f"precip_api_tau{h}h" for h in self.API_TAUS_HOURS]
        swvl_roll_vars = []
        for h in self.SWVL_ROLLING_HOURS:
            swvl_roll_vars += [f"swvl1_roll_mean_{h}h", f"swvl1_roll_min_{h}h"]

        self.met_vars = (
            ["t2m", "ssrd", "RH_from_VDP", "swvl1_era5", "swvl2_era5", "sd",
             "stl1_era5", "stl2_era5"]
            + api_vars + swvl_roll_vars
        )

        self.gpp_met_vars = [
            "t2m", "ssrd", "RH_from_VDP", "swvl2_era5", "sd",
            "swvl1_roll_mean_168h", "swvl1_roll_min_720h",
            "precip_api_tau168h",
        ]

        self.reco_met_vars = (
            ["t2m", "swvl1_era5", "stl2_era5", "sd"]
            + api_vars + swvl_roll_vars
        )

        for name, subset in (("gpp_met_vars", self.gpp_met_vars),
                             ("reco_met_vars", self.reco_met_vars)):
            unknown = set(subset) - set(self.met_vars)
            if unknown:
                raise ValueError(f"{name} {sorted(unknown)} missing from met_vars.")

    def load_model(self, path):
        self.pixel_model = load_model(path, custom_objects={
            "BroadcastToImage": BroadcastToImage,
            "ExpandLastDim": ExpandLastDim,
            "GlobalSumPooling": GlobalSumPooling,
            "DayMask": DayMask,
            "SelectFeatures": SelectFeatures,
            "GPPPenalty": GPPPenalty,
            "TimeIntegratedRatioPenalty": TimeIntegratedRatioPenalty,
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
        wrong_nighttime_train = ((self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=train_times) < 0) &
                                 (self.ds_cropped["ssrd"].sel(datetime_utc=train_times) == 0))
        train_times_qc0 = train_times[(qc_train == 0) & ~wrong_nighttime_train]

        val_times = self.cv_folds[cv_fold]["val_times"]
        qc_val = self.ds_cropped["NEE_VUT_REF_QC"].sel(datetime_utc=val_times)
        wrong_nighttime_val = ((self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=val_times) < 0) &
                               (self.ds_cropped["ssrd"].sel(datetime_utc=val_times) == 0))
        val_times_qc0 = val_times[(qc_val == 0) & ~wrong_nighttime_val]

        train_gen = self.generator_cls(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            times=train_times_qc0,
            workers=train_params["workers"],
            use_multiprocessing=train_params["multiprocessing"],
            max_queue_size=train_params["max_queue_size"],
            target=target, unc=target_unc,
            land_cover_classes=self.land_cover_classes,
            **self.generator_kwargs,
        )
        val_gen = self.generator_cls(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            times=val_times_qc0,
            shuffle=False,
            target=target, unc=target_unc,
            land_cover_classes=self.land_cover_classes,
            **self.generator_kwargs,
        )

        (Xsat_batch, Xstatic_batch, Xmet_batch, _, _, _), _ = train_gen[0]

        n_sat_features = Xsat_batch.shape[-1]
        n_static_features = Xstatic_batch.shape[-1]
        n_met_features = Xmet_batch.shape[-1]

        gpp_met_idx = [self.met_vars.index(v) for v in self.gpp_met_vars]
        reco_met_idx = [self.met_vars.index(v) for v in self.reco_met_vars]
        filter_size = 1

        # =========================================================
        # Inputs (same signature as pyvprnn_v1)
        # =========================================================
        self.sat_input = layers.Input(shape=(None, None, n_sat_features), name="sat")
        self.static_input = layers.Input(shape=(None, None, n_static_features), name="static")
        self.met_input = layers.Input(shape=(None, None, n_met_features), name="met")
        self.sw_in_pot_input = layers.Input(shape=(None, None, 1), name="sw_in_pot")
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
        # GPP branch
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
        # Reco branch
        # =========================================================
        x_reco = layers.Concatenate(name="reco_concat")([self.sat_input, self.static_input, met_bc_reco])
        for i in range(6):
            x_reco = layers.Conv2D(32, filter_size, padding="same", activation="softplus",
                                   kernel_initializer="he_normal")(x_reco)
        x_reco_map = layers.Conv2D(
            1, 1, activation="softplus", kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.7), name="x_reco_map",
        )(x_reco)
        reco_map = layers.Multiply(name="reco_map")([x_reco_map, flux_mask_exp])

        # ---- Reco-to-GPP constraint: unchanged from pyvprnn_v1/v3. ----
        reco_map = TimeIntegratedRatioPenalty(
            max_ratio=1.0,
            weight=1e-4,
            gpp_floor=1.0,
            name="time_integrated_ratio_penalty",
        )([gpp_map, reco_map, day_mask])

        reco_weighted = layers.Multiply(name="reco_weighted")([reco_map, fp_exp])
        reco_sum = GlobalSumPooling(name="reco_sum")(reco_weighted)

        # =========================================================
        # NEE
        # =========================================================
        nee = layers.Subtract(name="nee")([reco_sum, gpp_sum])
        nee = layers.Flatten(name="Output")(nee)

        self.model = Model(
            inputs=[self.sat_input, self.static_input, self.met_input, self.sw_in_pot_input,
                    self.fp_input, self.flux_mask_input],
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
                    self.flux_mask_input],
            outputs=[self.model.get_layer("gpp_map").output, self.model.get_layer("reco_map").output],
            name="pixel_flux_model",
        )
        self.pixel_model.save(save_path_model)