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
#     lagged-met input -> generator and network have the same 6-input
#     signature as pyvprnn_v1.
#   - No air-soil temperature gradient feature (t2m - stl2), which mainly
#     acted as a day/night switch for the Reco branch.
#   - Reco branch: t2m (aboveground autotrophic respiration) and stl2_era5
#     (soil respiration). stl1_era5 and RH are not used in the Reco branch:
#     both acted as day/night or dryness proxies whose daytime values lie
#     outside the nighttime support.
#   - Precipitation APIs enter through a saturating transform
#     K (1 - exp(-API/K)), K = API_SAT_MM = 30 mm: linear for small events (pulse
#     size vs. rain amount is learned from the data), bounded for monsoon-season
#     API, where raw values (>100 mm, <1 % of the data) were extrapolated.
#   - GPP branch receives antecedent moisture (precip_api_tau168h,
#     swvl1_roll_mean_168h, swvl1_roll_min_720h) for post-rain activation of
#     light-use efficiency.
#
# NEW in this version:
#   - GPP = SW_in x LUE(...)  (USE_LUE_LIGHT). The network output is a light-use
#     efficiency; SW_in (ssrd) remains an input of the LUE network, so the
#     shape of the light response is still learned. Multiplying by SW_in
#     enforces GPP = 0 without light, which removes the free GPP offset that
#     otherwise trades off with daytime Reco (NEE = Reco - GPP is invariant
#     under Reco + c, GPP + c).
#   - Soft light-saturation penalty on LUE (LUE_MONO_WEIGHT): the LUE network is
#     evaluated again with SW_in raised by LUE_MONO_DELTA_SW, at the observed SW
#     and/or at a random SW per sample (LUE_MONO_MODE; random covers light levels
#     a state was never observed at); an INcrease of LUE with light is penalised
#     (daytime states only). Physiologically,
#     LUE falls monotonically with light (light saturation); this removes
#     convex light responses in sparsely sampled high-light states.
#   - Temperature memory input t2m_ema_<H>h (USE_T_MEMORY, T_MEMORY_HOURS):
#     exponential moving average of t2m (e-folding H hours) in the GPP branch
#     (optionally also in Reco). Separates thermal acclimation / spring
#     recovery of photosynthetic capacity from the instantaneous temperature
#     response; ICE curves that vary t2m keep the memory fixed.
#   - Soft temperature-monotonicity penalty on Reco w.r.t. t2m
#     (MONO_TEMP_VARS). For each listed temperature, the Reco branch is
#     evaluated a second time with only that temperature raised by
#     MONO_DELTA_T; a decrease of Reco is penalised. The penalty fades out
#     above MONO_T_CRIT, so declines at extreme heat (and unimodal responses
#     with an optimum above MONO_T_CRIT) remain free. Apparent unimodality
#     caused by hot = dry is still representable through the moisture inputs.
#
# pyvprnn_v2_pub (publication configuration, class at the end of this file).
# All options are constructor arguments of pyvprnn_v2 with the previous behaviour
# as default, so models trained with pyvprnn_v2 load and run unchanged;
# pyvprnn_v2_pub only switches the defaults:
#   Training data and loss
#   - sigma = NEE_VUT_REF_RANDUNC instead of JOINTUNC (TARGET_UNC). The u*
#     component of JOINTUNC is systematic and mainly large at night; as a
#     per-point sigma it down-weights the night-time data that anchor Reco.
#   - Night-time NEE is no longer truncated at 0 (NEE < 0 & ssrd == 0 removed
#     all negative noise realisations and biased night-time Reco high). Only
#     night-time outliers NEE < -NIGHT_OUTLIER_SIGMA * sigma are removed; night
#     from the FLUXNET NIGHT flag (SW_IN_POT based, as the model's day mask).
#   - QC flag taken from the target (f"{target}_QC"), so u*-percentile targets
#     (NEE_VUT_16, ...) use their own QC.
#   - Laplace NLL (LOSS), sigma floor 0.5 as before.
#   - Daytime samples with gap-filled radiation (SW_IN_F_QC != 0) are not used
#     for training (REQUIRE_SW_QC0).
#   Air temperature
#   - "t2m" is the tower air temperature (ds["t2m"], TA_F) instead of t2m_era5
#     (T2M_SOURCE = "tower"); also for t2m_ema and the monotonicity penalty. RH_from_VDP
#     is computed from tower VPD and tower t2m, so RH and t2m now form a consistent pair.
#   Numerical safeguards for weakly constrained footprint pixels
#   - TimeIntegratedRatioPenalty off (USE_RATIO_PENALTY): a Reco/GPP ratio is
#     violated by every dormant period; replaced by an absolute soft cap on
#     Reco (RECO_PENALTY_THRESHOLD = 50 umol m-2 s-1). GPP soft cap at 80
#     (GPP_PENALTY_THRESHOLD). Both lie well outside the observed range.
#   GPP branch
#   - Hard LUE bound: LUE = LUE_MAX * sigmoid(z), LUE_MAX = 200 umol m-2 s-1 per
#     kW m-2 (quantum yield ~0.09 x ~2.2 umol photons per J). Removes the
#     divergence LUE ~ delta/SW for SW -> 0 by which GPP can mimic a Reco offset.
#   - Clearness index k = clip(SW_IN_F / SW_IN_POT, 0, 1.5) as an input
#     (USE_CLEARNESS; "clearness_k"): sky condition / diffuse light.
#   Reco branch
#   - Fewer moisture channels: swvl1, the four APIs, swvl1_roll_mean_2160h and
#     swvl1_roll_min_3600h (RECO_SWVL_ROLL); LSWI not used (RECO_SAT_VARS).
#   Training stability
#   - Met inputs standardised inside the branches (NORMALIZE_INPUTS; mean/std of
#     the training samples, stored in the model). Without it the LUE output
#     saturated early and stayed at 0 in 2-5 of 8 seeds on synthetic data,
#     also for the previous configuration; with it in 0 of 8.
#   - Temperature monotonicity on t2m and stl2_era5 with variable-specific fade-
#     out (MONO_T_CRIT = {"t2m": 40, "stl2_era5": 30} degC); unimodal intrinsic
#     responses above these temperatures and apparent unimodality through the
#     moisture inputs remain free. Full weight per variable (MONO_SPLIT_WEIGHT).
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
    (= water content of a linear reservoir with residence time tau.)
    """
    api = np.empty(len(precip_values), dtype=np.float32)
    decay = float(np.exp(-dt_hours / tau_hours))
    running = 0.0
    for i, p in enumerate(precip_values):
        running = running * decay + float(np.nan_to_num(p))
        api[i] = running
    return api


# Saturating API: API_sat = K * (1 - exp(-API / K)).  Linear for API << K (small
# events keep their amount dependence), bounded at K (rain beyond topsoil storage
# carries no extra information; avoids extrapolating the raw tail of monsoon
# seasons). None = raw API (previous behaviour).
API_SAT_MM = 30.0


def saturate_api(api, k_mm, tau_hours=None):
    """k_mm: float (same K for all time scales), dict {tau_hours: K} (per time
    scale; missing keys -> raw), or None (raw API)."""
    if isinstance(k_mm, dict):
        k_mm = k_mm.get(int(round(tau_hours))) if tau_hours is not None else None
    if k_mm is None:
        return api
    return (k_mm * (1.0 - np.exp(-api / k_mm))).astype(np.float32)


def compute_ema(values, dt_hours, tau_hours):
    """
    Exponential moving average with e-folding time tau_hours (causal):
    m_t = m_{t-1} + (1 - exp(-dt/tau)) * (x_t - m_{t-1}),  m_0 = x_0.
    NaNs are skipped (the average is carried forward).
    """
    x = np.asarray(values, dtype=np.float64)
    a = 1.0 - float(np.exp(-dt_hours / tau_hours))
    out = np.empty(len(x), dtype=np.float32)
    finite = np.isfinite(x)
    m = float(x[finite][0]) if finite.any() else 0.0
    for i, v in enumerate(x):
        if np.isfinite(v):
            m += a * (v - m)
        out[i] = m
    return out


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
# Custom layers (new in this version)
# =============================================================================

class LightUseEfficiencyGPP(layers.Layer):
    """
    GPP = max(SW_in, 0) * LUE * scale.

    Inputs: [lue, sw]
      lue : non-negative network output (B, H, W, 1), a light-use efficiency
      sw  : incoming shortwave radiation in the network's input units
            (B, H, W, 1); ssrd enters scaled to kW m-2 (range ~0..1.1)

    `scale` only sets the order of magnitude of the network output at
    initialisation (GPP ~ 10 umol m-2 s-1 per kW m-2 for LUE ~ 1); it does not
    restrict the learned light response.
    With a sigmoid LUE output (lue in (0, 1), lue_max option of pyvprnn_v2),
    `scale` is the hard upper bound of the light-use efficiency
    (umol m-2 s-1 per kW m-2).
    """

    def __init__(self, scale=10.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        lue, sw = inputs
        return lue * tf.nn.relu(sw) * self.scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update(scale=self.scale)
        return cfg


class ShiftFeatures(layers.Layer):
    """Adds a constant `delta` to selected feature channels (last axis)."""

    def __init__(self, indices, n_features, delta, **kwargs):
        super().__init__(**kwargs)
        self.indices = [int(i) for i in indices]
        self.n_features = int(n_features)
        self.delta = float(delta)
        shift = np.zeros(self.n_features, dtype=np.float32)
        shift[self.indices] = self.delta
        self._shift = tf.constant(shift)

    def call(self, x):
        return x + self._shift

    def get_config(self):
        cfg = super().get_config()
        cfg.update(indices=self.indices, n_features=self.n_features, delta=self.delta)
        return cfg


class TemperatureMonotonicityPenalty(layers.Layer):
    """
    Soft penalty against Reco DEcreasing when one temperature input rises.

    Inputs: [reco, reco_warm, mask, temp]
      reco, reco_warm : Reco maps for the original inputs and for the
                        temperature input shifted by +delta_T (B, H, W, 1)
      mask            : flux mask (B, H, W, 1)
      temp            : the shifted temperature in degC (B, 1, 1, 1), used for
                        the taper only

    Adds  weight * < w(T) * ReLU(ln(reco) - ln(reco_warm)) >_mask  to the loss
    and returns `reco` unchanged.

    w(T) = sigmoid((t_crit - T) / t_width): the penalty fades out above t_crit,
    so a decline at extreme heat remains free. t_crit=None -> w(T) = 1.
    """

    def __init__(self, weight=1.0, t_crit=45.0, t_width=2.0, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.t_crit = None if t_crit is None else float(t_crit)
        self.t_width = float(t_width)
        self.eps = float(eps)

    def call(self, inputs):
        reco, reco_warm, mask, temp = inputs
        drop = tf.nn.relu(tf.math.log(reco + self.eps) - tf.math.log(reco_warm + self.eps))
        if self.t_crit is not None:
            drop = drop * tf.sigmoid((self.t_crit - temp) / self.t_width)
        penalty = tf.reduce_sum(drop * mask) / (tf.reduce_sum(mask) + 1e-6)
        self.add_loss(self.weight * penalty)
        return reco

    def get_config(self):
        cfg = super().get_config()
        cfg.update(weight=self.weight, t_crit=self.t_crit, t_width=self.t_width, eps=self.eps)
        return cfg


class SetFeatureRandom(layers.Layer):
    """Replaces one feature channel by a uniform random value in [low, high],
    drawn per sample (same value for all pixels of a sample)."""

    def __init__(self, index, n_features, low=0.0, high=1.1, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)
        self.n_features = int(n_features)
        self.low = float(low)
        self.high = float(high)
        onehot = np.zeros(self.n_features, dtype=np.float32)
        onehot[self.index] = 1.0
        self._onehot = tf.constant(onehot)

    def call(self, x):
        shape = tf.concat([tf.shape(x)[:1], tf.ones([tf.rank(x) - 1], tf.int32)], axis=0)
        u = tf.random.uniform(shape, self.low, self.high, dtype=x.dtype)
        return x * (1.0 - self._onehot) + u * self._onehot

    def get_config(self):
        cfg = super().get_config()
        cfg.update(index=self.index, n_features=self.n_features, low=self.low, high=self.high)
        return cfg


class Standardize(layers.Layer):
    """(x - mean) / std with fixed per-feature constants (last axis), stored in
    the layer config. Used inside the GPP/Reco branches, after the penalty
    shifts, so that the shifts (delta SW, delta T) stay in physical units."""

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = [float(v) for v in np.ravel(mean)]
        self.std = [float(v) for v in np.ravel(std)]
        self._m = tf.constant(np.asarray(self.mean, dtype=np.float32))
        self._s = tf.constant(np.asarray(self.std, dtype=np.float32))

    def call(self, x):
        return (x - self._m) / self._s

    def get_config(self):
        cfg = super().get_config()
        cfg.update(mean=self.mean, std=self.std)
        return cfg


class LightSaturationPenalty(layers.Layer):
    """
    Soft penalty against LUE INcreasing with light.

    Inputs: [lue, lue_bright, mask]
      lue, lue_bright : LUE maps for the original inputs and for SW_in raised by
                        delta_SW (B, H, W, 1)
      mask            : daytime x flux mask (B, H, W, 1)

    Adds  weight * < ReLU(ln(lue_bright) - ln(lue)) >_mask  to the loss and
    returns `lue` unchanged. Relative (log) differences, as for the Reco
    temperature penalty, so the strength does not depend on the LUE level.
    """

    def __init__(self, weight=1.0, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.eps = float(eps)

    def call(self, inputs):
        # [lue, lue_bright, mask]            -> returns lue
        # [out, lue, lue_bright, mask]       -> penalty on (lue, lue_bright), returns out
        if len(inputs) == 4:
            out, lue, lue_bright, mask = inputs
        else:
            lue, lue_bright, mask = inputs
            out = lue
        rise = tf.nn.relu(tf.math.log(lue_bright + self.eps) - tf.math.log(lue + self.eps))
        penalty = tf.reduce_sum(rise * mask) / (tf.reduce_sum(mask) + 1e-6)
        self.add_loss(self.weight * penalty)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(weight=self.weight, eps=self.eps)
        return cfg


CUSTOM_OBJECTS = {
    "BroadcastToImage": BroadcastToImage,
    "ExpandLastDim": ExpandLastDim,
    "GlobalSumPooling": GlobalSumPooling,
    "DayMask": DayMask,
    "SelectFeatures": SelectFeatures,
    "GPPPenalty": GPPPenalty,
    "TimeIntegratedRatioPenalty": TimeIntegratedRatioPenalty,
    "LightUseEfficiencyGPP": LightUseEfficiencyGPP,
    "ShiftFeatures": ShiftFeatures,
    "TemperatureMonotonicityPenalty": TemperatureMonotonicityPenalty,
    "LightSaturationPenalty": LightSaturationPenalty,
    "SetFeatureRandom": SetFeatureRandom,
    "Standardize": Standardize,
}


# =============================================================================
# Batch generator
# =============================================================================

class EngineeredBatchGenerator(BatchGenerator):
    """
    BatchGenerator plus engineered met features. No lag history, i.e. the
    generator returns the same (sat, static, met, sw_in_pot, fp, mask), y
    tuple as pyvprnn_v1's BatchGenerator.

    Met-variable names recognized on top of BatchGenerator:
      - "t2m"                    : air temperature; t2m_source="era5" -> t2m_era5
                                    (previous behaviour), "tower" -> ds["t2m"]
                                    (tower TA_F). Also feeds t2m_ema_<H>h.
      - "sd"                     : ERA5 snow depth (stored as sd_era5).
      - "stl1_era5", "stl2_era5" : ERA5 soil temperature, layers 1 and 2.
      - "swvl1_era5"             : ERA5 soil moisture, layer 1 (raw).
      - "precip_api_tau<H>h"     : snow-gated Antecedent Precipitation Index
                                    at timescale H hours from self.precip_var
                                    (raw, mm).
      - "swvl1_roll_mean_<H>h"   : rolling mean of swvl1_era5 over H hours.
      - "swvl1_roll_min_<H>h"    : rolling min of swvl1_era5 over H hours.
      - "t2m_ema_<H>h"           : exponential moving average of t2m (degC),
                                    e-folding time H hours (temperature memory).
      - "clearness_k"            : clearness index clip(SW / SW_IN_POT, 0, K_MAX)
                                    from CLEARNESS_SW_VAR (= SW_IN_F, W m-2) and
                                    SW_IN_POT; 0 at night.
    Everything else is handled exactly as in v3 (apply_met_scaling).

    precip_var (default "P_F") is the single place that decides which
    precipitation source feeds the API features - swap it at upscale time.
    """

    ERA5_KEY_OVERRIDES = {"t2m": "t2m_era5", "sd": "sd_era5"}
    TEMPERATURE_VARS = ("t2m", "stl1_era5", "stl2_era5")
    CLEARNESS_SW_VAR = "ssrd"     # = SW_IN_F in the site files (W m-2, unscaled here)
    CLEARNESS_K_MAX = 1.5

    TOWER_T2M_VAR = "t2m"         # tower air temperature (TA_F) in the site files

    def __init__(self, *args, precip_var="P_F", api_sat_mm=API_SAT_MM, t2m_source="era5", **kwargs):
        if t2m_source not in ("era5", "tower"):
            raise ValueError(f"t2m_source must be 'era5' or 'tower', got {t2m_source!r}")
        self.precip_var = precip_var
        self.api_sat_mm = api_sat_mm
        self.t2m_source = t2m_source
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
                if var == "t2m" and self.t2m_source == "tower":
                    cache[var] = (self.ds_cropped[self.TOWER_T2M_VAR]
                                  .sel(datetime_utc=self.time).values.astype(np.float32))
                else:
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
                arr = saturate_api(compute_api(precip_gated, dt_hours, tau), self.api_sat_mm, tau)
            elif v == "clearness_k":
                sw_raw = self.ds_cropped[self.CLEARNESS_SW_VAR].sel(datetime_utc=self.time).values.astype(np.float64)
                pot = self.ds_cropped["SW_IN_POT"].sel(datetime_utc=self.time).values.astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    k = np.where(pot > 0, sw_raw / pot, 0.0)
                arr = np.clip(np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0), 0.0, self.CLEARNESS_K_MAX)
            elif v.startswith("t2m_ema_") and v.endswith("h"):
                h = float(v[len("t2m_ema_"):-1])
                arr = compute_ema(to_celsius(base("t2m")), dt_hours, h)
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

      HEAT                  -- t2m (aboveground autotrophic respiration) and
                               stl2_era5 (soil respiration) in the Reco branch.

      PULSE ONSET/DECAY     -- snow-gated precipitation API from precip_var
                               at API_TAUS_HOURS (Reco branch).

      DROUGHT MEMORY        -- swvl1_era5 rolling mean/min at
                               SWVL_ROLLING_HOURS (Reco branch).

      GPP                   -- GPP = SW_in x LUE(...), LUE learned from
                               radiation, temperature, humidity, moisture
                               history and satellite indices (USE_LUE_LIGHT).

      Reco PRIOR            -- soft penalty against Reco decreasing with t2m
                               below MONO_T_CRIT (MONO_TEMP_VARS, MONO_WEIGHT).

    Generator and network use the same 6-input signature as pyvprnn_v1.
    """

    API_TAUS_HOURS = (6, 24, 72, 168)             # as in pyvprnn_v3
    SWVL_ROLLING_HOURS = (168, 720, 2160, 3600)   # as in pyvprnn_v3
    DEFAULT_PRECIP_VAR = "P_F"  # tower measurement; swap for a gridded var at upscale time

    # GPP = SW_in x LUE
    USE_LUE_LIGHT = True
    LIGHT_VAR = "ssrd"          # must be in gpp_met_vars; enters in kW m-2
    LUE_SCALE = 10.0            # output scale only (see LightUseEfficiencyGPP)

    # Soft light-saturation penalty on LUE (dLUE/dSW <= 0)
    LUE_MONO_WEIGHT = 3.0       # 0 switches it off
    LUE_MONO_DELTA_SW = 0.05    # kW m-2 (= 50 W m-2), network units of ssrd
    # where the penalty is evaluated: "observed" (SW of the sample), "random" (SW drawn
    # uniformly from LUE_MONO_SW_RANGE per sample, i.e. also light levels the state was
    # never observed at), or "both"
    LUE_MONO_MODE = "both"
    LUE_MONO_SW_RANGE = (0.0, 1.1)  # kW m-2

    # Temperature memory (acclimation) input
    USE_T_MEMORY = True
    T_MEMORY_HOURS = 168        # e-folding time of the t2m EMA (7 days)
    T_MEMORY_IN_RECO = False    # also feed it to the Reco branch

    # Soft temperature-monotonicity penalty on Reco
    MONO_TEMP_VARS = ("t2m",)   # one extra Reco pass per listed variable
    MONO_DELTA_T = 1.0          # K (temperature inputs are in degC)
    MONO_WEIGHT = 1.0           # 0 switches the penalty off; scan e.g. 0.3 / 1 / 3
    MONO_T_CRIT = 45.0          # degC; the penalty fades out above this temperature (float or {var: degC})
    MONO_T_CRIT_FALLBACK = 45.0 # for variables missing from a MONO_T_CRIT dict
    MONO_T_WIDTH = 2.0          # K, width of the fade-out

    MONO_SPLIT_WEIGHT = True    # weight / n_vars per variable (True) or full weight each (False)

    # Training data and loss (only act during training)
    TARGET_UNC = "NEE_VUT_REF_JOINTUNC"
    LOSS = "nll_loss_from_stacked"
    SIGMA_FLOOR = 0.5           # umol m-2 s-1, as in the loss functions
    NIGHT_OUTLIER_SIGMA = None  # None: legacy (drop all NEE < 0 at ssrd == 0); float k: drop night NEE < -k sigma
    REQUIRE_SW_QC0 = False      # drop daytime samples with gap-filled radiation (SW_IN_F_QC != 0)

    # Numerical safeguards (soft caps; weight 1e-4)
    USE_RATIO_PENALTY = True    # TimeIntegratedRatioPenalty (Reco / GPP <= 1 over the batch, daytime)
    GPP_PENALTY_THRESHOLD = 40.0
    RECO_PENALTY_THRESHOLD = None   # None: no absolute Reco cap
    PENALTY_WEIGHT = 1e-4

    # Hard LUE bound (None: softplus output x LUE_SCALE, i.e. unbounded)
    LUE_MAX = None              # umol m-2 s-1 per kW m-2
    LUE_INIT = 25.0             # initial LUE with LUE_MAX (sets the output bias)

    # Clearness index input in the GPP branch
    USE_CLEARNESS = False

    # Source of the air temperature input "t2m" (also t2m_ema and the Reco
    # monotonicity penalty): "era5" (t2m_era5, previous behaviour) or "tower" (TA_F)
    T2M_SOURCE = "era5"

    # Standardise the met inputs inside the branches (mean/std of the training
    # samples, stored in the model). Without it, inputs of very different scale
    # (RH in %, temperatures in degC, API in mm next to swvl ~0.3) give large
    # hidden activations, and the LUE output can saturate early in training and
    # stay at 0 (observed on synthetic data for the previous configuration too).
    NORMALIZE_INPUTS = False

    # Reco branch inputs
    RECO_SWVL_ROLL = None       # None: all swvl1 rolling stats; else e.g. (("mean", 2160), ("min", 3600))
    RECO_SAT_VARS = None        # None: all sat_vars; else e.g. ("nirv",)

    # Used by utils.make_generator (training AND inference) so that the
    # engineered features are always computed the same way.
    generator_cls = EngineeredBatchGenerator

    def __init__(self, precip_var=None, use_lue_light=None, mono_weight=None,
                 mono_t_crit="default", mono_temp_vars=None, api_sat_mm="default",
                 lue_mono_weight=None, use_t_memory=None, t_memory_hours=None,
                 t_memory_in_reco=None,
                 target_unc=None, loss=None, night_outlier_sigma="default", require_sw_qc0=None,
                 use_ratio_penalty=None, gpp_penalty_threshold=None, reco_penalty_threshold="default",
                 lue_max="default", lue_init=None, use_clearness=None,
                 reco_swvl_roll="default", reco_sat_vars="default", mono_split_weight=None,
                 normalize_inputs=None, t2m_source=None, **kwargs):
        """
        Old models (trained before these options existed) need the matching
        inputs at inference, e.g.
            pyvprnn_v2(api_sat_mm=None, use_t_memory=False)   # raw API, no memory
        lue_mono_weight only acts during training.

        Options that change the network inputs or structure (use_clearness,
        lue_max, reco_swvl_roll, reco_sat_vars, normalize_inputs, t2m_source)
        must match the trained model at inference. "default" / None take the class attribute; options whose
        value None is meaningful (night_outlier_sigma, reco_penalty_threshold,
        lue_max, reco_swvl_roll, reco_sat_vars) use "default" as sentinel.
        """
        super().__init__(**kwargs)
        pick = lambda v, attr, sentinel=None: getattr(self, attr) if v is sentinel else v
        self.target_unc = pick(target_unc, "TARGET_UNC")
        self.loss_name = pick(loss, "LOSS")
        self.night_outlier_sigma = pick(night_outlier_sigma, "NIGHT_OUTLIER_SIGMA", "default")
        self.require_sw_qc0 = bool(pick(require_sw_qc0, "REQUIRE_SW_QC0"))
        self.use_ratio_penalty = bool(pick(use_ratio_penalty, "USE_RATIO_PENALTY"))
        self.gpp_penalty_threshold = float(pick(gpp_penalty_threshold, "GPP_PENALTY_THRESHOLD"))
        self.reco_penalty_threshold = pick(reco_penalty_threshold, "RECO_PENALTY_THRESHOLD", "default")
        self.lue_max = pick(lue_max, "LUE_MAX", "default")
        self.lue_init = float(pick(lue_init, "LUE_INIT"))
        self.use_clearness = bool(pick(use_clearness, "USE_CLEARNESS"))
        self.reco_swvl_roll = pick(reco_swvl_roll, "RECO_SWVL_ROLL", "default")
        self.reco_sat_vars = pick(reco_sat_vars, "RECO_SAT_VARS", "default")
        self.mono_split_weight = bool(pick(mono_split_weight, "MONO_SPLIT_WEIGHT"))
        self.normalize_inputs = bool(pick(normalize_inputs, "NORMALIZE_INPUTS"))
        self.t2m_source = pick(t2m_source, "T2M_SOURCE")
        self._met_stats = None
        if self.lue_max is not None and not (0 < self.lue_init < self.lue_max):
            raise ValueError("lue_init must lie in (0, lue_max).")

        self.sat_vars = ["lswi", "nirv"]
        self.precip_var = precip_var if precip_var is not None else self.DEFAULT_PRECIP_VAR
        self.api_sat_mm = API_SAT_MM if api_sat_mm == "default" else api_sat_mm
        # generator_kwargs must reach every generator (training AND inference), see utils.make_generator
        self.generator_kwargs = {"precip_var": self.precip_var, "api_sat_mm": self.api_sat_mm,
                                 "t2m_source": self.t2m_source}

        self.use_lue_light = self.USE_LUE_LIGHT if use_lue_light is None else bool(use_lue_light)
        self.mono_weight = self.MONO_WEIGHT if mono_weight is None else float(mono_weight)
        self.mono_t_crit = self.MONO_T_CRIT if mono_t_crit == "default" else mono_t_crit
        self.mono_temp_vars = tuple(self.MONO_TEMP_VARS if mono_temp_vars is None else mono_temp_vars)
        self.lue_mono_weight = self.LUE_MONO_WEIGHT if lue_mono_weight is None else float(lue_mono_weight)
        self.use_t_memory = self.USE_T_MEMORY if use_t_memory is None else bool(use_t_memory)
        self.t_memory_hours = self.T_MEMORY_HOURS if t_memory_hours is None else int(t_memory_hours)
        self.t_memory_in_reco = self.T_MEMORY_IN_RECO if t_memory_in_reco is None else bool(t_memory_in_reco)
        t_mem_vars = [f"t2m_ema_{self.t_memory_hours}h"] if self.use_t_memory else []

        api_vars = [f"precip_api_tau{h}h" for h in self.API_TAUS_HOURS]
        swvl_roll_vars = []
        for h in self.SWVL_ROLLING_HOURS:
            swvl_roll_vars += [f"swvl1_roll_mean_{h}h", f"swvl1_roll_min_{h}h"]

        clear_vars = ["clearness_k"] if self.use_clearness else []

        self.met_vars = (
            ["t2m", "ssrd", "RH_from_VDP", "swvl1_era5", "swvl2_era5", "sd",
             "stl1_era5", "stl2_era5"]
            + api_vars + swvl_roll_vars + t_mem_vars + clear_vars
        )

        self.gpp_met_vars = [
            "t2m", "ssrd", "RH_from_VDP", "swvl2_era5", "sd",
            "swvl1_roll_mean_168h", "swvl1_roll_min_720h",
            "precip_api_tau168h",
        ] + t_mem_vars + clear_vars

        if self.reco_swvl_roll is None:
            reco_roll_vars = swvl_roll_vars
        else:
            reco_roll_vars = [f"swvl1_roll_{stat}_{int(h)}h" for stat, h in self.reco_swvl_roll]
        self.reco_met_vars = (
            ["t2m", "swvl1_era5", "stl2_era5", "sd"]
            + api_vars + reco_roll_vars
            + (t_mem_vars if self.t_memory_in_reco else [])
        )
        self.reco_sat_vars_ = list(self.sat_vars if self.reco_sat_vars is None else self.reco_sat_vars)
        unknown_sat = set(self.reco_sat_vars_) - set(self.sat_vars)
        if unknown_sat:
            raise ValueError(f"reco_sat_vars {sorted(unknown_sat)} missing from sat_vars.")

        for name, subset in (("gpp_met_vars", self.gpp_met_vars),
                             ("reco_met_vars", self.reco_met_vars)):
            unknown = set(subset) - set(self.met_vars)
            if unknown:
                raise ValueError(f"{name} {sorted(unknown)} missing from met_vars.")
        if self.use_lue_light and self.LIGHT_VAR not in self.met_vars:
            raise ValueError(f"LIGHT_VAR '{self.LIGHT_VAR}' missing from met_vars.")

    def load_model(self, path):
        self.pixel_model = load_model(path, custom_objects=CUSTOM_OBJECTS)

    # -------------------------------------------------------------------------
    # Network
    # -------------------------------------------------------------------------
    def _build_model(self, n_sat_features, n_static_features, n_met_features):
        """Builds self.model (NEE) and the layer graph used for self.pixel_model."""
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
        # GPP branch:  GPP = SW_in x LUE(sat, static, GPP met incl. SW_in)
        # =========================================================
        # layers created once, so the branch can be evaluated again for SW + delta_SW
        gpp_concat = layers.Concatenate(name="gpp_concat")
        gpp_hidden = [
            layers.Conv2D(32, filter_size, padding="same", activation="softplus",
                          kernel_initializer="he_normal", name=f"gpp_conv_{i}")
            for i in range(6)
        ]
        lue_bounded = self.use_lue_light and self.lue_max is not None
        if lue_bounded:
            # LUE = LUE_MAX * sigmoid(z); bias so that the initial LUE is LUE_INIT
            p0 = self.lue_init / self.lue_max
            out_act, out_bias = "sigmoid", float(np.log(p0 / (1.0 - p0)))
        else:
            out_act, out_bias = "softplus", 0.3
        gpp_out = layers.Conv2D(
            1, 1, activation=out_act, kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(out_bias),
            name="x_lue_map" if self.use_lue_light else "x_gpp_map",
        )

        if self.normalize_inputs:
            if self._met_stats is None:
                raise RuntimeError("normalize_inputs=True needs the met statistics (set in train()).")
            mu, sd = self._met_stats
            std_gpp = Standardize(mu[gpp_met_idx], sd[gpp_met_idx], name="std_met_gpp")
            std_reco = Standardize(mu[reco_met_idx], sd[reco_met_idx], name="std_met_reco")
        else:
            std_gpp = std_reco = (lambda t: t)

        def gpp_branch(met_bc):
            x = gpp_concat([self.sat_input, self.static_input, std_gpp(met_bc)])
            for layer in gpp_hidden:
                x = layer(x)
            return gpp_out(x)

        x_gpp = gpp_branch(met_bc_gpp)

        # ---- Soft light-saturation penalty: LUE must not increase with SW ----
        if self.use_lue_light and self.lue_mono_weight > 0:
            j_sw = self.gpp_met_vars.index(self.LIGHT_VAR)
            n_gpp = len(self.gpp_met_vars)
            lue_mask = layers.Multiply(name="lue_penalty_mask")([day_mask, flux_mask_exp])
            modes = ["observed", "random"] if self.LUE_MONO_MODE == "both" else [self.LUE_MONO_MODE]
            w = self.lue_mono_weight / len(modes)
            if "observed" in modes:
                met_bright = ShiftFeatures([j_sw], n_gpp, self.LUE_MONO_DELTA_SW,
                                           name="met_gpp_bright")(met_gpp)
                met_bc_bright = BroadcastToImage(name="met_bc_gpp_bright")([met_bright, sat_static])
                x_lue_bright = gpp_branch(met_bc_bright)
                x_gpp = LightSaturationPenalty(weight=w, name="lue_light_penalty")(
                    [x_gpp, x_lue_bright, lue_mask])
            if "random" in modes:
                lo, hi = self.LUE_MONO_SW_RANGE
                met_rand = SetFeatureRandom(j_sw, n_gpp, lo, hi - self.LUE_MONO_DELTA_SW,
                                            name="met_gpp_randlight")(met_gpp)
                met_rand_b = ShiftFeatures([j_sw], n_gpp, self.LUE_MONO_DELTA_SW,
                                           name="met_gpp_randlight_bright")(met_rand)
                x_lue_r = gpp_branch(BroadcastToImage(name="met_bc_gpp_randlight")([met_rand, sat_static]))
                x_lue_rb = gpp_branch(BroadcastToImage(name="met_bc_gpp_randlight_bright")([met_rand_b, sat_static]))
                x_gpp = LightSaturationPenalty(weight=w, name="lue_light_penalty_random")(
                    [x_gpp, x_lue_r, x_lue_rb, lue_mask])

        if self.use_lue_light:
            sw = SelectFeatures([self.met_vars.index(self.LIGHT_VAR)], name="met_sw_in")(self.met_input)
            sw_bc = BroadcastToImage(name="sw_in_bc")([sw, sat_static])
            scale = self.lue_max if lue_bounded else self.LUE_SCALE
            x_gpp = LightUseEfficiencyGPP(scale=scale, name="x_gpp_map")([x_gpp, sw_bc])
        x_gpp = GPPPenalty(threshold=self.gpp_penalty_threshold, weight=self.PENALTY_WEIGHT,
                           name="gpp_penalty")(x_gpp)

        gpp_map_day_mask = layers.Multiply(name="gpp_map_masked")([x_gpp, day_mask])
        gpp_map = layers.Multiply(name="gpp_map")([gpp_map_day_mask, flux_mask_exp])
        gpp_weighted = layers.Multiply(name="gpp_weighted")([gpp_map, fp_exp])
        gpp_sum = GlobalSumPooling(name="gpp_sum_raw")(gpp_weighted)

        # =========================================================
        # Reco branch (layers created once, so the branch can be
        # evaluated again for the temperature-shifted inputs)
        # =========================================================
        reco_concat = layers.Concatenate(name="reco_concat")
        reco_hidden = [
            layers.Conv2D(32, filter_size, padding="same", activation="softplus",
                          kernel_initializer="he_normal", name=f"reco_conv_{i}")
            for i in range(6)
        ]
        reco_out = layers.Conv2D(
            1, 1, activation="softplus", kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.7), name="x_reco_map",
        )

        if self.reco_sat_vars_ == list(self.sat_vars):
            sat_reco = self.sat_input
        else:
            sat_reco = SelectFeatures([self.sat_vars.index(v) for v in self.reco_sat_vars_],
                                      name="sat_reco")(self.sat_input)

        def reco_branch(met_bc):
            x = reco_concat([sat_reco, self.static_input, std_reco(met_bc)])
            for layer in reco_hidden:
                x = layer(x)
            return reco_out(x)

        x_reco_map = reco_branch(met_bc_reco)

        # ---- Soft temperature-monotonicity penalty (one pass per variable) ----
        mono_vars = [v for v in self.mono_temp_vars if v in self.reco_met_vars]
        if self.mono_weight > 0 and mono_vars:
            for v in mono_vars:
                j = self.reco_met_vars.index(v)
                met_warm = ShiftFeatures([j], len(self.reco_met_vars), self.MONO_DELTA_T,
                                         name=f"met_reco_warm_{v}")(met_reco)
                met_bc_warm = BroadcastToImage(name=f"met_bc_reco_warm_{v}")([met_warm, sat_static])
                x_reco_warm = reco_branch(met_bc_warm)
                temp_v = SelectFeatures([self.met_vars.index(v)], name=f"met_mono_temp_{v}")(self.met_input)
                if isinstance(self.mono_t_crit, dict):
                    t_crit_v = self.mono_t_crit.get(v, self.MONO_T_CRIT_FALLBACK)
                else:
                    t_crit_v = self.mono_t_crit
                w_v = self.mono_weight / len(mono_vars) if self.mono_split_weight else self.mono_weight
                x_reco_map = TemperatureMonotonicityPenalty(
                    weight=w_v, t_crit=t_crit_v, t_width=self.MONO_T_WIDTH,
                    name=f"temp_mono_penalty_{v}",
                )([x_reco_map, x_reco_warm, flux_mask_exp, temp_v])

        # ---- Absolute soft cap on Reco (numerical safeguard, off by default) ----
        if self.reco_penalty_threshold is not None:
            x_reco_map = GPPPenalty(threshold=float(self.reco_penalty_threshold), weight=self.PENALTY_WEIGHT,
                                    name="reco_penalty")(x_reco_map)

        reco_map = layers.Multiply(name="reco_map")([x_reco_map, flux_mask_exp])

        # ---- Reco-to-GPP constraint: unchanged from pyvprnn_v1/v3 (optional). ----
        if self.use_ratio_penalty:
            reco_map = TimeIntegratedRatioPenalty(
                max_ratio=1.0,
                weight=self.PENALTY_WEIGHT,
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

    def _build_pixel_model(self):
        self.pixel_model = Model(
            inputs=[self.sat_input, self.static_input, self.met_input, self.sw_in_pot_input,
                    self.flux_mask_input],
            outputs=[self.model.get_layer("gpp_map").output, self.model.get_layer("reco_map").output],
            name="pixel_flux_model",
        )

    # -------------------------------------------------------------------------
    # Training data selection
    # -------------------------------------------------------------------------
    def _select_training_times(self, times, target, target_unc, label=""):
        """QC = 0 of the target; night-time treatment (legacy truncation or
        NIGHT_OUTLIER_SIGMA); optionally daytime SW_IN_F_QC == 0."""
        ds = self.ds_cropped
        sel = lambda v: ds[v].sel(datetime_utc=times).values
        qc_var = f"{target}_QC" if f"{target}_QC" in ds else "NEE_VUT_REF_QC"
        y = sel(target).astype(np.float64)
        keep = sel(qc_var) == 0
        n_qc = int(keep.sum())
        if self.night_outlier_sigma is None:
            night_drop = (y < 0) & (sel("ssrd") == 0)              # legacy: all negative night-time NEE
            rule = "NEE < 0 at ssrd == 0"
        else:
            night = (sel("NIGHT") == 1) if "NIGHT" in ds else (sel("SW_IN_POT") <= 0)
            sigma = np.maximum(sel(target_unc).astype(np.float64), self.SIGMA_FLOOR)
            night_drop = night & (y < -float(self.night_outlier_sigma) * sigma)
            rule = f"night NEE < -{self.night_outlier_sigma:g} sigma"
        keep &= ~night_drop
        n_night = n_qc - int(keep.sum())
        n_sw = 0
        if self.require_sw_qc0 and "SW_IN_F_QC" in ds:
            before = int(keep.sum())
            keep &= ~((sel("SW_IN_POT") > 0) & (sel("SW_IN_F_QC") != 0))
            n_sw = before - int(keep.sum())
        print(f"[pyvprnn_v2] {label}: {n_qc} samples with {qc_var} == 0; removed {n_night} ({rule})"
              + (f", {n_sw} (daytime SW_IN_F_QC != 0)" if self.require_sw_qc0 else "")
              + f" -> {int(keep.sum())}")
        return times[keep]

    def describe(self):
        """Prints the configuration (useful in logs and for the methods section)."""
        items = dict(t2m_source=self.t2m_source, target_unc=self.target_unc, loss=self.loss_name, night_outlier_sigma=self.night_outlier_sigma,
                     require_sw_qc0=self.require_sw_qc0, use_ratio_penalty=self.use_ratio_penalty,
                     gpp_penalty_threshold=self.gpp_penalty_threshold,
                     reco_penalty_threshold=self.reco_penalty_threshold, lue_max=self.lue_max,
                     use_clearness=self.use_clearness, mono_temp_vars=self.mono_temp_vars,
                     mono_t_crit=self.mono_t_crit, mono_split_weight=self.mono_split_weight,
                     gpp_met_vars=self.gpp_met_vars, reco_met_vars=self.reco_met_vars,
                     reco_sat_vars=self.reco_sat_vars_, normalize_inputs=self.normalize_inputs)
        print(f"[{type(self).__name__}]")
        for k, v in items.items():
            print(f"  {k:24s} {v}")

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
                            "loss": None},
              target="NEE_VUT_REF",
              target_unc=None,
              cv_fold=0,
              random_state=41):
        """target_unc=None and train_params["loss"]=None (or missing) take the
        model's configuration (TARGET_UNC / LOSS, see describe())."""
        target_unc = self.target_unc if target_unc is None else target_unc
        loss_name = train_params.get("loss") or self.loss_name
        self.describe()

        train_times_qc0 = self._select_training_times(self.cv_folds[cv_fold]["train_times"],
                                                      target, target_unc, label="train")
        val_times_qc0 = self._select_training_times(self.cv_folds[cv_fold]["val_times"],
                                                    target, target_unc, label="val")

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

        if self.use_lue_light:
            sw_max = float(np.nanmax(train_gen.met_array[..., self.met_vars.index(self.LIGHT_VAR)]))
            print(f"[pyvprnn_v2] GPP = SW x LUE; max {self.LIGHT_VAR} input = {sw_max:.3f}"
                  + (f"; LUE <= {self.lue_max:g} per kW m-2" if self.lue_max is not None else ""))
            if sw_max > 5.0:
                print("[pyvprnn_v2] WARNING: light input looks like W m-2, not kW m-2; "
                      "adjust LUE_SCALE (e.g. 0.01) so the initial GPP is of the right order.")

        if self.normalize_inputs:
            met2d = train_gen.met_array.reshape(len(train_gen.met_array), -1)[train_gen.indexes]
            mu = np.nanmean(met2d, axis=0)
            sd = np.nanstd(met2d, axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)     # constant inputs (e.g. no snow)
            self._met_stats = (np.nan_to_num(mu), sd)

        self._build_model(Xsat_batch.shape[-1], Xstatic_batch.shape[-1], Xmet_batch.shape[-1])

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

        if loss_name == "nll_loss_from_stacked":
            loss = nll_loss_from_stacked
        elif loss_name == "nll_loss_laplace_from_stacked":
            loss = nll_loss_laplace_from_stacked
        elif loss_name == "mse":
            loss = mse_true_only
        else:
            raise NotImplementedError(f"Unknown loss '{loss_name}'")

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

        self._build_pixel_model()
        self.pixel_model.save(save_path_model)


# =============================================================================
# Publication configuration
# =============================================================================

class pyvprnn_v2_pub(pyvprnn_v2):
    """
    pyvprnn_v2 with the publication defaults (see the header of this file).
    Every option can still be switched per instance for ablations, e.g.
        pyvprnn_v2_pub(use_clearness=False, lue_max=None)
    """

    # training data and loss
    TARGET_UNC = "NEE_VUT_REF_RANDUNC"
    LOSS = "nll_loss_laplace_from_stacked"
    NIGHT_OUTLIER_SIGMA = 3.0
    REQUIRE_SW_QC0 = True

    # numerical safeguards
    USE_RATIO_PENALTY = False
    GPP_PENALTY_THRESHOLD = 80.0
    RECO_PENALTY_THRESHOLD = 50.0

    # GPP branch
    LUE_MAX = 200.0
    LUE_INIT = 25.0
    USE_CLEARNESS = True

    # training stability (see NORMALIZE_INPUTS in pyvprnn_v2)
    NORMALIZE_INPUTS = True

    # tower air temperature (consistent with RH_from_VDP, and with NT / DT)
    T2M_SOURCE = "tower"

    # Reco branch
    RECO_SWVL_ROLL = (("mean", 2160), ("min", 3600))
    RECO_SAT_VARS = ("nirv",)
    MONO_TEMP_VARS = ("t2m", "stl2_era5")
    MONO_T_CRIT = {"t2m": 40.0, "stl2_era5": 30.0}
    MONO_SPLIT_WEIGHT = False


# Ablation variants for the paper: (class, constructor kwargs)
#   A: previous model; An: A with standardised met inputs (training stability
#   only, same function class); B: + training data / loss / safeguards (steps 1-4);
#   C: + LUE bound and clearness index (steps 5-6); D: + Reco inputs and
#   temperature monotonicity (steps 8-9) = publication model.
_RECO_LEGACY = dict(reco_swvl_roll=None, reco_sat_vars=None, mono_temp_vars=("t2m",), mono_t_crit=45.0)
ABLATIONS = {
    "A": (pyvprnn_v2, {}),
    "An": (pyvprnn_v2, dict(normalize_inputs=True)),
    "B": (pyvprnn_v2_pub, dict(lue_max=None, use_clearness=False, require_sw_qc0=False, **_RECO_LEGACY)),
    "C": (pyvprnn_v2_pub, dict(**_RECO_LEGACY)),
    "D": (pyvprnn_v2_pub, {}),
}


def make_ablation(name, **kwargs):
    """make_ablation("C") -> model instance of ablation variant C (extra kwargs override)."""
    cls, kw = ABLATIONS[name]
    return cls(**{**kw, **kwargs})