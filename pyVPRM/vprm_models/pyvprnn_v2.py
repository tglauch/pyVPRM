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
# pyvprnn_v2 -- flux-partitioning model (one configuration; pyvprnn_v2_pub is an alias)
#
# NEE = Reco - GPP, both learned per footprint pixel and summed with the flux
# footprint. The class defaults are the model configuration; constructor options
# exist only to load models trained with earlier settings (see __init__).
#
# GPP branch
#   - GPP = SW_in x LUE(...): GPP = 0 without light, which removes the free GPP
#     offset that otherwise trades off with daytime Reco (NEE is invariant under
#     Reco + c, GPP + c). SW_in stays an input of the LUE network, so the shape of
#     the light response is learned.
#   - Hard LUE bound: LUE = LUE_MAX * sigmoid(z), LUE_MAX = 200 umol m-2 s-1 per
#     kW m-2 (quantum yield ~0.09 x ~2.2 umol photons per J).
#   - Soft light-saturation penalty: LUE must not increase with SW (evaluated at
#     the observed SW and at a random SW per sample, daytime only).
#   - Inputs: t2m (tower), SW_in, RH (from tower VPD and t2m), swvl2_era5, snow
#     depth, swvl1 rolling mean 168 h / min 720 h, precipitation API 168 h, t2m
#     memory (EMA, 168 h), clearness index k = SW_in / SW_IN_POT, LSWI, NIRv,
#     static inputs (NIRv percentiles; land cover enters through linear mixing, see below).
#   - Soft sign constraints on the GPP response at otherwise fixed inputs
#     (GPP_SIGN_CONSTRAINTS): NIRv (+), snow depth (-), soil moisture (+; current
#     value and rolling statistics shifted together), each with a relative
#     tolerance; flat responses are always allowed. Temperature, RH / VPD, clearness,
#     LSWI and rain history are unconstrained (the RH constraint is commented out in
#     GPP_SIGN_CONSTRAINTS: it has known exceptions, and the VPD response is learned).
#
# Land cover (LINEAR_MIXING = True)
#   - GPP and Reco branches have one output head per land-cover class on shared hidden layers;
#     the pixel flux is the area-weighted sum of the class fluxes (fluxes are additive over
#     area). The fractions are not a network input, so they cannot be combined nonlinearly
#     (e.g. as a pixel fingerprint). Fractions are normalised to sum 1 per pixel; pixels
#     without land-cover information get equal weights. The LUE bound holds per class and
#     therefore for the mixture; all penalties act on the mixed pixel values.
#
# Reco branch
#   - Inputs: t2m (tower; aboveground autotrophic respiration), stl2_era5 (soil
#     respiration), swvl1_era5, snow depth, precipitation APIs at 12/24/72/168 h,
#     swvl1 rolling mean 2160 h / min 3600 h, NIRv, static inputs. No light, no
#     RH and no stl1 (day/night proxies outside the night-time support).
#   - Soft temperature monotonicity: Reco must not decrease when t2m or stl2
#     rises, fading out above 40 degC (t2m) and 30 degC (stl2).
#
# Precipitation
#   - Snow-gated API from measured precipitation (precip_var = "P_F"; swap for a
#     gridded product at upscaling time), saturating transform
#     K (1 - exp(-API/K)), K = 30 mm.
#   - Shortest time scale 12 h (previous runs: 6 h). Every API jumps by the full
#     rain amount at rain onset; at open-path sites only ~10-20 % of the first
#     hours after rain are measured, and the 6 h scale produced unconstrained
#     Reco spikes there. 12 h keeps a fast scale for the pulse onset with less
#     dependence on hourly rain timing.
#
# Training
#   - Laplace NLL, sigma = NEE_VUT_REF_RANDUNC (floor 0.5). QC = 0 of the target;
#     night-time outliers NEE < -3 sigma removed (NIGHT flag); daytime samples with
#     gap-filled radiation (SW_IN_F_QC != 0) not used.
#   - Met inputs standardised inside the branches (training mean / std, stored in
#     the model).
#   - Numerical safeguards: soft caps GPP < 80 and Reco < 50 umol m-2 s-1.
#   - Diffusion bound from LE (DIFFUSION_CAP_WEIGHT = 1, training only, see diffusion_bound()):
#     footprint GPP must not exceed Gs Ca / 1.6 (Fick's law with Ci = 0), Gs from
#     Penman-Monteith on LE corrected for the energy balance gap plus 2 x LE_RANDUNC, on all
#     half-hours where it is measurable. Soft penalty on the excess; it only removes GPP that
#     is physically impossible within the measurement uncertainty of LE. Sites without LE /
#     LE_RANDUNC: no bound.
#
# Loading models from previous runs (same network, older inputs):
#     pyvprnn_v2(api_taus_hours=(6, 24, 72, 168))
# Training-only settings (sign constraints, penalties, loss) do not matter at
# inference.
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

# Land-cover fractions that do not sum to 1 (files from the pipeline version that dropped
# unsplit ESA "tree cover"): the missing weight is assigned to this class (1 = evergreen forest,
# the default of the fixed pipeline). None: no filling.
LC_FILL_CLASS = 1


# =============================================================================
# Diffusion bound on GPP from LE (Fick's law), training only
# =============================================================================
# GPP <= Gs * Ca / 1.6 (Ci = 0: no leaf can take up CO2 faster than the canopy conductance
# allows). Gs from inverting Penman-Monteith on LE; imposed form E P / D where wind / u* are
# missing. Conservative with respect to the measurement of LE:
#   - energy balance gap removed: median LE_CORR / LE per (month, solar hour); without LE_CORR
#     the site's closure ratio (H+LE)/(Rn-G) per (month, solar hour); LE is only ever increased
#   - LE + n_sigma * LE_RANDUNC
# Evaluated per half-hour wherever it is measurable: daytime (SW_IN_POT > 0), SW > sw_min,
# LE QC = 0, VPD > vpd_min (the inversion is unstable as VPD -> 0), no rain in rain_excl_h.
# LE <= 0 counts as Gs = 0. Identical to the notebook cell 5e-V ("closed + 2 sigma").
DIFFUSION_CAP_KW = dict(sw_min=200.0, vpd_min=0.5, rain_excl_h=24, p_min=0.2, n_sigma=2.0,
                        ca_default=415.0)


def _tower_series(ds, names, index):
    """First available variable of `names` as a float series on `index` (gridded fields: nearest
    valid cell); None if none is present."""
    for n in ([names] if isinstance(names, str) else names):
        if n not in ds.variables or "datetime_utc" not in ds[n].dims:
            continue
        v = ds[n]
        if len(v.dims) > 1:
            v = sel_nearest_valid(ds[[n]].compute(), lon=ds.attrs["site_lon"], lat=ds.attrs["site_lat"])[n]
        sr = pd.Series(np.asarray(v.values, dtype=np.float64), index=pd.DatetimeIndex(ds["datetime_utc"].values))
        sr = sr[~sr.index.duplicated()].reindex(index)
        return sr.mask(sr <= -9990)
    return None


def diffusion_bound(ds, verbose=True, **kw):
    """
    Half-hourly diffusion bound on site GPP [umol m-2 s-1] on ds.datetime_utc (NaN where it is
    not measurable), or None if LE / LE_RANDUNC / the meteorology are missing.
    """
    kw = {**DIFFUSION_CAP_KW, **kw}
    idx = pd.DatetimeIndex(ds["datetime_utc"].values)
    get = lambda names: _tower_series(ds, names, idx)
    le, h, unc = get(["LE_F_MDS", "LE"]), get(["H_F_MDS", "H"]), get(["LE_RANDUNC"])
    ta, vpd, sw = get(["t2m", "TA_F", "TA_F_MDS", "TA"]), get(["VPD_F", "VPD_F_MDS", "VPD", "VPD_ERA"]), \
        get(["ssrd", "SW_IN_F", "SW_IN_F_MDS", "SW_IN"])
    missing = [n for n, v in (("LE", le), ("H", h), ("LE_RANDUNC", unc), ("TA", ta), ("VPD", vpd), ("SW", sw))
               if v is None or v.isna().all()]
    if missing:
        if verbose:
            print(f"[pyvprnn_v2] diffusion cap: {', '.join(missing)} missing, trained without it")
        return None
    le_qc = get(["LE_F_MDS_QC"])
    le_ok = (le_qc == 0) if (le_qc is not None and le_qc.notna().any()) else le.notna()
    ta = ta - 273.15 if np.nanmedian(ta) > 150 else ta
    vpd = vpd / 10.0                                                  # FLUXNET VPD in hPa
    sw = sw * 1000.0 if np.nanpercentile(sw, 99) < 5 else sw          # kW m-2 -> W m-2
    pa = get(["PA_F", "PA", "PA_ERA"])
    pa = pd.Series(101.3, index=idx) if pa is None else pa
    pa = (pa / 10.0 if np.nanmedian(pa) > 200 else pa).fillna(101.3)    # hPa -> kPa
    ws, ust = get(["WS_F", "WS", "WS_ERA"]), get(["USTAR"])
    co2 = get(["CO2_F_MDS", "CO2"])
    ca = pd.Series(kw["ca_default"], index=idx) if co2 is None else co2.where(co2 > 200).fillna(kw["ca_default"])
    swpot = get(["SW_IN_POT"])
    day = (swpot > 0) if swpot is not None else (sw > 0)
    p = get(["P_F", "P", "P_ERA"])

    # energy balance closure per (month, solar hour); LE only ever increased
    lon = float(ds.attrs.get("site_lon", 0.0) or 0.0)
    mon = pd.Series(idx.month, index=idx)
    hr = pd.Series(np.floor((idx.hour + idx.minute / 60 + lon / 15.0) % 24), index=idx)
    dn = idx.normalize()
    le_corr = get(["LE_CORR"])
    if le_corr is not None and le_corr[le_ok].notna().mean() > 0.5:
        r = (le_corr / le).where(le.abs() > 10).clip(1.0, 2.0)
        ok = r.notna() & le_ok
        fac = r[ok].groupby([mon[ok], hr[ok]]).median()
        scale = pd.Series(fac.reindex(pd.MultiIndex.from_arrays([mon.values, hr.values])).values, index=idx)
        scale = scale.fillna(r.groupby(dn).transform("median")).fillna(r.median()).fillna(1.0)
        src = "LE_CORR"
    else:
        rn, g0 = get(["NETRAD", "NETRAD_F", "RN"]), get(["G_F_MDS", "G"])
        if rn is not None and rn.notna().any():
            gg = g0.fillna(0) if g0 is not None else 0.0
            ok = le_ok & h.notna() & rn.notna() & (sw > kw["sw_min"])
            cr = ((h + le)[ok].groupby([mon[ok], hr[ok]]).sum()
                  / (rn - gg)[ok].groupby([mon[ok], hr[ok]]).sum()).clip(0.5, 1.0)
            scale = (1.0 / pd.Series(cr.reindex(pd.MultiIndex.from_arrays([mon.values, hr.values])).values,
                                     index=idx)).fillna(1.0)
            src = "closure ratio (H+LE)/(Rn-G)"
        else:
            scale, src = pd.Series(1.0, index=idx), "none"
    sig = unc.fillna(unc[le_ok].median())
    le_b = le * scale + kw["n_sigma"] * sig
    avail = (h + le) * scale

    # Penman-Monteith inversion (imposed form where wind / u* are missing)
    es_ = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    delta = 4098 * es_ / (ta + 237.3) ** 2
    gamma = 0.000665 * pa
    rho_cp = pa * 1000 / (287.05 * (ta + 273.15)) * 1004.0
    gs_imp = le_b / (2.501e6 - 2361 * ta) / 0.018015 / (vpd / pa)
    if ws is not None and ust is not None:
        ga = 1.0 / (ws / ust ** 2 + 6.2 * ust ** (-2 / 3))
        den = delta * avail + rho_cp * vpd * ga - le_b * (delta + gamma)
        gs_pm = (gamma * ga * le_b / den * pa * 1000 / (8.314 * (ta + 273.15))).where(den > 0)
        gs = gs_pm.where(ws.notna() & ust.notna(), gs_imp)
    else:
        gs = gs_imp
    gs = gs.where(gs < 5).where(~(le_b <= 0), 0.0)

    wet = pd.Series(False, index=idx)
    if p is not None and p.notna().any() and kw["rain_excl_h"] > 0:
        n = int(2 * kw["rain_excl_h"]) + 1
        pf = p.fillna(0)
        wet = ((pf > kw["p_min"]).astype(int).rolling(n, min_periods=1).max() > 0) | \
              (pf.rolling(n, min_periods=1).sum() >= 1.0)
    valid = day.fillna(False) & (sw > kw["sw_min"]) & le_ok & (vpd > kw["vpd_min"]) & ~wet & ta.notna()
    bound = (gs * ca / 1.6).where(valid)
    if verbose:
        print(f"[pyvprnn_v2] diffusion cap: closure {src}, LE + {kw['n_sigma']:g} x LE_RANDUNC; "
              f"{int(bound.notna().sum())} half-hours with a measurable bound "
              f"({100 * bound.notna().sum() / max(int(day.sum()), 1):.1f} % of daytime), "
              f"median {np.nanmedian(bound):.2f} umol m-2 s-1")
    return bound


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

    def __init__(self, indices, n_features, delta, deltas=None, **kwargs):
        """deltas: optional per-index shifts (same length as indices); overrides delta."""
        super().__init__(**kwargs)
        self.indices = [int(i) for i in indices]
        self.n_features = int(n_features)
        self.delta = float(delta)
        self.deltas = None if deltas is None else [float(d) for d in deltas]
        shift = np.zeros(self.n_features, dtype=np.float32)
        shift[self.indices] = self.deltas if self.deltas is not None else self.delta
        self._shift = tf.constant(shift)

    def call(self, x):
        return x + self._shift

    def get_config(self):
        cfg = super().get_config()
        cfg.update(indices=self.indices, n_features=self.n_features, delta=self.delta, deltas=self.deltas)
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


class MixClasses(layers.Layer):
    """
    Linear mixing over land-cover classes: out = sum_c w_c * head_c, with w = the pixel's
    land-cover fractions normalised to sum 1 (pixels without any land-cover information: equal
    weights). Fluxes are additive over area, so a pixel's flux is the area-weighted sum of the
    fluxes of its parts. Inputs: [heads (B, H, W, C), fractions (B, H, W, C)] -> (B, H, W, 1).
    """

    def __init__(self, min_sum=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.min_sum = float(min_sum)

    def call(self, inputs):
        heads, frac = inputs
        frac = tf.nn.relu(tf.where(tf.math.is_finite(frac), frac, tf.zeros_like(frac)))
        s = tf.reduce_sum(frac, axis=-1, keepdims=True)
        n = tf.cast(tf.shape(frac)[-1], frac.dtype)
        w = tf.where(s > self.min_sum, frac / tf.maximum(s, self.min_sum), tf.ones_like(frac) / n)
        return tf.reduce_sum(w * heads, axis=-1, keepdims=True)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(min_sum=self.min_sum)
        return cfg


class GPPSignPenalty(layers.Layer):
    """
    Soft one-sided penalty on the sign of the GPP (LUE) response to one input
    or one group of inputs shifted together.

    Inputs: [out, lue, lue_shift, mask]  or  [out, lue, lue_shift, mask, taper_var]
      out             : returned unchanged
      lue, lue_shift  : LUE maps for the original inputs and for the shifted
                        inputs (B, H, W, 1); SW is not shifted, so the LUE ratio
                        equals the GPP ratio
      mask            : daytime x flux mask (B, H, W, 1)
      taper_var       : optional (B, 1, 1, 1) physical value of a variable; the
                        penalty fades out above taper_max (e.g. RH near saturation)

    sign = +1 (GPP must not decrease):  v = ReLU(ln(lue) - ln(lue_shift) - tol)
    sign = -1 (GPP must not increase):  v = ReLU(ln(lue_shift) - ln(lue) - tol)
    Adds  weight * < w * v >_mask  to the loss. Flat responses are never
    penalised; changes in the forbidden direction smaller than tol (relative,
    per shift) are free, so small genuine effects (dew films, wet leaves) remain
    representable and only clear inversions cost.
    """

    def __init__(self, weight=1.0, sign=1, tol=0.02, eps=1e-3,
                 taper_max=None, taper_width=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.sign = int(np.sign(sign)) or 1
        self.tol = float(tol)
        self.eps = float(eps)
        self.taper_max = None if taper_max is None else float(taper_max)
        self.taper_width = float(taper_width)

    def call(self, inputs):
        out, lue, lue_shift, mask = inputs[:4]
        d = tf.math.log(lue + self.eps) - tf.math.log(lue_shift + self.eps)   # > 0: GPP fell
        v = tf.nn.relu(self.sign * d - self.tol)
        if self.taper_max is not None and len(inputs) > 4:
            v = v * tf.sigmoid((self.taper_max - inputs[4]) / self.taper_width)
        penalty = tf.reduce_sum(v * mask) / (tf.reduce_sum(mask) + 1e-6)
        self.add_loss(self.weight * penalty)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(weight=self.weight, sign=self.sign, tol=self.tol, eps=self.eps,
                   taper_max=self.taper_max, taper_width=self.taper_width)
        return cfg


CUSTOM_OBJECTS = {
    "BroadcastToImage": BroadcastToImage,
    "ExpandLastDim": ExpandLastDim,
    "GlobalSumPooling": GlobalSumPooling,
    "DayMask": DayMask,
    "SelectFeatures": SelectFeatures,
    "GPPPenalty": GPPPenalty,
    "GPPSignPenalty": GPPSignPenalty,
    "MixClasses": MixClasses,
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
      - "t2m"                    : air temperature; t2m_source="tower" (default,
                                    as the model) -> ds["t2m"], "era5" -> t2m_era5
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

    # dataset variables read under another name in older site files (new name: old names)
    DS_ALIASES = {"RH_from_VPD": ("RH_from_VDP",)}
    # share of missing time steps of a met input above which a loud warning is given
    MAX_MISSING_SHARE = 0.2

    def __init__(self, *args, precip_var="P_F", api_sat_mm=API_SAT_MM, t2m_source="tower",
                 bound_var=None, lc_fill_class=LC_FILL_CLASS, zero_precip=False, **kwargs):
        """bound_var: optional dataset variable with the diffusion bound on GPP (training only);
        appended to the target pack as a third column (NaN -> -1 = no bound).
        lc_fill_class: land-cover class that receives the weight missing from pixels whose
        fractions sum to less than 1 (None: no filling).
        zero_precip: counterfactual without rain - precipitation set to 0 everywhere before the
        API features are computed (inference only; all other inputs unchanged)."""
        if t2m_source not in ("era5", "tower"):
            raise ValueError(f"t2m_source must be 'era5' or 'tower', got {t2m_source!r}")
        self.precip_var = precip_var
        self.zero_precip = bool(zero_precip)
        self.api_sat_mm = api_sat_mm
        self.t2m_source = t2m_source
        self.lc_fill_class = lc_fill_class
        super().__init__(*args, **kwargs)
        if bound_var is not None:
            b = self.ds_cropped[bound_var].sel(datetime_utc=self.time).values.astype(np.float32)
            self.y_pack = np.concatenate(
                [self.y_pack, np.nan_to_num(b, nan=-1.0)[..., None]], axis=-1).astype(np.float32)

    def _fill_land_cover(self, lc):
        """Assign the weight missing from each pixel (1 - sum of fractions) to lc_fill_class."""
        import warnings
        lc = np.nan_to_num(lc, nan=0.0)
        fill = getattr(self, "lc_fill_class", None)
        if fill is not None and fill not in self.land_cover_classes:
            raise ValueError(f"lc_fill_class {fill} not in land_cover_classes {self.land_cover_classes}")
        tot = lc.sum(axis=-1)
        missing = np.clip(1.0 - tot, 0.0, 1.0)
        filled = missing > 0.01
        over = tot > 1.01
        if filled.any() or over.any():
            site = self.ds_cropped.attrs.get("site_name", self.ds_cropped.attrs.get("site", "?"))
            action = (f"MISSING WEIGHT ASSIGNED TO CLASS {fill} (evergreen forest by default)" if fill is not None
                      else "NOT FILLED (lc_fill_class=None): fractions are renormalised to 1 in the mixing")
            msg = (f"LAND COVER DOES NOT SUM TO 1 (site {site}): {filled.mean():.1%} of pixels sum to < 0.99 "
                   f"(mean missing {missing[filled].mean() if filled.any() else 0:.2f}, "
                   f"max missing {missing.max():.2f}), {over.mean():.1%} of pixels sum to > 1.01. {action}. "
                   f"The input file is probably from the broken land-cover pipeline -> rerun the pipeline "
                   f"with --overwrite-landcover-cache for this site.")
            bar = "!" * 100
            print(f"\n{bar}\n!!! WARNING !!! {msg}\n{bar}\n")
            warnings.warn(msg, UserWarning, stacklevel=2)
        if fill is not None and filled.any():
            lc = lc.copy()
            lc[..., self.land_cover_classes.index(fill)] += np.where(filled, missing, 0.0)
        return lc

    def _ds_name(self, var):
        """Name under which `var` is stored in this dataset (old aliases accepted)."""
        if var in self.ds_cropped.variables:
            return var
        for alt in self.DS_ALIASES.get(var, ()):
            if alt in self.ds_cropped.variables:
                return alt
        return var

    def _required_ds_vars(self, sat_vars, met_vars):
        """Dataset variables needed for the given inputs: {dataset name: reason}."""
        req = {}
        add = lambda v, why: req.setdefault(v, why)
        t2m_name = self.TOWER_T2M_VAR if self.t2m_source == "tower" else self.ERA5_KEY_OVERRIDES["t2m"]
        for v in met_vars:
            if v == "t2m" or v.startswith("t2m_ema_"):
                add(t2m_name, v)
            elif v == "sd":
                add(self.ERA5_KEY_OVERRIDES["sd"], v)
            elif v.startswith("precip_api_tau"):
                add(self.precip_var, v)
                add(self.ERA5_KEY_OVERRIDES["sd"], f"{v} (snow gating)")
            elif v.startswith("swvl1_roll_"):
                add("swvl1_era5", v)
            elif v == "clearness_k":
                add(self.CLEARNESS_SW_VAR, v)
                add("SW_IN_POT", v)
            else:
                add(v, v)
        for v in sat_vars:
            add(v, "satellite input")
        for v in ("SW_IN_POT", "land_cover_map", "nirv_90pct", "nirv_10pct", "flux_mask"):
            add(v, "static / mask input")
        add(self._target, "target")
        add(self._unc, "target uncertainty")
        return req

    def _check_required_vars(self, sat_vars, met_vars):
        """Fail before anything is computed if an input variable is missing from the dataset."""
        req = self._required_ds_vars(sat_vars, met_vars)
        missing = {v: why for v, why in req.items() if self._ds_name(v) not in self.ds_cropped.variables}
        if missing:
            site = self.ds_cropped.attrs.get("site_name", self.ds_cropped.attrs.get("site", "?"))
            lines = "\n".join(f"    {v:28s} needed for {why}" for v, why in missing.items())
            raise KeyError(f"[pyvprnn_v2] site {site}: input variables missing from the dataset:\n{lines}\n"
                           f"Add them to the site file (python derived_variables.py <file>, or rerun the pipeline) - the model does not "
                           f"run with missing inputs.")
        old = [(v, self._ds_name(v)) for v in req if self._ds_name(v) != v]
        for new, alt in old:
            print(f"[pyvprnn_v2] reading {new} from the old variable name {alt}")

    def _check_met_coverage(self, met_vars):
        """All-NaN inputs are an error; inputs missing in more than MAX_MISSING_SHARE of the
        time steps give a loud warning (NaN inputs make the sample unusable)."""
        import warnings
        x = self.met_array.reshape(len(self.met_array), -1, len(met_vars))[:, 0, :]
        nan_share = np.isnan(x).mean(axis=0)
        dead = [v for v, f in zip(met_vars, nan_share) if f >= 0.999]
        if dead:
            raise ValueError(f"[pyvprnn_v2] input(s) entirely missing (all NaN): {', '.join(dead)}")
        bad = [(v, f) for v, f in zip(met_vars, nan_share) if f > self.MAX_MISSING_SHARE]
        if bad:
            msg = "inputs with many missing time steps: " + ", ".join(f"{v} {f:.0%}" for v, f in bad)
            print(f"\n{'!' * 100}\n!!! WARNING !!! {msg}\n{'!' * 100}\n")
            warnings.warn(msg, UserWarning, stacklevel=2)

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

        self._check_required_vars(sat_vars, met_vars)

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
            p_nan = float(np.isnan(precip_raw).mean())
            if p_nan >= 0.999:
                raise ValueError(f"[pyvprnn_v2] {self.precip_var} is entirely missing (all NaN)")
            if p_nan > 0.01 and not self.zero_precip:
                import warnings
                msg = (f"{self.precip_var} missing in {p_nan:.1%} of time steps; missing rain enters the API "
                       f"features as NO rain")
                print(f"\n{'!' * 100}\n!!! WARNING !!! {msg}\n{'!' * 100}\n")
                warnings.warn(msg, UserWarning, stacklevel=2)
            if self.zero_precip:
                precip_raw = np.zeros_like(precip_raw)
                print("[pyvprnn_v2] counterfactual: precipitation set to 0 (API features = 0)")
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
                da = self.ds_cropped[self._ds_name(v)].sel(datetime_utc=self.time)
                arr = apply_met_scaling(v, da.values)

            met_list.append(np.asarray(arr, dtype=np.float32))

        self.met_array = np.stack(met_list, axis=-1).astype(np.float32)
        self._check_met_coverage(met_vars)
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
        self.lc = self._fill_land_cover(self.lc)

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

    API_TAUS_HOURS = (12, 24, 72, 168)            # previous runs: (6, 24, 72, 168)
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
    MONO_TEMP_VARS = ("t2m", "stl2_era5")   # one extra Reco pass per listed variable
    MONO_DELTA_T = 1.0          # K (temperature inputs are in degC)
    MONO_WEIGHT = 1.0           # 0 switches the penalty off; scan e.g. 0.3 / 1 / 3
    MONO_T_CRIT = {"t2m": 40.0, "stl2_era5": 30.0}   # degC; penalty fades out above (float or {var: degC})
    MONO_T_CRIT_FALLBACK = 45.0 # for variables missing from a MONO_T_CRIT dict
    MONO_T_WIDTH = 2.0          # K, width of the fade-out

    MONO_SPLIT_WEIGHT = False   # full weight per variable (True: weight / n_vars)

    # Training data and loss (only act during training)
    TARGET_UNC = "NEE_VUT_REF_RANDUNC"     # u* part of JOINTUNC would down-weight the night data
    LOSS = "nll_loss_laplace_from_stacked"
    SIGMA_FLOOR = 0.5           # umol m-2 s-1, as in the loss functions
    NIGHT_OUTLIER_SIGMA = 3.0   # drop night NEE < -k sigma (None: legacy, drop all NEE < 0 at ssrd == 0)
    REQUIRE_SW_QC0 = True       # drop daytime samples with gap-filled radiation (SW_IN_F_QC != 0)

    # Numerical safeguards (soft caps; weight 1e-4)
    USE_RATIO_PENALTY = False   # Reco / GPP <= 1 is violated by every dormant period
    GPP_PENALTY_THRESHOLD = 80.0
    RECO_PENALTY_THRESHOLD = 50.0   # None: no absolute Reco cap
    PENALTY_WEIGHT = 1e-4

    # Hard LUE bound (None: softplus output x LUE_SCALE, i.e. unbounded)
    LUE_MAX = 200.0             # umol m-2 s-1 per kW m-2
    LUE_INIT = 25.0             # initial LUE with LUE_MAX (sets the output bias)

    # Soft sign constraints on the GPP response (GPPSignPenalty), at otherwise fixed
    # inputs (same SW, clearness, temperature). The LUE network is evaluated again with
    # the listed inputs shifted by GPP_SIGN_DELTA_SD training standard deviations (all
    # inputs of a group together); a change in the forbidden direction larger than
    # GPP_SIGN_TOL (relative) is penalised. Flat responses are always allowed.
    #   nirv     (+) more green canopy -> GPP not lower
    #   snow     (-) more snow -> GPP not higher
    #   moisture (+) wetter soil -> GPP not lower; current value and rolling statistics
    #                are shifted TOGETHER, so differences between them (drying vs.
    #                acclimation signals) stay unconstrained. Exempt at waterlogged sites
    #                (wetlands, peat, rice): gpp_sign_exclude=("moisture",)
    #   rh       (+) moister air at the same temperature -> GPP not lower; fades out above
    #                RH 0.90 (dew / fog films near saturation). CURRENTLY OFF (commented out
    #                below): known exceptions; the VPD response is learned from the data.
    # Not constrained: temperature (optimum), clearness / diffuse light (learned, a finding),
    # LSWI (snow raises it), rain history (wet leaves; acts through soil moisture).
    # Inputs missing from the GPP branch, or constant in the training data (sd at sites
    # without snow), are skipped.
    GPP_SIGN_WEIGHT = 1.0       # 0 switches all sign constraints off (weight per constraint)
    GPP_SIGN_DELTA_SD = 0.25    # shift in training standard deviations of each input
    GPP_SIGN_TOL = 0.02         # tolerated relative change in the forbidden direction per shift
    GPP_SIGN_CONSTRAINTS = (
        dict(name="nirv", source="sat", vars=("nirv",), sign=+1),
        dict(name="snow", source="met", vars=("sd",), sign=-1),
        dict(name="moisture", source="met",
             vars=("swvl2_era5", "swvl1_roll_mean_168h", "swvl1_roll_min_720h"), sign=+1),
        # dict(name="rh", source="met", vars=("RH_from_VPD",), sign=+1,
        #      taper_max=0.90, taper_width=0.03),      # RH as fraction; % detected automatically
    )

    # Linear mixing over land-cover classes: GPP and Reco branches have one output head per class
    # (shared hidden layers), the pixel flux is the area-weighted sum of the class fluxes
    # (MixClasses). The land-cover fractions are then NOT an input of the networks, so they cannot
    # be combined nonlinearly (e.g. as a pixel fingerprint / wind-direction proxy). False: previous
    # behaviour (fractions as network inputs, one output).
    LINEAR_MIXING = True
    LC_FILL_CLASS = LC_FILL_CLASS   # class receiving missing land-cover weight (None: no filling)

    # Diffusion bound on footprint GPP from LE (Fick's law, Ci = 0; see diffusion_bound()):
    # training only, penalty weight * < ReLU(GPP_site - bound) > (umol m-2 s-1) on all half-hours
    # with a measurable bound. Sites without LE / LE_RANDUNC are trained without it.
    DIFFUSION_CAP_WEIGHT = 1.0
    DIFFUSION_BOUND_VAR = "GPP_DIFFUSION_BOUND"

    # Clearness index input in the GPP branch
    USE_CLEARNESS = True

    # Source of the air temperature input "t2m" (also t2m_ema and the Reco
    # monotonicity penalty): "tower" (TA_F; consistent with RH_from_VPD and with NT / DT)
    # or "era5" (t2m_era5)
    T2M_SOURCE = "tower"

    # Standardise the met inputs inside the branches (mean/std of the training
    # samples, stored in the model). Without it, inputs of very different scale
    # (RH in %, temperatures in degC, API in mm next to swvl ~0.3) give large
    # hidden activations, and the LUE output can saturate early in training and
    # stay at 0 (observed on synthetic data for the previous configuration too).
    NORMALIZE_INPUTS = True

    # Reco branch inputs
    RECO_SWVL_ROLL = (("mean", 2160), ("min", 3600))   # None: all swvl1 rolling stats
    RECO_SAT_VARS = ("nirv",)   # None: all sat_vars (LSWI also)

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
                 normalize_inputs=None, t2m_source=None,
                 gpp_sign_weight=None, gpp_sign_exclude=(), gpp_sign_constraints="default",
                 api_taus_hours=None, diffusion_cap_weight=None, diffusion_cap_kw=None,
                 linear_mixing=None, lc_fill_class="default", zero_precip=False, **kwargs):
        """
        The defaults are the model configuration. The options exist to load
        models trained with earlier settings, which need the matching inputs at
        inference, e.g.
            pyvprnn_v2(api_taus_hours=(6, 24, 72, 168))       # previous runs
            pyvprnn_v2(api_sat_mm=None, use_t_memory=False)   # raw API, no memory
        Options that change the network inputs or structure (api_taus_hours,
        use_clearness, lue_max, reco_swvl_roll, reco_sat_vars, normalize_inputs,
        t2m_source, api_sat_mm, use_t_memory) must match the trained model.
        zero_precip=True: no-rain counterfactual at inference (precipitation = 0 before the API
        features; see set_counterfactual). Not allowed in train().
        Training-only options (losses, penalties, sign constraints, e.g.
        gpp_sign_exclude=("moisture",) at waterlogged sites) do not matter at
        inference. "default" / None take the class attribute; options whose
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
        self.gpp_sign_weight = float(pick(gpp_sign_weight, "GPP_SIGN_WEIGHT"))
        self.api_taus_hours = tuple(int(h) for h in pick(api_taus_hours, "API_TAUS_HOURS"))
        self.diffusion_cap_weight = float(pick(diffusion_cap_weight, "DIFFUSION_CAP_WEIGHT"))
        self.linear_mixing = bool(pick(linear_mixing, "LINEAR_MIXING"))
        self.lc_fill_class = pick(lc_fill_class, "LC_FILL_CLASS", "default")
        self.diffusion_cap_kw = dict(diffusion_cap_kw or {})
        if 168 not in self.api_taus_hours:
            raise ValueError("api_taus_hours must contain 168 (input of the GPP branch).")
        cons = pick(gpp_sign_constraints, "GPP_SIGN_CONSTRAINTS", "default")
        excl = {gpp_sign_exclude} if isinstance(gpp_sign_exclude, str) else set(gpp_sign_exclude or ())
        unknown_excl = excl - {c["name"] for c in cons}
        if unknown_excl:
            raise ValueError(f"gpp_sign_exclude {sorted(unknown_excl)} not in the sign constraints.")
        self.gpp_sign_constraints = [dict(c) for c in cons if c["name"] not in excl]
        self._met_stats = None
        self._sign_stats = None
        if self.lue_max is not None and not (0 < self.lue_init < self.lue_max):
            raise ValueError("lue_init must lie in (0, lue_max).")

        self.sat_vars = ["lswi", "nirv"]
        self.precip_var = precip_var if precip_var is not None else self.DEFAULT_PRECIP_VAR
        self.api_sat_mm = API_SAT_MM if api_sat_mm == "default" else api_sat_mm
        # generator_kwargs must reach every generator (training AND inference), see utils.make_generator
        self.generator_kwargs = {"precip_var": self.precip_var, "api_sat_mm": self.api_sat_mm,
                                 "lc_fill_class": self.lc_fill_class,
                                 "zero_precip": bool(zero_precip),
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

        api_vars = [f"precip_api_tau{h}h" for h in self.api_taus_hours]
        swvl_roll_vars = []
        for h in self.SWVL_ROLLING_HOURS:
            swvl_roll_vars += [f"swvl1_roll_mean_{h}h", f"swvl1_roll_min_{h}h"]

        clear_vars = ["clearness_k"] if self.use_clearness else []

        self.met_vars = (
            ["t2m", "ssrd", "RH_from_VPD", "swvl1_era5", "swvl2_era5", "sd",
             "stl1_era5", "stl2_era5"]
            + api_vars + swvl_roll_vars + t_mem_vars + clear_vars
        )

        self.gpp_met_vars = [
            "t2m", "ssrd", "RH_from_VPD", "swvl2_era5", "sd",
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

        # static input = [nirv_90pct, nirv_10pct, land-cover fractions (land_cover_classes)]
        n_lc = len(self.land_cover_classes)
        if self.linear_mixing:
            if n_static_features <= n_lc:
                raise ValueError("linear_mixing: static input has no non-land-cover features.")
            lc_idx = list(range(n_static_features - n_lc, n_static_features))
            trunk_idx = list(range(n_static_features - n_lc))
            static_trunk = SelectFeatures(trunk_idx, name="static_trunk")(self.static_input)
            lc_frac = SelectFeatures(lc_idx, name="lc_fractions")(self.static_input)
            n_heads = n_lc
            mix_lue = MixClasses(name="x_lue_map" if self.use_lue_light else "x_gpp_map")
            mix_reco = MixClasses(name="x_reco_map")
        else:
            static_trunk, lc_frac, n_heads = self.static_input, None, 1

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
            n_heads, 1, activation=out_act, kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(out_bias),
            name=("x_lue_heads" if self.use_lue_light else "x_gpp_heads") if self.linear_mixing
            else ("x_lue_map" if self.use_lue_light else "x_gpp_map"),
        )

        if self.normalize_inputs:
            if self._met_stats is None:
                raise RuntimeError("normalize_inputs=True needs the met statistics (set in train()).")
            mu, sd = self._met_stats
            std_gpp = Standardize(mu[gpp_met_idx], sd[gpp_met_idx], name="std_met_gpp")
            std_reco = Standardize(mu[reco_met_idx], sd[reco_met_idx], name="std_met_reco")
        else:
            std_gpp = std_reco = (lambda t: t)

        def gpp_branch(met_bc, sat=None):
            x = gpp_concat([self.sat_input if sat is None else sat, static_trunk, std_gpp(met_bc)])
            for layer in gpp_hidden:
                x = layer(x)
            heads = gpp_out(x)                                  # one LUE per class (bounded per class)
            return mix_lue([heads, lc_frac]) if self.linear_mixing else heads

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

        # ---- Soft sign constraints on the GPP response (GPP_SIGN_CONSTRAINTS) ----
        if self.gpp_sign_weight > 0 and self.gpp_sign_constraints:
            x_gpp = self._add_gpp_sign_penalties(x_gpp, gpp_branch, met_gpp, met_bc_gpp,
                                                 sat_static, day_mask, flux_mask_exp)

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
            n_heads, 1, activation="softplus", kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.7),
            name="x_reco_heads" if self.linear_mixing else "x_reco_map",
        )

        if self.reco_sat_vars_ == list(self.sat_vars):
            sat_reco = self.sat_input
        else:
            sat_reco = SelectFeatures([self.sat_vars.index(v) for v in self.reco_sat_vars_],
                                      name="sat_reco")(self.sat_input)

        def reco_branch(met_bc):
            x = reco_concat([sat_reco, static_trunk, std_reco(met_bc)])
            for layer in reco_hidden:
                x = layer(x)
            heads = reco_out(x)                                 # one Reco per class
            return mix_reco([heads, lc_frac]) if self.linear_mixing else heads

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

    def _add_gpp_sign_penalties(self, x_gpp, gpp_branch, met_gpp, met_bc_gpp,
                                sat_static, day_mask, flux_mask_exp):
        """One extra GPP-branch pass per sign constraint (see GPP_SIGN_CONSTRAINTS).
        Shifts are DELTA_SD x training std, in physical input units (before the
        branch standardisation). Only acts during training."""
        if self._sign_stats is None:
            raise RuntimeError("gpp_sign_weight > 0 needs the input statistics (set in train()).")
        st = self._sign_stats
        mask = layers.Multiply(name="gpp_sign_mask")([day_mask, flux_mask_exp])
        n_gpp = len(self.gpp_met_vars)
        for c in self.gpp_sign_constraints:
            name, src = c["name"], c["source"]
            names = self.gpp_met_vars if src == "met" else self.sat_vars
            sds = st["met_sd"] if src == "met" else st["sat_sd"]
            full = self.met_vars if src == "met" else self.sat_vars
            idx, deltas = [], []
            for v in c["vars"]:
                if v not in names:
                    print(f"[pyvprnn_v2] sign constraint '{name}': {v} not in the GPP inputs, skipped")
                    continue
                sd = float(sds[full.index(v)])
                if not np.isfinite(sd) or sd < 1e-6:
                    print(f"[pyvprnn_v2] sign constraint '{name}': {v} constant in the training data, skipped")
                    continue
                idx.append(names.index(v))
                deltas.append(self.GPP_SIGN_DELTA_SD * sd)
            if not idx:
                continue
            if src == "met":
                met_s = ShiftFeatures(idx, n_gpp, 0.0, deltas=deltas, name=f"met_gpp_sign_{name}")(met_gpp)
                met_bc_s = BroadcastToImage(name=f"met_bc_gpp_sign_{name}")([met_s, sat_static])
                x_shift = gpp_branch(met_bc_s)
            else:
                sat_s = ShiftFeatures(idx, len(self.sat_vars), 0.0, deltas=deltas,
                                      name=f"sat_gpp_sign_{name}")(self.sat_input)
                x_shift = gpp_branch(met_bc_gpp, sat=sat_s)
            inputs = [x_gpp, x_gpp, x_shift, mask]
            taper_max = taper_width = None
            if c.get("taper_max") is not None and src == "met":
                tv = c["vars"][0]
                scale = st["rh_scale"] if tv == "RH_from_VPD" else 1.0
                taper_max, taper_width = c["taper_max"] * scale, c.get("taper_width", 0.03) * scale
                inputs.append(SelectFeatures([self.met_vars.index(tv)], name=f"met_sign_taper_{name}")(self.met_input))
            x_gpp = GPPSignPenalty(weight=self.gpp_sign_weight, sign=c["sign"], tol=self.GPP_SIGN_TOL,
                                   taper_max=taper_max, taper_width=taper_width or 1.0,
                                   name=f"gpp_sign_penalty_{name}")(inputs)
            print(f"[pyvprnn_v2] sign constraint '{name}' ({'+' if c['sign'] > 0 else '-'}): "
                  + ", ".join(f"{names[i]} +{d:.3g}" for i, d in zip(idx, deltas)))
        return x_gpp

    def _set_sign_stats(self, train_gen):
        """Training std of the met and sat inputs (shift sizes of the sign constraints)."""
        met2d = train_gen.met_array.reshape(len(train_gen.met_array), -1)[train_gen.indexes]
        sat = train_gen.sat_array.reshape(-1, train_gen.sat_array.shape[-1])
        rh = met2d[:, self.met_vars.index("RH_from_VPD")] if "RH_from_VPD" in self.met_vars else np.array([0.5])
        self._sign_stats = dict(met_sd=np.nanstd(met2d, axis=0), sat_sd=np.nanstd(sat, axis=0),
                                rh_scale=100.0 if np.nanmedian(rh) > 1.5 else 1.0)

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
        items = dict(api_taus_hours=self.api_taus_hours, t2m_source=self.t2m_source, target_unc=self.target_unc, loss=self.loss_name, night_outlier_sigma=self.night_outlier_sigma,
                     require_sw_qc0=self.require_sw_qc0, use_ratio_penalty=self.use_ratio_penalty,
                     gpp_penalty_threshold=self.gpp_penalty_threshold,
                     reco_penalty_threshold=self.reco_penalty_threshold, lue_max=self.lue_max,
                     use_clearness=self.use_clearness, mono_temp_vars=self.mono_temp_vars,
                     mono_t_crit=self.mono_t_crit, mono_split_weight=self.mono_split_weight,
                     gpp_met_vars=self.gpp_met_vars, reco_met_vars=self.reco_met_vars,
                     reco_sat_vars=self.reco_sat_vars_, normalize_inputs=self.normalize_inputs,
                     gpp_sign_weight=self.gpp_sign_weight, diffusion_cap_weight=self.diffusion_cap_weight,
                     linear_mixing=self.linear_mixing, lc_fill_class=self.lc_fill_class,
                     gpp_sign_constraints=[f"{c['name']}({'+' if c['sign'] > 0 else '-'})"
                                           for c in self.gpp_sign_constraints] if self.gpp_sign_weight > 0 else None)
        print(f"[{type(self).__name__}]")
        for k, v in items.items():
            print(f"  {k:24s} {v}")

    def set_counterfactual(self, zero_precip=True):
        """Switch the no-rain counterfactual on / off for all generators built afterwards
        (inference, e.g. on a loaded model): precipitation = 0 everywhere, so all API features
        are 0. Soil moisture, LSWI and the other inputs keep their observed values."""
        self.generator_kwargs["zero_precip"] = bool(zero_precip)
        return self

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
        if self.generator_kwargs.get("zero_precip"):
            raise ValueError("zero_precip is a counterfactual for inference only; train without it.")
        self.describe()

        # diffusion bound from LE (training only): computed once and stored in ds_cropped
        cap_kwargs = {}
        if self.diffusion_cap_weight > 0:
            bv = self.DIFFUSION_BOUND_VAR
            if bv not in self.ds_cropped.variables:
                b = diffusion_bound(self.ds_cropped, **self.diffusion_cap_kw)
                if b is not None:
                    self.ds_cropped[bv] = ("datetime_utc", b.values.astype(np.float32))
            if bv in self.ds_cropped.variables and np.isfinite(self.ds_cropped[bv].values).any():
                cap_kwargs = {"bound_var": bv}

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
            **self.generator_kwargs, **cap_kwargs,
        )
        val_gen = self.generator_cls(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            times=val_times_qc0,
            shuffle=False,
            target=target, unc=target_unc,
            land_cover_classes=self.land_cover_classes,
            **self.generator_kwargs, **cap_kwargs,
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
        if self.gpp_sign_weight > 0:
            self._set_sign_stats(train_gen)

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

        fit_model, metrics = self.model, [mse_true_only]
        if cap_kwargs:
            # training model with outputs [NEE, footprint GPP]; shares all layers (and the add_loss
            # penalties) with self.model, which keeps its NEE-only signature
            gpp_site = layers.Flatten(name="gpp_site")(self.model.get_layer("gpp_sum_raw").output)
            fit_model = Model(inputs=self.model.inputs,
                              outputs=layers.Concatenate(name="nee_gpp")([self.model.output, gpp_site]))
            base_loss, w_cap = loss, self.diffusion_cap_weight

            def loss(y_true, y_pred):
                b = y_true[..., 2:3]
                excess = tf.nn.relu(y_pred[..., 1:2] - b) * tf.cast(b >= 0, y_pred.dtype)
                return base_loss(y_true, y_pred[..., :1]) + w_cap * tf.reduce_mean(excess)

            def mse_true_only(y_with_sigma_true, y_pred):          # same metric name as without the cap
                return tf.reduce_mean(tf.square(y_with_sigma_true[..., 0:1] - y_pred[..., :1]))

            def diffusion_excess(y_true, y_pred):                  # mean excess over the bound where it applies
                b = y_true[..., 2:3]
                valid = tf.cast(b >= 0, y_pred.dtype)
                return tf.reduce_sum(tf.nn.relu(y_pred[..., 1:2] - b) * valid) / (tf.reduce_sum(valid) + 1e-6)

            metrics = [mse_true_only, diffusion_excess]

        fit_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=train_params["learning rate"]),
            loss=loss,
            metrics=metrics,
        )
        fit_model.summary()

        early_stop = EarlyStopping(monitor="val_loss", patience=train_params["patience"], restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=train_params["plateau_patience"], min_lr=1e-6, verbose=1,
        )
        callbacks = [early_stop, reduce_lr]
        if train_params["max_runtime_in_seconds"] is not None:
            callbacks.append(TimeLimitCallback(train_params["max_runtime_in_seconds"]))

        history = fit_model.fit(
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
# Alias: configs and scripts that use the name pyvprnn_v2_pub get the same model
# (and type(model).__name__ == "pyvprnn_v2", so utils.resolve_pyvprnn_model works).
# =============================================================================

pyvprnn_v2_pub = pyvprnn_v2