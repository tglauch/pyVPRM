import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from pyVPRM.vprm_models.pyvprnn import pyvprnn
from pyVPRM.lib.fancy_plot import figsize
from pyVPRM.lib.functions import sel_nearest_valid


class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        print(f"Elapsed time: {elapsed:.1f}s")

        if elapsed > self.max_seconds:
            print(f"\nStopping training after {elapsed:.1f} seconds")
            self.model.stop_training = True


class BroadcastToImage(tf.keras.layers.Layer):
    def call(self, inputs):
        m, ref = inputs
        m_shape = tf.shape(m)
        ref_shape = tf.shape(ref)

        if m.shape.rank == 2:
            m = tf.expand_dims(tf.expand_dims(m, axis=1), axis=1)  # (B,1,1,F)
            m = tf.tile(m, [1, ref_shape[1], ref_shape[2], 1])

        elif m.shape.rank == 4:
            tile_h = tf.math.floordiv(ref_shape[1], m_shape[1])
            tile_w = tf.math.floordiv(ref_shape[2], m_shape[2])
            tile_h = tf.where(m_shape[1] == ref_shape[1], 1, tile_h)
            tile_w = tf.where(m_shape[2] == ref_shape[2], 1, tile_w)
            m = tf.tile(m, [1, tile_h, tile_w, 1])

        else:
            raise ValueError(f"Unsupported met input rank: {m.shape.rank}")

        return m

    def compute_output_shape(self, input_shape):
        m_shape, ref_shape = input_shape
        if len(m_shape) == 2:
            return (m_shape[0], ref_shape[1], ref_shape[2], m_shape[1])
        elif len(m_shape) == 4:
            out_H = ref_shape[1] if m_shape[1] == 1 else m_shape[1]
            out_W = ref_shape[2] if m_shape[2] == 1 else m_shape[2]
            return (m_shape[0], out_H, out_W, m_shape[3])
        else:
            raise ValueError(f"Unsupported met input shape: {m_shape}")

    def get_config(self):
        return super().get_config()


class ExpandLastDim(layers.Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=-1)

    def get_config(self):
        return super().get_config()


class GlobalSumPooling(layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=[1, 2, 3])

    def get_config(self):
        return super().get_config()


class DayMask(layers.Layer):
    def __init__(self, ssrd_idx, **kwargs):
        super().__init__(**kwargs)
        self.ssrd_idx = ssrd_idx

    def call(self, m):
        # m shape: (batch, y, x, n_features) - as of this change, m is
        # always the broadcast SW_IN_POT tensor (1 feature), so ssrd_idx
        # is always 0. Kept as a parameter (rather than hardcoding 0
        # inside the layer) so the layer class itself stays general and
        # get_config()/serialization keeps working the same as before.
        sw = m[..., self.ssrd_idx]
        mask = tf.cast(sw > 0, tf.float32)
        return tf.expand_dims(mask, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ssrd_idx": self.ssrd_idx})
        return cfg


class SelectFeatures(tf.keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def call(self, x):
        return tf.gather(x, self.indices, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"indices": self.indices})
        return cfg


class GPPPenalty(layers.Layer):
    def __init__(self, threshold=40.0, weight=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.weight = weight

    def call(self, x):
        excess = tf.nn.relu(x - self.threshold)
        penalty = self.weight * tf.reduce_mean(excess)
        self.add_loss(penalty)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold, "weight": self.weight})
        return cfg


def apply_met_scaling(var_name, arr):
    """
    Single place for the met-variable scaling that used to be duplicated
    inline (and flagged as "really bad, fix ASAP"). Values themselves stay
    hardcoded per the decision to keep sat_vars/met_vars/scaling out of
    config - this only removes the duplication/inline-comment situation,
    not the hardcoding itself.
    """
    if var_name == "ssrd":
        return arr / 1000
    if var_name in ("SWC_F_MDS_1", "SWC_F_MDS_2"):
        return arr / 10
    return arr

def compute_hours_since_rain(precip_values, dt_hours=0.5, rain_threshold_mm=0.1):
    """
    Hours elapsed since precipitation last exceeded rain_threshold_mm.
    NaN before the first recorded rain event in the series (no prior
    history to draw on yet).
    """
    is_rain = precip_values > rain_threshold_mm
    hours_since = np.full(len(precip_values), np.nan, dtype=np.float32)
    last = np.nan
    for i, rained in enumerate(is_rain):
        if rained:
            last = 0.0
        elif not np.isnan(last):
            last += dt_hours
        hours_since[i] = last
    return hours_since

def compute_precip_rolling_sum(precip_values, window_steps):
    """
    Rolling sum of precipitation over the trailing window_steps (inclusive
    of the current step). Uses min_periods=1 so early-record timesteps get
    a sum over whatever history is actually available, rather than NaN -
    this slightly understates true accumulation right at the start of the
    record (same soft bias as hours_since_rain's early NaNs), but doesn't
    need hard exclusion the way the lag-window NaNs did, since it's a
    mild underestimate rather than a meaningless value.
    """
    s = pd.Series(precip_values)
    return s.rolling(window=window_steps, min_periods=1).sum().values.astype(np.float32)
    
class BatchGenerator(tf.keras.utils.Sequence):
    """
    NOTE on batch tuple shape: __getitem__ returns
    (sat, static, met, sw_in_pot, fp, mask), ypack.
    """

    def __init__(self, ds_cropped, sat_vars, met_vars,
                 batch_size, land_cover_classes, shuffle=True, met_dim=1, times=None,
                 workers=1, use_multiprocessing=False,
                 max_queue_size=1, target="NEE_VUT_REF",
                 unc="NEE_VUT_REF_JOINTUNC"):
        super().__init__(workers=workers,
                          use_multiprocessing=use_multiprocessing,
                          max_queue_size=max_queue_size)

        self.ds_cropped = ds_cropped
        self.sat_vars = sat_vars
        self.met_vars = met_vars
        self.land_cover_classes = land_cover_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.met_dim = met_dim
        self._target = target
        self._unc = unc
        self.init_training_cache(sat_vars, met_vars, met_dim)
        self._static_cache_B = None
        self._static_cache = None
        self._mask_cache = None
        self._mask_cache_B = None

        if times is not None:
            time_idx = np.array([self.time_index[t] for t in times])
            self.indexes = time_idx
        else:
            self.indexes = np.arange(len(self.time))
        self.n = len(self.indexes)
        self.on_epoch_end()

    def init_training_cache(self, sat_vars, met_vars, met_dim=1):
        self.time = self.ds_cropped["datetime_utc"].values
        self.time_index = {t: i for i, t in enumerate(self.time)}
        self.sat_time_index = self.ds_cropped["days_since_t0"].values

        # MET DATA
        met_list = []
        for v in met_vars:
            if v == "hours_since_rain":
                precip = self.ds_cropped["P_F"].sel(datetime_utc=self.time).values  # check actual precip variable name in your dataset
                arr = np.log1p(compute_hours_since_rain(precip))
            elif v.startswith("precip_sum_"):
                window_steps = int(v.split("_")[-1])  # e.g. "precip_sum_144" -> 144 half-hour steps = 72h
                precip = self.ds_cropped["P_F"].sel(datetime_utc=self.time).values
            elif v.endswith("_era5"):
                da = sel_nearest_valid(
                    self.ds_cropped[[v]].compute(),
                    lon=self.ds_cropped.attrs["site_lon"],
                    lat=self.ds_cropped.attrs["site_lat"]
                )[v]
            else:
                da = self.ds_cropped[v]

            da = da.sel(datetime_utc=self.time)
            arr = apply_met_scaling(v, da.values)
            met_list.append(arr)

        self.met_array = np.stack(met_list, axis=-1).astype(np.float32)
        if met_dim == 1:
            self.met_array = self.met_array[:, None, None, :]

        # DAY/NIGHT MASK SOURCE - SW_IN_POT (potential/clear-sky radiation,
        # a pure function of solar geometry) rather than ssrd (actual
        # measured radiation). This means day/night classification isn't
        # thrown off by cloud cover the way actual measured irradiance
        # would be (e.g. a heavily overcast midday no longer risks looking
        # like night). Loaded ONLY for this purpose - deliberately not
        # part of met_vars/gpp_met_vars/reco_met_vars, so it never reaches
        # the model as a regular input, per the "don't use it anywhere
        # else" requirement.
        sw_in_pot = self.ds_cropped["SW_IN_POT"].sel(datetime_utc=self.time).values.astype(np.float32)
        self.sw_in_pot_array = sw_in_pot
        if met_dim == 1:
            self.sw_in_pot_array = self.sw_in_pot_array[:, None, None, None]

        # SAT DATA
        self.sat_array = np.stack(
            [self.ds_cropped[v].values for v in sat_vars],
            axis=-1).astype(np.float32)

        # STATIC - land_cover_classes is the one config-driven exception
        # (derived from vprm_config_path, see resolve_vprm_config_path() in
        # utils.py) - getting the class count wrong here silently breaks
        # the static input's shape/meaning, so it isn't left hardcoded like
        # sat_vars/met_vars.
        self.lc = np.moveaxis(
            self.ds_cropped["land_cover_map"]
            .sel(vprm_classes=self.land_cover_classes)
            .values,
            0, -1).astype(np.float32)

        self.nirv_max = self.ds_cropped["nirv_90pct"].values[..., None].astype(np.float32)
        self.nirv_min = self.ds_cropped["nirv_10pct"].values[..., None].astype(np.float32)

        # TARGETS
        self.y_array = self.ds_cropped[self._target].sel(
            datetime_utc=self.time).values.astype(np.float32)

        self.y_unc_array = self.ds_cropped[self._unc].sel(
            datetime_utc=self.time).values.astype(np.float32)

        self.fp_array = self.ds_cropped["ffp_footprint"].sel(
            t=self.time).values.astype(np.float32)

        self.mask_static = self.ds_cropped["flux_mask"].values.astype(np.float32)

        self.static_stack = np.concatenate(
            [self.nirv_max, self.nirv_min, self.lc], axis=-1)

        self.y_pack = np.stack(
            [self.y_array, self.y_unc_array], axis=-1).astype(np.float32)

    def times_to_idx(self, times):
        return np.array([self.time_index[t] for t in times])

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_batch_times(self, batch_index):
        batch_idxs = self.indexes[
            batch_index * self.batch_size: (batch_index + 1) * self.batch_size
        ]
        return self.time[batch_idxs]

    def __getitem__(self, batch_index):
            batch_idxs = self.indexes[
                batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            Xmet = self.met_array[batch_idxs]
            Xsw_in_pot = self.sw_in_pot_array[batch_idxs]
            fp = self.fp_array[batch_idxs]
            ypack = self.y_pack[batch_idxs]
            # Expand only this batch's worth of samples, not the whole record.
            sat = self.sat_array[self.sat_time_index[batch_idxs]]
            B = sat.shape[0]
    
            if self._static_cache_B != B:
                self._static_cache = np.broadcast_to(self.static_stack, (B,) + self.static_stack.shape)
                self._static_cache_B = B
    
            if self._mask_cache_B != B:
                self._mask_cache = np.broadcast_to(self.mask_static, (B,) + self.mask_static.shape)
                self._mask_cache_B = B
            mask = self._mask_cache
            return (sat, self._static_cache, Xmet, Xsw_in_pot, fp, mask), ypack


class pyvprnn_v1(pyvprnn):
    """
    Base class for all pyvprnn models.

    sat_vars/met_vars/gpp_met_vars/reco_met_vars are intentionally hardcoded
    class defaults, not config-driven. land_cover_classes is the one
    exception that stays settable, since a mismatch there silently breaks
    the static input rather than just being a modeling choice.
    """

    DEFAULT_LAND_COVER_CLASSES = [1, 2, 3, 4, 5, 6, 7]

    def __init__(self, land_cover_classes=None):
        super().__init__()
        self.train_weights = None
        self.valid_weights = None

        self.sat_vars = ["lswi", "nirv", "ndre"]
        self.met_vars = ["t2m", "ssrd", "RH_from_VDP", "swvl1_era5", "swvl2_era5"]
        # GPP excludes swvl1_era5 (kept for Reco's water-availability signal);
        # Reco excludes ssrd (no direct light-driven respiration signal) and
        # swvl2_era5 (deeper layer, judged more relevant to GPP's root-zone
        # water access). Note ssrd (actual measured radiation) still feeds
        # GPP as a real driver of photosynthetic rate - only the day/night
        # MASK now comes from SW_IN_POT instead, a separate concern.
        self.gpp_met_vars = ["t2m", "ssrd", "RH_from_VDP", "swvl2_era5"]
        self.reco_met_vars = ["t2m", "RH_from_VDP", "swvl1_era5"]

        self.land_cover_classes = (
            list(land_cover_classes) if land_cover_classes is not None
            else list(self.DEFAULT_LAND_COVER_CLASSES)
        )
        return

    def load_model(self, path):
        self.pixel_model = load_model(path, custom_objects={
            "BroadcastToImage": BroadcastToImage,
            "ExpandLastDim": ExpandLastDim,
            "GlobalSumPooling": GlobalSumPooling,
            "DayMask": DayMask,
            "SelectFeatures": SelectFeatures,
            "GPPPenalty": GPPPenalty,
        })

    def make_veg_fraction_plot(self, opath):
        import matplotlib.pyplot as plt

        fracs = (self.ds_cropped["land_cover_map"] * self.ds_cropped["ffp_footprint"]).sum(dim=["x", "y"])
        valid_footprint_mask = fracs.sum(dim="vprm_classes") != 0
        fig, axes = plt.subplots(9, 1, figsize=figsize(1.0, 1.4), sharex=True)
        classes = [
            (1, "grey", "EF"), (2, "grey", "DF"), (3, "grey", "MF"),
            (4, "grey", "SH"), (5, "grey", "SAV"), (6, "grey", "CRO"),
            (7, "grey", "GRA"), (8, "grey", "URB"), (9, "grey", "WET"),
        ]

        for i, (ax, (cls, color, label)) in enumerate(zip(axes, classes)):
            fracs.where(valid_footprint_mask, drop=True).sel({"vprm_classes": cls}).plot(ax=ax, color=color)
            ax.text(0.01, 0.99, label, horizontalalignment="left", verticalalignment="top",
                    transform=ax.transAxes, color="red")
            ax.grid(alpha=0.3)
            ax.set_title("")
            if i != 8:
                ax.set_xlabel("")

        axes[-1].set_xlabel("Time")
        if opath is not None:
            fig.savefig(opath, dpi=300, bbox_inches="tight")
        return

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

        train_gen = BatchGenerator(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            land_cover_classes=self.land_cover_classes,
            times=train_times_qc0,
            workers=train_params["workers"],
            use_multiprocessing=train_params["multiprocessing"],
            max_queue_size=train_params["max_queue_size"],
            target=target, unc=target_unc,
        )

        val_gen = BatchGenerator(
            self.ds_cropped, self.sat_vars, self.met_vars,
            batch_size=train_params["batch_size"],
            land_cover_classes=self.land_cover_classes,
            times=val_times_qc0,
            shuffle=False,
            target=target, unc=target_unc,
        )

        (Xsat_batch, Xstatic_batch, Xmet_batch, _, _, _), _ = train_gen[0]

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
        # Day/night mask - from SW_IN_POT, not ssrd (see BatchGenerator note)
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
        # RECO branch
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
        reco_weighted = layers.Multiply(name="reco_weighted")([reco_map, fp_exp])
        reco_sum = GlobalSumPooling(name="reco_sum")(reco_weighted)

        # =========================================================
        # NEE (physics only)
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
            # https://research.fs.usda.gov/download/treesearch/21276.pdf
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
            inputs=[self.sat_input, self.static_input, self.met_input, self.sw_in_pot_input, self.flux_mask_input],
            outputs=[self.model.get_layer("gpp_map").output, self.model.get_layer("reco_map").output],
            name="pixel_flux_model",
        )
        self.pixel_model.save(save_path_model)