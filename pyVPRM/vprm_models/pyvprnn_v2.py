import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

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
)


class LaggedBatchGenerator(BatchGenerator):
    """
    Same as BatchGenerator, but also precomputes a windowed/lagged met
    array (one row per requested lag, per lagged variable) for the Reco
    branch's temporal-history encoder.

    NOTE on batch tuple shape: BatchGenerator.__getitem__ now returns
    (sat, static, met, sw_in_pot, fp, mask) - this class inserts
    lagged_met right after sw_in_pot, giving
    (sat, static, met, sw_in_pot, lagged_met, fp, mask), ypack.

    IMPORTANT - units: `lags` in variable_lags are counted in *rows of
    met_array*, i.e. in whatever step size self.time actually is (half-
    hourly for a typical FLUXNET-derived datetime_utc axis) - NOT
    necessarily hours, despite variable names like "24" suggesting a day.
    Verify this matches your intent; multiply lag values by 2 if you meant
    hours but the underlying axis is half-hourly.
    """

    def __init__(self, *args, lagged_met_vars, variable_lags, lag_window=None, **kwargs):
        self.lagged_met_vars = lagged_met_vars
        self.variable_lags = variable_lags
        # Derived automatically rather than kept as a separately-hardcoded
        # constant that has to be manually kept in sync with variable_lags
        # (exactly the kind of "two numbers must agree" drift this project
        # keeps running into) - lag_window is now just "how far back do we
        # need full history," computed from the lags actually requested.
        self.lag_window = lag_window if lag_window is not None else max(
            max(lags) for lags in variable_lags.values()
        )
        super().__init__(*args, **kwargs)

        # Exclude any requested sample lacking full lag history, instead of
        # just warning about it - a NaN row here would otherwise silently
        # corrupt the whole batch's loss/gradient during training.
        valid_mask = ~np.isnan(self.lagged_met_array).any(axis=(1, 2))
        n_before = len(self.indexes)
        self.indexes = self.indexes[valid_mask[self.indexes]]
        self.n = len(self.indexes)
        n_dropped = n_before - self.n
        if n_dropped:
            import logging
            logging.getLogger("vprm_pipeline").warning(
                "Dropped %d/%d requested samples lacking full lag history "
                "(need >= %d steps of prior record) - these were within "
                "max(lags) of the start of the dataset.",
                n_dropped, n_before, self.lag_window,
            )
    
    def init_training_cache(self, sat_vars, met_vars, met_dim=1):
        super().init_training_cache(sat_vars, met_vars, met_dim)

        missing = set(self.lagged_met_vars) - set(met_vars)
        if missing:
            raise ValueError(
                f"lagged_met_vars {sorted(missing)} not present in met_vars {met_vars} - "
                f"every lagged variable must also be fetched as a regular met_var."
            )

        n_lags_per_var = {v: len(self.variable_lags[v]) for v in self.lagged_met_vars}
        n_lags = next(iter(n_lags_per_var.values()))
        if len(set(n_lags_per_var.values())) != 1:
            raise ValueError(
                f"All lagged_met_vars must have the same number of lags (Conv1D needs a "
                f"fixed sequence length) - got {n_lags_per_var}."
            )

        n_times = len(self.time)
        n_vars = len(self.lagged_met_vars)
        met_flat = self.met_array[:, 0, 0, :]

        lagged = np.full((n_times, n_lags, n_vars), np.nan, dtype=np.float32)
        for v_idx, var in enumerate(self.lagged_met_vars):
            var_col = met_flat[:, met_vars.index(var)]
            for lag_idx, lag in enumerate(self.variable_lags[var]):
                if lag <= 0:
                    raise ValueError(f"Lags must be positive (steps into the past); got {lag} for {var}.")
                lagged[lag:, lag_idx, v_idx] = var_col[: n_times - lag]

        self.lagged_met_array = lagged

        n_nan_rows = np.isnan(lagged).any(axis=(1, 2)).sum()
        if n_nan_rows:
            import logging
            logging.getLogger("vprm_pipeline").warning(
                "%d of %d timesteps have incomplete lag history (NaN) - make sure your "
                "train/val time selections start at least max(lags)=%d steps into the "
                "record, or these rows will feed NaN into the Reco branch.",
                n_nan_rows, n_times, max(max(v) for v in self.variable_lags.values()),
            )

    def __getitem__(self, batch_index):
        (sat, static, met, sw_in_pot, fp, mask), ypack = super().__getitem__(batch_index)
        batch_idxs = self.indexes[
            batch_index * self.batch_size: (batch_index + 1) * self.batch_size
        ]
        lagged_met = self.lagged_met_array[batch_idxs]
        return (sat, static, met, sw_in_pot, lagged_met, fp, mask), ypack


class pyvprnn_v2(pyvprnn_v1):
    """
    pyvprnn_v1 plus a lagged-history encoder feeding the Reco branch only.
    """

    DEFAULT_VARIABLE_LAGS = {
        "t2m": [2, 6, 12, 24, 48],
        "swvl1_era5": [6, 12, 24, 72, 168],
        "stl2_era5": [3, 6, 12, 24, 48],   # single soil temperature level - dropped stl1_era5 entirely
    }
    DEFAULT_LAGGED_MET_VARS = ["t2m", "swvl1_era5", "stl2_era5"]
    
    def __init__(self, lagged_met_vars=None, variable_lags=None, lag_window=None, **kwargs):
        super().__init__(**kwargs)

        self.met_vars = ["t2m", "ssrd", "RH_from_VDP", "swvl1_era5", "swvl2_era5",
                         "hours_since_rain", 'precip_sum_1', 'precip_sum_4', 'precip_sum_2976']
        self.gpp_met_vars = ["t2m", "ssrd", "RH_from_VDP", "swvl2_era5"]
        self.reco_met_vars = ["t2m", "RH_from_VDP", "swvl1_era5", "hours_since_rain",'precip_sum_1' , 'precip_sum_4', 'precip_sum_2976']
        
        self.lagged_met_vars = list(lagged_met_vars) if lagged_met_vars is not None else list(self.DEFAULT_LAGGED_MET_VARS)
        self.variable_lags = dict(variable_lags) if variable_lags is not None else dict(self.DEFAULT_VARIABLE_LAGS)
        self.lag_window = lag_window if lag_window is not None else max(
            max(lags) for lags in self.variable_lags.values()
        )
    
        # Anything referenced by lagged_met_vars must be fetched into
        # met_array for LaggedBatchGenerator to read its history from - stl2_era5
        # isn't part of pyvprnn_v1's base met_vars (v1 has no lag mechanism to
        # use it), so extend it here, for v2 specifically, instead of making
        # v1 fetch a variable it would never read.
        for v in self.lagged_met_vars:
            if v not in self.met_vars:
                self.met_vars.append(v)
    
        unknown = set(self.lagged_met_vars) - set(self.variable_lags)
        if unknown:
            raise ValueError(f"lagged_met_vars {sorted(unknown)} have no entry in variable_lags.")

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
            variable_lags=self.variable_lags,
            lag_window=self.lag_window,
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
        # Day/night mask - from SW_IN_POT, not ssrd (see BatchGenerator note)
        # =========================================================
        sw_in_pot_bc = BroadcastToImage(name="sw_in_pot_bc")([self.sw_in_pot_input, sat_static])
        day_mask = DayMask(0, name="day_mask")(sw_in_pot_bc)

        # =========================================================
        # Lagged-history encoder (Reco branch only)
        # =========================================================
        x_met_lagged = layers.Conv1D(
            filters=16, kernel_size=4, padding="causal",
            activation="softplus", kernel_initializer="he_normal",
            name="conv1d_lagged_1",
        )(self.met_input_lagged)
        x_met_lagged = layers.Conv1D(
            filters=16, kernel_size=4, padding="causal",
            activation="softplus", kernel_initializer="he_normal",
            name="conv1d_lagged_2",
        )(x_met_lagged)
        x_met_lagged = layers.Dense(3, activation="softplus", name="dense_lagged_summary")(x_met_lagged[:, -1, :])
        met_bc_lagged = BroadcastToImage(name="met_bc_lagged")([x_met_lagged, sat_static])

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
        # Reco branch - now also sees the lagged-history encoding
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
            inputs=[self.sat_input, self.static_input, self.met_input, self.sw_in_pot_input,
                    self.met_input_lagged, self.flux_mask_input],
            outputs=[self.model.get_layer("gpp_map").output, self.model.get_layer("reco_map").output],
            name="pixel_flux_model",
        )
        self.pixel_model.save(save_path_model)