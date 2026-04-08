import xarray as xr
import glob
import os
import time
import numpy as np
from dateutil import parser
import pygrib
import copy
import uuid
import datetime
import pandas as pd
import itertools
from loguru import logger
from pyVPRM.vprm_models.pyvprnn import pyvprnn
from pyVPRM.lib.fancy_plot import figsize
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from pyproj import Transformer
from pyVPRM.lib.functions import sel_nearest_valid

import tensorflow as tf

class BroadcastToImage(tf.keras.layers.Layer):
    def call(self, inputs):
        m, ref = inputs
        m_shape = tf.shape(m)
        ref_shape = tf.shape(ref)
        
        if m.shape.rank == 2:
            m = tf.expand_dims(tf.expand_dims(m, axis=1), axis=1)  # (B,1,1,F)
            m = tf.tile(m, [1, ref_shape[1], ref_shape[2], 1])
        
        elif m.shape.rank == 4:
            tile_h = tf.math.floordiv(ref_shape[1], m_shape[1])  # dynamic division
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
        cfg = super().get_config()
        return cfg

class GlobalSumPooling(layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=[1, 2, 3])
    def get_config(self):
        cfg = super().get_config()
        return cfg

class DayMask(layers.Layer):
    def __init__(self, ssrd_idx, **kwargs):
        super().__init__(**kwargs)
        self.ssrd_idx = ssrd_idx

    def call(self, m):
        # m shape: (batch, y, x, n_met_features)
        ssrd = m[..., self.ssrd_idx]            # shape: (batch, y, x)
        mask = tf.cast(ssrd > 0, tf.float32)   # same shape
        return tf.expand_dims(mask, axis=-1)   # shape: (batch, y, x, 1)

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
        config = super().get_config()
        config.update({"indices": self.indices})
        return config

class GPPPenalty(layers.Layer):
    def __init__(self, threshold=40.0, weight=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.weight = weight

    def call(self, x):
        excess = tf.nn.relu(x - self.threshold)
        penalty = self.weight * tf.reduce_max(excess)
        self.add_loss(penalty)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "threshold": self.threshold,
            "weight": self.weight
        })
        return cfg

class pyvprnn_v1(pyvprnn):
    """
    Base class for all pyvprnn models
    """
    
    def __init__(self):
        super().__init__()
        self.train_weights = None
        self.valid_weights = None
        return

    def load_model(self, path):
        self.pixel_model = load_model(path,
        custom_objects = {
            "BroadcastToImage": BroadcastToImage,
            "ExpandLastDim": ExpandLastDim,
            "GlobalSumPooling": GlobalSumPooling,
            "DayMask": DayMask,
            "SelectFeatures": SelectFeatures,
            "GPPPenalty": GPPPenalty,
        })

    def make_veg_fraction_plot(self, opath):
        fracs = (self.ds_cropped['land_cover_map']*ds_cropped['ffp_footprint']).sum(dim=['x', 'y'])
        valid_footprint_mask = fracs.sum(dim='vprm_classes') != 0
        fig, axes = plt.subplots(9, 1, figsize=figsize(1.0, 1.4), sharex=True)
        classes = [
            (1, 'grey',   'EF'),
            (2, 'grey', 'DF'),
            (3, 'grey', 'MF'),
            (4, 'grey', 'SH'),
            (5, 'grey', 'SAV'),
            (6, 'grey', 'CRO'),
            (7, 'grey', 'GRA'),
            (8, 'grey', 'URB'),
            (9, 'grey', 'WET')]
        
        for i, (ax, (cls, color, label)) in enumerate(zip(axes, classes)):
            fracs.where(valid_footprint_mask, drop=True).sel({'vprm_classes': cls}).plot(ax=ax, color=color)
            ax.text(0.01, 0.99, label,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform = ax.transAxes,
                   color='red')
            ax.grid(alpha=0.3)
            ax.set_title('')
            if i !=8:
                ax.set_xlabel('')
        
        axes[-1].set_xlabel("Time")
        if opath is not None:
            fig.savefig(
                opath,
                dpi=300,
                bbox_inches="tight")
        return

    def build_nn_arrays(self, times, met_dim=1):
        import time
        t0 = time.time()
        self.met_vars = ["t2m", "ssrd", 'RH_from_VDP',  #
                         'swvl1_era5', 'swvl2_era5']  # , 'stl2_era5' # 'TS_F_MDS_1'
        # self.met_vars = ["t2m", "ssrd", 'VPD_F',
        #                  'swvl1_era5', 'swvl2_era5']
        
        # --- times as DataArray ---
        times = xr.DataArray(times, dims="datetime_utc")
        print('time xarray', time.time()-t0)
        met_stack_list = []

        if met_dim == 1:
            for var in self.met_vars:
                if var.endswith("_era5"):
                    selected = sel_nearest_valid(self.ds_cropped[[var]],
                        lon=self.ds_cropped.attrs['site_lon'],
                        lat=self.ds_cropped.attrs['site_lat'])[var].sel(datetime_utc=times)
                    arr = selected.values
                   #arr = self.ds_cropped[var].sel(datetime_utc=times).values
                else:
                    arr = self.ds_cropped[var].sel(datetime_utc=times).values
                    if var == "ssrd":
                        arr = arr / 1000
                    elif var in ['SWC_F_MDS_1', 'SWC_F_MDS_2']:
                        arr = arr / 10
                met_stack_list.append(arr)
                print(var, arr.min(), arr.max())
        elif met_dim ==2 :
            # --- transformer from ds CRS to WGS84 ---
            transformer = Transformer.from_crs(self.ds_cropped.crs, "EPSG:4326", always_xy=True)
            X, Y = np.meshgrid(self.ds_cropped.x.values, self.ds_cropped.y.values)
            lon, lat = transformer.transform(X, Y)
            lon_da = xr.DataArray(lon, dims=("y", "x"))
            lat_da = xr.DataArray(lat, dims=("y", "x"))
            for var in self.met_vars:
                if var.endswith("_era5"):
                    selected = self.ds_cropped[var].sel(
                        lon=lon_da,
                        lat=lat_da,
                        method="nearest"
                    ).sel(datetime_utc=times)
                    arr = selected.values
                    print(np.shape(arr))
                else:
                    arr_1d = self.ds_cropped[var].sel(datetime_utc=times).values
                    arr = np.broadcast_to(arr_1d[:, None, None], (len(times), len(self.ds_cropped.y), len(self.ds_cropped.x)))
                    if var == "ssrd":
                        arr = arr / 1000
                    elif var in ['SWC_F_MDS_1', 'SWC_F_MDS_2']:
                        arr = arr / 10
                print(var, arr.min(), arr.max())
                met_stack_list.append(arr)
        else:
            print('met_dim kwarg should be 1 or 2')
        print('Sel Mets', time.time()-t0)
        
        met_stack = np.stack(met_stack_list, axis=-1).astype(np.float32)
        if met_dim == 1:
            met_stack = met_stack[:,np.newaxis, np.newaxis, :]
        print(np.shape(met_stack))
        y_target = self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=times).values.astype(np.float32)
        y_unc = self.ds_cropped["NEE_VUT_REF_JOINTUNC"].sel(datetime_utc=times).values.astype(np.float32)
        print('Met stack', time.time()-t0)
    
        footprints = (
            self.ds_cropped["ffp_footprint"]
            .sel(t=times)
            .fillna(0.0)
            .values).astype(np.float32)

        print('Sel footprints', time.time()-t0)
        sat_vars = ['lswi','nirv','ndre']
        sat_imgs = self.ds_cropped[sat_vars].sel(
            {'time_gap_filled': self.ds_cropped.sel(datetime_utc=times)['days_since_t0']})
    
        sat_stack = np.stack([sat_imgs[v].values for v in sat_vars], axis=-1)
        print('Sat stack', time.time()-t0)
        lc = np.moveaxis(self.ds_cropped['land_cover_map'].sel({'vprm_classes': [1,2,3,4,5,6]}).values, 0, -1)
        print('lc', time.time()-t0)
        T = sat_stack.shape[0]
        lc_time = np.repeat(lc[None, ...], T, axis=0)
        nirv_max = np.repeat(self.ds_cropped['nirv_90pct'].values[None, ..., None], T, axis=0)
        nirv_min = np.repeat(self.ds_cropped['nirv_10pct'].values[None, ..., None], T, axis=0)
        sat_stack = np.concatenate([sat_stack, nirv_max, nirv_min, lc_time], axis=-1).astype(np.float32)
        print('repeat concat', time.time()-t0)
        print(np.shape(sat_stack))
        flux_mask = np.repeat(self.ds_cropped["flux_mask"].values[None, ...], T, axis=0).astype(np.float32)
        print('Time for input construction', time.time()-t0)
        return met_stack, sat_stack, footprints, flux_mask, y_target, y_unc 
         
    def train(self, save_path_model,
              save_path_history=None,
              train_params={'batch_size': 42,
                            'epochs': 1000,
                            'patience': 10,
                            'plateau_patience': 5,
                            'learning rate': 5e-4},
              cv_fold=0,
              random_state=41):

        train_times = self.cv_folds[cv_fold]['train_times']
        qc_train = self.ds_cropped["NEE_VUT_REF_QC"].sel(datetime_utc=train_times)
        wrong_nigttime_train = ((self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=train_times)<0) &\
                          (self.ds_cropped["ssrd"].sel(datetime_utc=train_times)==0))
        train_times_qc0 = train_times[(qc_train == 0) & ~wrong_nigttime_train]
        Xmet_train, Xsat_train, fp_train, flux_mask_train, y_train, y_unc_train = \
            self.build_nn_arrays(train_times_qc0, met_dim=1) # 

        val_times = self.cv_folds[cv_fold]['val_times']
        qc_val = self.ds_cropped["NEE_VUT_REF_QC"].sel(datetime_utc=val_times)
        wrong_nigttime_val = ((self.ds_cropped["NEE_VUT_REF"].sel(datetime_utc=val_times)<0) &\
                          (self.ds_cropped["ssrd"].sel(datetime_utc=val_times)==0))
        val_times_qc0 = val_times[(qc_val == 0)  & ~wrong_nigttime_val]
        Xmet_test, Xsat_test, fp_test, flux_mask_test, y_test, y_unc_test =\
            self.build_nn_arrays(val_times_qc0, met_dim=1) # 

        train_weights = self.cv_folds[cv_fold]['train_weights'][(qc_train == 0) & ~wrong_nigttime_train]
        val_weights = self.cv_folds[cv_fold]['val_weights'][(qc_val == 0)  & ~wrong_nigttime_val]
        
        # =========================================================
        # Indices
        # =========================================================
        
        n_sat_features = Xsat_train.shape[-1]
        n_met_features = Xmet_train.shape[-1]
        
        ssrd_idx = self.met_vars.index("ssrd")
        swvl2_idx = self.met_vars.index('swvl2_era5')
        swvl1_idx = self.met_vars.index('swvl1_era5')
      #  skt_idx = self.met_vars.index('stl2_era5') # "skt_era5"
        t2m_idx = self.met_vars.index("t2m")        
        
        # RECO must NOT see ssrd
        reco_met_idx = [i for i in range(n_met_features)
                        if i not in [ssrd_idx, swvl2_idx]] # swvl2_idx 
        
        gpp_met_idx = [i for i in range(n_met_features)
                        if i not in [swvl1_idx]] #skt_idx
        filter_size=1
        
        # =========================================================
        # Inputs
        # =========================================================
        
        self.sat_input = layers.Input(
            shape=(None, None, n_sat_features),
            name="sat")
        
        self.met_input = layers.Input(
            shape=(None, None, n_met_features),
            name="met")
        
        self.fp_input = layers.Input(
            shape=(None, None),
            name="fp")
        
        self.flux_mask_input = layers.Input(
            shape=(None, None),
            name="flux_mask")
        
        flux_mask_exp = ExpandLastDim(name="flux_mask_exp")(self.flux_mask_input)
        
        fp_exp = ExpandLastDim(name="fp_exp")(self.fp_input)
        
        # =========================================================
        # Meteorology branches
        # =========================================================
        
        met_gpp = SelectFeatures(
            gpp_met_idx,
            name="met_gpp")(self.met_input)
        met_bc_gpp = BroadcastToImage(name="met_bc_gpp")([met_gpp, self.sat_input])
        
        met_reco = SelectFeatures(
            reco_met_idx,
            name="met_reco")(self.met_input)
        met_bc_reco = BroadcastToImage(name="met_bc_reco")([met_reco, self.sat_input])
        
        # =========================================================
        # GPP branch
        # =========================================================
        
        x_gpp = layers.Concatenate(name="gpp_concat")([
            self.sat_input,
            met_bc_gpp])
        
        for i in range(6):
            x_gpp = layers.Conv2D(
                32, filter_size, padding="same",
                activation="softplus",
                kernel_initializer="he_normal")(x_gpp)
        
        x_gpp = layers.Conv2D(
            1, 1,
            activation="softplus",
            kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.3),
            name='x_gpp_map',)(x_gpp)

        x_gpp = GPPPenalty(
            threshold=40.0,
            weight=1e-4,
            name="gpp_penalty"
        )(x_gpp)
        
        day_mask = DayMask(ssrd_idx, name="day_mask")(met_bc_gpp)
        #day_mask_img = BroadcastToImage(name="day_mask_bc")([day_mask[:, None], sat_input])
        
        gpp_map_day_mask = layers.Multiply(name="gpp_map_masked")([
            x_gpp,
            day_mask])

        gpp_map = layers.Multiply(name="gpp_map")([
            gpp_map_day_mask,
            flux_mask_exp])
        
        gpp_weighted = layers.Multiply(name="gpp_weighted")([
            gpp_map,
            fp_exp])
        
        gpp_sum = GlobalSumPooling(name="gpp_sum_raw")(gpp_weighted)
        
        # ========================================================= #
        # RECO branch                                               #
        # ========================================================= #
        
        x_reco = layers.Concatenate(name="reco_concat")([
            self.sat_input,
            met_bc_reco])
        
        for i in range(6):
            x_reco = layers.Conv2D(
                32, filter_size, padding="same",
                activation="softplus",
                kernel_initializer="he_normal")(x_reco)
            
        x_reco_map = layers.Conv2D(
            1, 1,
            activation="softplus",
            kernel_initializer="he_normal",
            bias_initializer=tf.keras.initializers.Constant(0.7),
            name="x_reco_map")(x_reco)
        
        reco_map = layers.Multiply(name="reco_map")([
            x_reco_map,
            flux_mask_exp])
        
        reco_weighted = layers.Multiply(name="reco_weighted")([
            reco_map,
            fp_exp])
        
        reco_sum = GlobalSumPooling(name="reco_sum")(reco_weighted)
        
        # =========================================================
        # NEE (physics only)
        # =========================================================
        
        nee = layers.Subtract(name="nee")([
            reco_sum,
            gpp_sum])
        nee = layers.Flatten(name='Output')(nee) 
        
        # =========================================================
        # Model
        # =========================================================
        
        self.model = Model(
            inputs=[self.sat_input, self.met_input,
                    self.fp_input, self.flux_mask_input], #  
            outputs=nee)

        from tensorflow.keras.losses import Huber
        def nll_loss_from_stacked(y_with_sigma_true, y_pred):
            y_true = y_with_sigma_true[..., 0][..., None] 
            sigma = tf.maximum(y_with_sigma_true[..., 1], 0.1)[..., None] 
            return tf.reduce_mean((y_true - y_pred)**2 / (2 * sigma**2)+ 0.5 * tf.math.log(2 * np.pi * sigma**2))

        def mse_true_only(y_with_sigma_true, y_pred):
            y_true = y_with_sigma_true[..., 0][..., None]
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_params['learning rate']),
            loss=nll_loss_from_stacked, #'mse', #Huber(delta=5.0), #'mse'
            metrics=[mse_true_only])
        
        self.model.summary()
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=train_params['patience'],
            restore_best_weights=True)
        
        batch_size = train_params['batch_size']
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=train_params['plateau_patience'],
            min_lr=1e-6,
            verbose=1)

        print(np.min(y_unc_train), np.mean(y_unc_train), np.max(y_unc_train))
        y_train_with_sigma = np.stack([y_train, y_unc_train], axis=1)
        y_test_with_sigma = np.stack([y_test, y_unc_test], axis=1)

        history = self.model.fit(
            x=[Xsat_train, Xmet_train, fp_train, flux_mask_train],
            y=y_train_with_sigma, 
            validation_data=([Xsat_test, Xmet_test, fp_test, flux_mask_test],
                              y_test_with_sigma, val_weights), # _with_sigma
            epochs=train_params['epochs'],
            batch_size=batch_size,
            sample_weight=train_weights,
            callbacks=[early_stop, reduce_lr])
        
        best_val_loss = min(history.history["val_loss"])
        print("Best val_loss:", best_val_loss)
        
        # Convert to DataFrame
        hist_df = pd.DataFrame(history.history)
        
        # Add epoch column
        hist_df["epoch"] = range(1, len(hist_df) + 1)

        if save_path_history is not None:
            hist_df.to_csv(
                save_path_history,
                index=False)
        self.pixel_model = Model(
            inputs=[self.sat_input, self.met_input, self.flux_mask_input],
            outputs=[
                self.model.get_layer("gpp_map").output,
                self.model.get_layer("reco_map").output],
            name="pixel_flux_model")

        self.pixel_model.save(save_path_model)





    
        
    
      
          
