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
from pyVPRM.lib.functions import sel_nearest_valid, central_nxn_mean, vpd_hpa_to_rh
from pyVPRM.lib.fancy_plot import *
import psutil
        
def log_mem(msg):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e9
    print(f"{msg}: {mem:.2f} GB")

def all_files_exist(item):
    for key in item.assets.keys():
        path = str(item.assets[key].href[7:])
        if not os.path.exists(path):
            print('{} does not exist. Skip'.format(path))
            return False
    return True

def calculate_saturation_vapor_pressure(temp):
    """
    Calculate the saturation vapor pressure (es) using the temperature (T) in Celsius.
    Return the result in kPa.
    """
    es_Pa = 6.1078*np.exp(17.27*temp/(temp + 237.3))
    return es_Pa/10

def calculate_actual_vapor_pressure(dew_temp):
    """
    Calculate the actual vapor pressure (ea) using the dew point temperature (Td) in Celsius.
    Return the result in kPa.
    """
    ea_Pa = 6.1078*np.exp(17.27*dew_temp/(dew_temp + 237.3))
    return ea_Pa/10


class pyvprnn:
    """
    Base class for all pyvprnn models
    """
    def __init__(self, lag_window=0):
        self.lag_window=lag_window
        return

    def set_ds(self, ds):
        self.ds = ds
    
    def prepare_base_dataset(self, nee_qc_flags=[0], opath=None,
                             mass_fraction_threshold=0.925,
                            chunks_t=256):
    

        log_mem('Beginning')
        if (not self.ds.chunks) and (not chunks_t == False):
            self.ds = self.ds.chunk({"t": chunks_t})
        ds = self.ds  # work on local reference
    

        ds["RH_from_VDP"] = vpd_hpa_to_rh(ds["VPD_F"], ds["t2m"]) / 100
        ds["nirv_90pct"] = ds["nirv"].quantile(0.9, dim="time_gap_filled")
        ds["nirv_10pct"] = ds["nirv"].quantile(0.1, dim="time_gap_filled")

        scl = ds["scl"]
        valid_classes = [4, 5, 6]
    
        scl_valid = scl.where(scl.isin(valid_classes))
    
        counts = xr.concat(
            [(scl_valid == c).sum(dim="time") for c in valid_classes],
            dim="class"
        ).assign_coords({"class": valid_classes})
    
        if "dominant_scl" not in ds:
            ds["dominant_scl"] = counts.idxmax(dim="class")
    
        ds["flux_mask"] = ds["land_cover_map"].sel(vprm_classes=7) < 0.99
        log_mem('flux_mask')
    
        self.ds_cropped = ds
        if mass_fraction_threshold is not None:        
            self.crop_to_mass_fraction(mass_fraction=mass_fraction_threshold)
        log_mem('cropped')
    
        ds_cropped = self.ds_cropped
    
        spatial_sum = ds_cropped["ffp_footprint"].sum(dim=["x", "y"])
        ds_cropped["ffp_footprint"] = ds_cropped["ffp_footprint"] / spatial_sum
    
        valid_footprint_mask = (
            ds_cropped["ffp_footprint"].sum(dim=["x", "y"], skipna=True) != 0
        )

        qc_mask = ds_cropped["NEE_VUT_REF_QC"].isin(nee_qc_flags)
        
        qc_mask = qc_mask.compute()   # <-- only small boolean vector
        valid_times = ds_cropped["datetime_utc"].where(qc_mask, drop=True)
        ds_cropped = ds_cropped.sel(datetime_utc=valid_times)

        log_mem('valid times selected')

        valid_fp = ds_cropped["ffp_footprint"]
        
        mask = valid_footprint_mask.compute()
        fp_times = ds_cropped["t"].values[mask.values]        
    
        common_times = valid_times.sel(
            datetime_utc=valid_times.isin(fp_times)
        )
    
        footprint_sum = (
            ds_cropped["ffp_footprint"]
            .sel(t=common_times)
            .sum(dim=["x", "y"])
        ).compute()
        

        common_times = common_times.where(footprint_sum > 1e-5, drop=True)
        self.common_times = np.sort(pd.to_datetime(common_times.values))
        if self.lag_window > 0:
            self.common_times = self.common_times[self.lag_window:]
    
        self.ds_cropped = ds_cropped.sel(datetime_utc=self.common_times)
        self.ds_cropped = self.ds_cropped.sel(t=self.common_times)
    
        counts = pd.Series(self.common_times).dt.year.value_counts().sort_index()
        print("Valid times per year:", counts)
        log_mem('End')
    
        return

    
    def prepare_base_dataset_old(self, nee_qc_flags=[0], opath=None,
                             mass_fraction_threshold=0.925):
        self.ds['nirv_90pct'] = self.ds['nirv'].quantile(0.9, dim='time_gap_filled')
        self.ds['nirv_10pct'] = self.ds['nirv'].quantile(0.1, dim='time_gap_filled')
        self.ds['RH_from_VDP']= vpd_hpa_to_rh(self.ds['VPD_F'], self.ds['t2m'])/100
        scl = self.ds["scl"]
        valid_classes = [4, 5, 6]
        # mask invalid classes
        scl_valid = scl.where(scl.isin(valid_classes))
        counts = xr.concat(
            [(scl_valid == c).sum(dim="time") for c in valid_classes],
            dim="class"
        )
        counts = counts.assign_coords({"class": valid_classes})
        if "dominant_scl" not in self.ds:
            dominant = counts.idxmax(dim="class")
            self.ds['dominant_scl'] = dominant
        self.ds["flux_mask"] = self.ds['land_cover_map'].sel({'vprm_classes': 7}) < 0.99

        if mass_fraction_threshold is not None:
            self.crop_to_mass_fraction(mass_fraction=mass_fraction_threshold)
        spatial_sum = self.ds_cropped['ffp_footprint'].sum(dim=['x', 'y'])
        self.ds_cropped['ffp_footprint'] = self.ds_cropped['ffp_footprint'] / spatial_sum

        valid_footprint_mask = (self.ds_cropped['ffp_footprint'].sum(dim=['x', 'y'], skipna=True) != 0)
        vars_datetime = [
            v for v in self.ds.data_vars
            if "datetime_utc" in self.ds[v].dims]
        
        valid_mask = (self.ds_cropped["NEE_VUT_REF_QC"].isin(nee_qc_flags))
        valid_times = self.ds_cropped.datetime_utc.where(valid_mask, drop=True)
        self.ds_cropped = self.ds_cropped.sel(datetime_utc=valid_times)
        
        common_times = valid_times.sel(
            datetime_utc=valid_times.isin(self.ds_cropped['ffp_footprint'].where(valid_footprint_mask, drop=True)['t']))
        
        footprint_sum = (
            self.ds_cropped['ffp_footprint']
            .sel(t=common_times)
            .sum(dim=['x', 'y']))
        
        common_times = common_times.where(footprint_sum > 1e-5, drop=True)
        self.common_times = pd.to_datetime(common_times.values)
        self.common_times = np.sort(self.common_times)
        if self.lag_window > 0:
            self.common_times = self.common_times[self.lag_window:]
        
        counts = pd.Series(self.common_times).dt.year.value_counts().sort_index()
        self.ds_cropped = self.ds_cropped.sel(datetime_utc=self.common_times)
        self.ds_cropped = self.ds_cropped.sel(t=self.common_times)
        print('Valid times per year:', counts)
        return

    def split_train_val_test(self, k=5, val_frac=0.15, 
                             test_frac=None, random_state=42):
        """
        K-fold random (non-daywise) cross-validation:
            - Randomly shuffles timestamps
            - Splits them into k equally sized test folds
            - For each fold:
                 - test = fold f
                 - remaining = train+val
                 - val_frac of remaining -> validation set
                 - the rest -> training set
        
        Output format matches daywise K-fold:
            self.cv_folds = [
                { "train_times", "val_times", "test_times" },
                ...
            ]
        """

        times = np.array(self.common_times)
        rng = np.random.default_rng(random_state)
    
        # --- special case: no CV, just one split ---
        if k == 1:
            times_shuffled = rng.permutation(times)
            n = len(times_shuffled)
    
            # default: same size as one CV fold
            if test_frac is None:
                test_frac = 1 / 5  # mimic k=5 default
    
            n_test = int(n * test_frac)
            n_val = int((n - n_test) * val_frac)
    
            test_times = np.sort(times_shuffled[:n_test])
            val_times = np.sort(times_shuffled[n_test:n_test + n_val])
            train_times = np.sort(times_shuffled[n_test + n_val:])
    
            self.cv_folds = [
                dict(
                    train_times=train_times,
                    val_times=val_times,
                    test_times=test_times,
                    train_weights=np.ones(len(train_times)),
                    val_weights=np.ones(len(val_times)),
                )
            ]
    
            return
        times = np.array(self.common_times)
    
       # --- original k-fold logic ---
        times_shuffled = rng.permutation(times)
        n = len(times_shuffled)
    
        # --- partition indices into k folds (balanced) ---
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[: n % k] += 1
    
        folds_idx = []
        start = 0
        for fs in fold_sizes:
            folds_idx.append(times_shuffled[start : start + fs])
            start += fs
    
        # --- generate cv_folds in same API as daywise kfold ---
        self.cv_folds = []
    
        for i in range(k):
            test_times = np.sort(folds_idx[i])
    
            # remaining timestamps (train + val)
            remaining = np.concatenate([folds_idx[j] for j in range(k) if j != i])
    
            # shuffle remaining for val split
            rem_shuffled = rng.permutation(remaining)
    
            n_rem = len(rem_shuffled)
            n_val = int(n_rem * val_frac)
    
            val_times   = np.sort(rem_shuffled[:n_val])
            train_times = np.sort(rem_shuffled[n_val:])
    
            self.cv_folds.append(
                dict(
                    train_times=train_times,
                    val_times=val_times,
                    test_times=test_times,
                    train_weights=np.ones(len(train_times)),
                    val_weights=np.ones(len(val_times)),
                )
            )
    
        return

    def split_train_val_test_daywise(
            self,
            k=5,
            val_frac=0.15,
            test_frac=None,
            block_days=1,
            shuffle=True,
            random_state=42
        ):
        """
        K-fold daywise cross-validation with separate validation folds.
    
        For each fold f:
            - test blocks = fold f
            - remaining blocks split into train/val using val_frac
    
        Sets:
            self.cv_folds = [
                {
                    "train_times": np.array([...]),
                    "val_times":   np.array([...]),
                    "test_times":  np.array([...])
                },
                ...
            ]
        """

        def blocks_to_times(blocks):
            out = []
            for block in blocks:
                for d in block:
                    out.extend(day_to_times[d])
            return np.array(out, dtype="datetime64[ns]")

        if k == 1:
            rng = np.random.default_rng(random_state)
        
            times = pd.to_datetime(self.common_times)
            times = times.sort_values()
        
            dates = np.array([t.date() for t in times])
            unique_days = np.unique(dates)
        
            # --- blocks ---
            day_blocks = [
                unique_days[i:i + block_days]
                for i in range(0, len(unique_days), block_days)
            ]
            day_blocks = np.array(day_blocks, dtype=object)
        
            if shuffle:
                day_blocks = rng.permutation(day_blocks)
        
            # --- split into train/val/test ---
            n_blocks = len(day_blocks)
            if test_frac is None:     
                test_frac = 1 / 5
            n_test = max(1, int(test_frac * n_blocks))
            n_val  = max(1, int(val_frac * (n_blocks - n_test)))
        
            test_blocks = day_blocks[:n_test]
            val_blocks  = day_blocks[n_test:n_test + n_val]
            train_blocks = day_blocks[n_test + n_val:]
        
            # build mapping (same as your code)
            day_to_times = {}
            for t in times:
                d = t.date()
                day_to_times.setdefault(d, []).append(t)
        
            self.cv_folds = [
                dict(
                    train_times=np.sort(blocks_to_times(train_blocks)),
                    val_times=np.sort(blocks_to_times(val_blocks)),
                    test_times=np.sort(blocks_to_times(test_blocks)),
                    train_weights=np.ones(len(blocks_to_times(train_blocks))),
                    val_weights=np.ones(len(blocks_to_times(val_blocks))),
                )
            ]
            return
    
        # --- ensure datetime ---
        times = pd.to_datetime(self.common_times)
        times = times.sort_values()
    
        # --- extract unique days ---
        dates = np.array([t.date() for t in times])
        unique_days = np.unique(dates)
    
        # --- construct N-day blocks ---
        day_blocks = [
            unique_days[i:i + block_days]
            for i in range(0, len(unique_days), block_days)
        ]
        day_blocks = np.array(day_blocks, dtype=object)
    
        # --- shuffle blocks ---
        if shuffle:
            rng = np.random.default_rng(random_state)
            day_blocks = rng.permutation(day_blocks)
        else:
            rng = np.random.default_rng(random_state)  # still needed for val split
    
        # --- split blocks into k test-folds ---
        n_blocks = len(day_blocks)
        fold_sizes = np.full(k, n_blocks // k, dtype=int)
        fold_sizes[:n_blocks % k] += 1
    
        block_folds = []
        start = 0
        for fs in fold_sizes:
            block_folds.append(day_blocks[start:start + fs])
            start += fs
    
        # --- map day → timestamps ---
        day_to_times = {}
        for t in times:
            d = t.date()
            day_to_times.setdefault(d, []).append(t)
    
    
        # --- build all folds ---
        self.cv_folds = []
    
        for i in range(k):
            # Test blocks for this fold
            test_blocks = block_folds[i]
    
            # Remaining blocks for train+val
            remaining_blocks = np.concatenate(
                [block_folds[j] for j in range(k) if j != i]
            )
    
            # --- validation split inside remaining blocks ---
            n_rem = len(remaining_blocks)
            n_val = int(val_frac * n_rem)
    
            # shuffle remaining blocks for val split
            rem_shuffled = rng.permutation(remaining_blocks)
            val_blocks = rem_shuffled[:n_val]
            train_blocks = rem_shuffled[n_val:]
    
            # --- convert to timestamps ---
            fold = dict(
                train_times=np.sort(blocks_to_times(train_blocks)),
                val_times=np.sort(blocks_to_times(val_blocks)),
                test_times=np.sort(blocks_to_times(test_blocks)),
                train_weights=np.ones(len(blocks_to_times(train_blocks))),
                val_weights=np.ones(len(blocks_to_times(val_blocks))),
            )
    
            self.cv_folds.append(fold)
        return
    
    def split_train_val_test_years(self, num_test_years=1, test_year=None,
                                   val_size=0.2, random_state=42, max_factor=5):
        """
        Split timestamps into train, validation, and test sets.
        
        - Test set is based on full years (no leakage).
        - Train/validation split is by unique days within train years.
        - Oversampling is applied independently to train and validation sets.
        """
        times = pd.to_datetime(self.common_times)
        years = np.unique(times.year)
        if test_year is not None:
            test_years = [y for y in years if y in test_year]
            train_years = [y for y in years if y not in test_years]
            print(train_years)
        else:
            test_years = years[-num_test_years:]
            train_years = [y for y in years if y not in test_years]
    
        # Split timestamps
        self.test_times = times[times.year.isin(test_years)]
        train_times = times[times.year.isin(train_years)]
    
        days = train_times.normalize()
        unique_days = np.unique(days)
    
        train_days, val_days = train_test_split(
            unique_days,
            test_size=val_size,
            random_state=random_state
        )
    
        self.train_times = np.sort(train_times[np.isin(days, train_days)])
        self.val_times = np.sort(train_times[np.isin(days, val_days)])
        return

    def balance_subset(self, times_subset, random_state=41):
        """
        Oversample times_subset to balance t2m.
        """
        # --- get t2m values at the given times ---
        t2m_vals = self.ds_cropped["t2m"].sel(datetime_utc=times_subset).values
    
        # --- make dataframe for balancing ---
        df = pd.DataFrame({
            "time": times_subset,
            "t2m": t2m_vals
        })
    
        # --- define bins for t2m ---
        bins_fixed = pd.cut(df["t2m"], bins=np.arange(-2, 31, 4))  # e.g., -2 to 30 °C in 4°C bins
        max_count = df.groupby(bins_fixed).size().max()
        max_factor = 2  # allow each row to appear up to 2x
    
        # --- oversample within bins ---
        df_balanced = (
            df.groupby(bins_fixed, group_keys=False)
              .apply(lambda x: x.sample(
                  min(len(x)*max_factor, max_count),
                  replace=True,
                  random_state=random_state))
        )
    
        # --- shuffle rows ---
        df_balanced = df_balanced.sample(frac=1, random_state=random_state)
    
        return df_balanced["time"].values

    def compute_sample_weights_t2m_vpd_global(
        self,
        times_train,
        times_apply,
        vpd_var='VPD_F',
        temp_var='t2m',
        n_bins=20,
        alpha=0.5,
        clip_max=5.0
    ):
        """
        Compute sample weights based on train distribution, apply to train or validation.
    
        Parameters
        ----------
        times_train : array-like
            Times to compute the reference rank/frequency (training set)
        times_apply : array-like
            Times to map and compute weights for (train or validation)
        """
        # --- extract values ---
        df_train = pd.DataFrame({
            "t2m": self.ds_cropped[temp_var].sel(datetime_utc=times_train).values,
            "vpd": self.ds_cropped[vpd_var].sel(datetime_utc=times_train).values
        })
        
        df_apply = pd.DataFrame({
            "t2m": self.ds_cropped[temp_var].sel(datetime_utc=times_apply).values,
            "vpd": self.ds_cropped[vpd_var].sel(datetime_utc=times_apply).values
        })
    
        # --- rank thresholds based on train ---
        t_edges = np.linspace(df_train["t2m"].min(), df_train["t2m"].max(), n_bins+1)
        vpd_edges = np.linspace(df_train["vpd"].min(), df_train["vpd"].max(), n_bins+1)
        
        # --- bin the train set to compute frequency ---
        t_bin_train = np.digitize(df_train["t2m"], bins=t_edges) - 1
        vpd_bin_train = np.digitize(df_train["vpd"], bins=vpd_edges) - 1
        joint_train = t_bin_train * 10000 + vpd_bin_train
        counts = pd.Series(joint_train).value_counts()
    
        # --- bin the apply set using same edges ---
        t_bin_apply = np.digitize(df_apply["t2m"], bins=t_edges) - 1
        vpd_bin_apply = np.digitize(df_apply["vpd"], bins=vpd_edges) - 1
        joint_apply = t_bin_apply * 10000 + vpd_bin_apply
    
        # --- inverse frequency weighting ---
        weights = pd.Series(joint_apply).map(lambda x: 1 / (counts.get(x, 1) ** alpha))

        weights = weights / weights.mean()
        weights = np.clip(weights, 0, clip_max)
        return weights.values
    
    def calculate_train_val_weights(self, alpha=0.5, clip_max=5.0):
        """
        Apply oversampling independently to train and validation sets
        after day-level splitting.
        """
        self.train_weights = self.compute_sample_weights_t2m_vpd_global(
            times_train=self.train_times,
            times_apply=self.train_times,
            alpha=alpha,
            clip_max=clip_max,
        )

        print(self.train_weights.min(), self.train_weights.max())
        self.valid_weights = self.compute_sample_weights_t2m_vpd_global(
            times_train=self.train_times,  # same reference
            times_apply=self.val_times,
            alpha=alpha,
            clip_max=clip_max,
        )
        return
    
    def balance_train_val(self):
        """
        Apply oversampling independently to train and validation sets
        after day-level splitting.
        """
        self.train_times_balanced = self.balance_subset(self.train_times)
        self.valid_times_balanced = self.balance_subset(self.val_times)
        return

    def get_data_for_upscaling(self,vprm_pre=None, met=None, datetimes=None,
                               base_path=None, meteo_vars=None):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.ds = self.vprm_pre.sat_imgs.sat_img.drop(['time'])
        t0 = np.datetime64(self.vprm_pre.timestamp_start)
        self.ds = self.ds.assign_coords(
            days_since_t0=(
                "datetime_utc",
                ((datetimes - t0) / np.timedelta64(1, "D")).astype(int)))
        self.era5_inst.reduce_time(datetimes[0], datetimes[1])
        for k in list(meteo_vars.keys()):
            if meteo_vars[k] is not None:
                self.era5_inst.ds_out[k] = meteo_vars[k](self.era5_inst.ds_out[k])
        es = calculate_saturation_vapor_pressure(self.era5_inst.ds_out['t2m'])
        ea = calculate_actual_vapor_pressure(self.era5_inst.ds_out['d2m'])
        self.era5_inst.ds_out['vpd'] =  es - ea
        self.era5_inst.ds_out['vpd_decorrelated'] = self.era5_inst.ds_out['vpd']/es
        self.era5_inst.ds_out['vpd_decorrelated_log'] = np.log(np.maximum(self.era5_inst.ds_out['vpd'], 0.01)) - np.log(es)
        for key in self.era5_inst.ds_out.keys():
            print(key)
            self.ds[key+'_era5'] = self.era5_inst.ds_out[key].sel({'valid_time': dateteims}, method='nearest')
        sat_vars = ['lswi','evi', 'nirv', 'ndre']
        self.ds[sat_vars] = (
            self.ds[sat_vars]
            .fillna(0.0))
        self.ds.attrs["crs"] = self.ds.attrs["crs"].to_wkt()
        self.ds.to_netcdf(os.path.join(base_path, 'out.nc'))
        return

    def get_training_data(self, vprm_pre=None, met=None, footprint=None,
                 flux_tower=None, base_path=None, meteo_vars=None):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.flux_tower = flux_tower
        self.ffp_handler = footprint
        self.ds = self.vprm_pre.sat_imgs.sat_img.drop(['time'])
        self.ds = self.ds.assign_attrs(crs=self.ds.rio.crs)

        flux_tower_keys = flux_tower.flux_data.keys()
        for key in flux_tower_keys:
            try:
                self.ds[key] = xr.DataArray(
                    self.flux_tower.flux_data[key].values,
                    dims=('datetime_utc',),
                    coords={
                        'datetime_utc': self.flux_tower.flux_data['datetime_utc']
                    },
                    attrs={'units': 'K', 'long_name': '2m air temperature'}
                )
            except:
                print('Problem with {}'.format(key))
                continue
        
        t0 = np.datetime64(self.vprm_pre.timestamp_start)
        self.ds = self.ds.assign_coords(
            days_since_t0=(
                "datetime_utc",
                ((self.ds.datetime_utc.data - t0) / np.timedelta64(1, "D")).astype(int)))
        
        mask = (
         #   (self.ds["NEE_VUT_REF_QC"] < 2) &
            (self.ds["ZL"] > -1000))
        
        footprint_timestamps = (
            self.ds["datetime_utc"]
            .where(mask, drop=True))

        footprints = [] 
        for i, chunk_of_timestamps in enumerate(np.array_split(footprint_timestamps, 100)):
            print(i)
            self.ffp_handler.set_timestamps(chunk_of_timestamps)
            self.ffp_handler.make_calculation_grid()
            self.ffp_handler.calculate_footprints()
            self.ffp_handler.regrid_calculation_grid_to_satellite_grid(vprm_pre.sat_imgs.sat_img, base_path)
            footprints.append(self.ffp_handler.footprint_on_satellite_grid['footprint'].astype("float32"))
        
        self.ds['ffp_footprint'] = xr.concat(footprints, dim='t')
        self.ds['land_cover_map'] = self.vprm_pre.land_cover_type.sat_img
        
        self.era5_inst.reduce_time(self.flux_tower.flux_data['datetime_utc'].iloc[0],
                         self.flux_tower.flux_data['datetime_utc'].iloc[-1])
        
        # self.era5_inst.ds_out = sel_nearest_valid(self.era5_inst.ds_out,
        #                                           flux_tower.lon, flux_tower.lat) 
        for k in list(meteo_vars.keys()):
            if meteo_vars[k] is not None:
                self.era5_inst.ds_out[k] = meteo_vars[k](self.era5_inst.ds_out[k])
        
        es = calculate_saturation_vapor_pressure(self.era5_inst.ds_out['t2m'])
        ea = calculate_actual_vapor_pressure(self.era5_inst.ds_out['d2m'])
        self.era5_inst.ds_out['vpd'] =  es - ea
        self.era5_inst.ds_out['vpd_decorrelated'] = self.era5_inst.ds_out['vpd']/es
        self.era5_inst.ds_out['vpd_decorrelated_log'] = np.log(np.maximum(self.era5_inst.ds_out['vpd'], 0.01)) - np.log(es)
        for key in self.era5_inst.ds_out.keys():
            print(key)
            self.ds[key+'_era5'] = self.era5_inst.ds_out[key].sel({'valid_time': self.ds['datetime_utc']}, method='nearest')
            
        sat_vars = ['lswi','evi', 'nirv', 'ndre']
        self.ds[sat_vars] = (
            self.ds[sat_vars]
            .fillna(0.0))
        self.ds["ffp_footprint"] = self.ds["ffp_footprint"].fillna(0.0)
        self.ds.attrs["crs"] = self.ds.attrs["crs"].to_wkt()
        self.ds.attrs["site"] = flux_tower.site_name
        self.ds.attrs["site_lat"] = flux_tower.lat    
        self.ds.attrs["site_lon"] = flux_tower.lon  
        self.ds.to_netcdf(os.path.join(base_path, 'out.nc'))
        return

    def crop_to_mass_fraction(
        self,
        var="ffp_footprint",
        time_dim="t",
        x_dim="x",
        y_dim="y",
        mass_fraction=0.99,
    ):
        """
        Memory-safe cropping to approximate mass_fraction bounding box.
        Avoids global sorting and large materialization.
        """
    
        import numpy as np
        import xarray as xr
        import numpy as np
        import pandas as pd
        import psutil, os
        
        def log_mem(msg):
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1e9
            print(f"{msg}: {mem:.2f} GB")
        print(self.ds[var].data)
        print(self.ds[var].chunks)
        # --- 1) Sum over time (lazy) ---
        log_mem('before foot')
        foot = self.ds[var].sum(dim=time_dim)
    
        # --- 2) total mass (scalar only → safe to compute) ---
        log_mem('Before Compute')
        total_mass = foot.sum().compute()
        log_mem('After Compute')
        threshold = mass_fraction * total_mass
    
        # --- 3) normalize (still lazy) ---
        foot_norm = foot / total_mass
    
        # --- 4) threshold small values (no sorting!) ---
        # heuristic threshold (tune if needed)
        mask = foot_norm > 1e-6
    
        # --- 5) reduce to 1D masks (still lazy) ---
        y_mask = mask.any(dim=x_dim)
        x_mask = mask.any(dim=y_dim)
    
        # --- 6) compute ONLY 1D boolean arrays ---
        log_mem('before y mask np')
        y_mask_np = y_mask.compute().values
        x_mask_np = x_mask.compute().values
        log_mem('after y mask np')
    
        # --- 7) get index ranges ---
        y_idx = np.where(y_mask_np)[0]
        x_idx = np.where(x_mask_np)[0]
    
        if len(y_idx) == 0 or len(x_idx) == 0:
            raise ValueError("Cropping failed: no valid footprint region found")
    
        ymin_i, ymax_i = y_idx.min(), y_idx.max()
        xmin_i, xmax_i = x_idx.min(), x_idx.max()
    
        # --- 8) use isel (cheap slicing, no copying) ---
        self.ds_cropped = self.ds.isel(
            {
                y_dim: slice(ymin_i, ymax_i + 1),
                x_dim: slice(xmin_i, xmax_i + 1),
            }
        )
    
        return
    

    def crop_to_mass_fraction_old(
        self,
        var="ffp_footprint",
        time_dim="t",
        x_dim="x",
        y_dim="y",
        mass_fraction=0.99,
    ):
        """
        Crop dataset to smallest rectangular region containing
        given fraction of total mass of `var` summed over time.
    
        Returns
        -------
        ds_cropped : xr.Dataset
        bbox : dict
        """
    
        # 1) Sum over time
        foot = self.ds[var].sum(dim=time_dim)
    
        # 2) Compute threshold
        total_mass = foot.sum()
        threshold = mass_fraction * total_mass
    
        # 3) Flatten and sort by contribution (descending)
        flat = foot.stack(z=(y_dim, x_dim))
        flat_sorted = flat.sortby(flat, ascending=False)
    
        # 4) Cumulative mass
        cumsum = flat_sorted.cumsum()
    
        # 5) Select cells contributing to mass_fraction
        mask_flat = cumsum <= threshold
    
        # Ensure we include the first cell exceeding threshold
        first_over = cumsum.where(cumsum > threshold, drop=True)
        if first_over.size > 0:
            mask_flat = mask_flat | (cumsum == first_over.min())
        mask = mask_flat.unstack("z")
        # 6) Find bounding indices (dimension-safe way)
        y_mask = mask.any(dim=x_dim)
        x_mask = mask.any(dim=y_dim)
        y_vals = foot[y_dim].where(y_mask, drop=True)
        x_vals = foot[x_dim].where(x_mask, drop=True)
    
        ymin = y_vals.min().item()
        ymax = y_vals.max().item()
        xmin = x_vals.min().item()
        xmax = x_vals.max().item()
        bbox = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
        # 7) Direction-aware slicing
        def directional_slice(coord, vmin, vmax):
            if coord[0] < coord[-1]:
                return slice(vmin, vmax)
            else:
                return slice(vmax, vmin)
    
        self.ds_cropped = self.ds.sel(
            {y_dim: directional_slice(self.ds[y_dim], ymin, ymax),
             x_dim: directional_slice(self.ds[x_dim], xmin, xmax)})

        return

    def clear_ds(self):
        del self.ds
        import gc
        gc.collect() 
        return
    
    def make_satellite_animation(self, opath=None):
        from pyproj import Transformer
        lon, lat = self.flux_tower.get_lonlat()
        transformer = Transformer.from_crs(
            "EPSG:4326",
            self.ds.rio.crs,   # or "EPSG:32632"
            always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        # -------------------------
        # Precompute limits once
        # -------------------------
        vmax = np.percentile(self.ds['evi'], 99.5)
        vmin = np.percentile(self.ds['evi'], 0.5)
        timestamp_start_np = np.datetime64(self.vprm_pre.timestamp_start)
        days_array = self.vprm_pre.sat_imgs.sat_img['time_gap_filled'].values
        plt_times = timestamp_start_np + np.timedelta64(1, 'D') * days_array
        
        ntime = self.ds.dims["time_gap_filled"]
        fig, ax = newfig(0.9, 0.7)
        
        xmin, ymin, xmax, ymax = self.ds.rio.bounds()
        im = ax.imshow(
            self.ds['evi'].isel(time_gap_filled=0).values,
            cmap="Greens",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=[xmin, xmax, ymin, ymax],
            animated=True
        )
        
        tower = ax.scatter(x_utm, y_utm, marker="*", color="k", zorder=3)
        title = ax.set_title(str(plt_times[0])[:10])
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("EVI")
        def update(frame):
            im.set_array(self.ds['evi'].isel(time_gap_filled=frame).values)
            title.set_text(str(plt_times[frame])[:10])
            return im, title
        ani = FuncAnimation(
            fig,
            update,
            frames=ntime,
            interval=150,     # ms between frames
            blit=True)
        
        if opath is not None:
            ani.save(
                opath,
                writer=PillowWriter(fps=6)
            )

    def partial_dependence_preprocessed_ice(
        self,
        X_sat,
        X_sat_static,
        X_met,
        X_mask,
        X_met_lagged,
        feature_idx,
        condition_mask=None,
        normalize_ice_curves=False,
        n_points=20,
        subsample=None,
        random_state=42,
        output_var_idx=0,
        add_to_temp_range_min=0,
        add_to_temp_range_max=0,
    ):
        """
        Compute ICE curves and PDP for one meteorological feature,
        optionally subsampling for plotting.
    
        Returns:
            f_values, ice, pdp, subsample_idx
        """
        t0 = time.time()
        if condition_mask is not None:
            X_met_c = X_met[condition_mask]
            X_sat_c = X_sat[condition_mask]
            X_sat_static_c = X_sat_static[condition_mask]
            X_mask_c = X_mask[condition_mask]
            if X_met_lagged is not None:
                X_met_lagged_c = X_met_lagged[condition_mask]
        else:
            X_met_c = X_met
            X_sat_c = X_sat
            X_sat_static_c = X_sat_static
            X_mask_c = X_mask
            if X_met_lagged is not None:
                X_met_lagged_c = X_met_lagged
    
        n_samples = X_met_c.shape[0]
        # -------------------------
        # Subsampling
        # -------------------------
        if subsample is not None and subsample < n_samples:
            rng = np.random.default_rng(random_state)
            subsample_idx = rng.choice(n_samples, subsample, replace=False)
            X_met_c = X_met_c[subsample_idx]
            X_sat_c = X_sat_c[subsample_idx]
            X_mask_c = X_mask_c[subsample_idx]
            X_sat_static_c = X_sat_static_c[subsample_idx]
            if X_met_lagged is not None:
                X_met_lagged_c = X_met_lagged_c[subsample_idx]
        else:
            subsample_idx = np.arange(n_samples)
    
        f_min = X_met_c[:, feature_idx].min() - add_to_temp_range_min
        f_max = X_met_c[:, feature_idx].max() + add_to_temp_range_max
        f_values = np.linspace(f_min, f_max, n_points)
    
        ice = np.zeros((X_met_c.shape[0], n_points))

        for j, val in enumerate(f_values):
            X_met_tmp = X_met_c.copy()
            X_met_tmp[:, feature_idx] = val
            if X_met_lagged is not None:
                y_pred = self.pixel_model.predict([X_sat_c, X_met_tmp[:,np.newaxis, np.newaxis,:],
                                                   X_met_lagged_c, X_mask_c], verbose=0)[output_var_idx].squeeze() # 
            else:
                y_pred = self.pixel_model.predict([X_sat_c, X_sat_static_c,
                                                   X_met_tmp[:,np.newaxis, np.newaxis,:], X_mask_c],
                                                  verbose=0)[output_var_idx].squeeze() # 
            ice[:, j] = y_pred
    
        # Normalize ICE curves (optional)
        if normalize_ice_curves:
            for i in range(ice.shape[0]):
                ice[i, :] = ice[i, :] / np.max(ice[i, :])
    
        pdp = np.median(ice, axis=0)
    
        return f_values, ice, pdp, subsample_idx
    

    def plot_ice_pdp(
        self,
        f_values,
        ice,
        pdp,
        xlabel,
        ylabel,
        title,
        show_ices=True,
        show_band=False,
        color_var=None,  # now expects an array with same length as ice.shape[0]
        cmap="viridis",
        ice_alpha=0.6,
        out_path='',
        ax=None,
        fig=None,
    ):
        """
        Plot ICE curves and PDP, optionally coloring ICE curves by a variable.
    
        Parameters
        ----------
        f_values : np.ndarray
            Feature grid (x-axis)
        ice : np.ndarray
            ICE curves, shape (n_samples, n_points)
        pdp : np.ndarray
            Median PDP curve
        color_var : np.ndarray, optional
            Array of same length as n_samples to color each ICE line
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
    
        # -------------------------
        # Determine colors
        # -------------------------
        if color_var is not None:
            if len(color_var) != ice.shape[0]:
                raise ValueError(
                    f"color_var must have length {ice.shape[0]}, got {len(color_var)}"
                )
            vmin = np.nanpercentile(color_var, 5)
            vmax = np.nanpercentile(color_var, 95)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colors = plt.cm.get_cmap(cmap)(norm(color_var))
        else:
            colors = ["gray"] * ice.shape[0]
    
        # -------------------------
        # ICE curves
        # -------------------------
        
        if show_ices:
            for i in range(ice.shape[0]):
                ax.plot(
                    f_values,
                    ice[i],
                    color=colors[i],
                    linewidth=0.4,
                    alpha=ice_alpha)
                
        if show_band:
                p5  = np.nanpercentile(ice, 5, axis=0)
                p95 = np.nanpercentile(ice, 95, axis=0)
                
                # Plot shaded uncertainty band
                ax.fill_between(
                    f_values,
                    p5,
                    p95,
                    color="gray",
                    alpha=0.3,
                    label="5–95% ICE range")
    
        # -------------------------
        # PDP
        # -------------------------
        ax.plot(
            f_values,
            pdp,
            color="black",
            linewidth=1,
            label="Median PDP",
            zorder=10)
    
        # -------------------------
        # Colorbar
        # -------------------------
        if color_var is not None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cax = inset_axes(
                ax,
                width="70%",
                height="30%",
                loc="upper center",
                bbox_to_anchor=(0.0, 1.01, 1.0, 0.1), 
                bbox_transform=ax.transAxes,
                borderpad=0
            )
            
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            cbar.set_label("EVI", labelpad=5)
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(title)
        ax.grid(True)
        # ax.legend()
        if (out_path != '') & (fig != None):  
            fig.savefig(out_path, dpi=300,
                bbox_inches="tight")
        return ax
        
    
      
          
