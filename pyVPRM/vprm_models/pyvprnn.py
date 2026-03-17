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
from pyVPRM.lib.functions import sel_nearest_valid, central_nxn_mean
from pyVPRM.lib.fancy_plot import *

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
    def __init__(self, ):
        return

    def set_ds(self, ds):
        self.ds = ds

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
        self.ds['min_evi'] =self.vprm_pre.min_max_evi.sat_img['min_evi']
        self.ds['max_evi'] = self.vprm_pre.min_max_evi.sat_img['max_evi']
        self.ds['th'] = self.vprm_pre.min_max_evi.sat_img['th']
        self.ds['min_lswi'] =  self.vprm_pre.min_lswi.sat_img['min_lswi']
        self.ds['max_lswi'] =  self.vprm_pre.max_lswi.sat_img['max_lswi']

        flux_tower_keys = flux_tower.flux_data.keys()
        # ['t2m', 'ssrd', 'ZL', 'FETCH_90', 'NEE_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_DT_VUT_REF',
        #  'NEE_VUT_REF_QC', 'GPP_NT_VUT_REF', 'RECO_NT_VUT_REF']
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
        
        if 'GPP_DT_VUT_REF' in list(self.ds.keys()):
            gpp_key = 'GPP_DT_VUT_REF'
        elif 'GPP_NT_VUT_REF' in list(self.ds.keys()):
            gpp_key = 'GPP_NT_VUT_REF'
        else:
            print('No GPP variable available')
        
        mask = (
            (self.ds["NEE_VUT_REF_QC"] < 2) &
            (self.ds["ZL"] > -1000))
        
        footprint_timestamps = (
            self.ds["datetime_utc"]
            .where(mask, drop=True))

        self.ffp_handler.set_timestamps(footprint_timestamps)
        self.ffp_handler.make_calculation_grid()
        self.ffp_handler.calculate_footprints()
        self.ffp_handler.regrid_calculation_grid_to_satellite_grid(self.vprm_pre.sat_imgs.sat_img, base_path)
        
        self.ds['ffp_footprint'] = self.ffp_handler.footprint_on_satellite_grid['footprint']
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
        X_met,
        X_mask,
        feature_idx,
        condition_mask=None,
        normalize_ice_curves=False,
        n_points=20,
        subsample=None,
        random_state=42,
        output_var_idx=0,
        add_to_temp_range=0,
    ):
        """
        Compute ICE curves and PDP for one meteorological feature,
        optionally subsampling for plotting.
    
        Returns:
            f_values, ice, pdp, subsample_idx
        """
        if condition_mask is not None:
            X_met_c = X_met[condition_mask]
            X_sat_c = X_sat[condition_mask]
            X_mask_c = X_mask[condition_mask]
        else:
            X_met_c = X_met
            X_sat_c = X_sat
            X_mask_c = X_mask
    
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
        else:
            subsample_idx = np.arange(n_samples)
    
        f_min = X_met_c[:, feature_idx].min() - add_to_temp_range
        f_max = X_met_c[:, feature_idx].max() + add_to_temp_range
        f_values = np.linspace(f_min, f_max, n_points)
    
        ice = np.zeros((X_met_c.shape[0], n_points))
    
        for j, val in enumerate(f_values):
            X_met_tmp = X_met_c.copy()
            X_met_tmp[:, feature_idx] = val
            y_pred = self.pixel_model.predict([X_sat_c, X_met_tmp[:,np.newaxis, np.newaxis,:], X_mask_c], verbose=0)[output_var_idx].squeeze() # 
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
        out_path=''
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
            linewidth=3,
            label="Median PDP",
            zorder=10)
    
        # -------------------------
        # Colorbar
        # -------------------------
        if color_var is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("EVI")
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(title)
        ax.grid(True)
        ax.legend()
        if out_path != '':  
            fig.savefig(out_path, dpi=300,
                bbox_inches="tight")
        
    
      
          
