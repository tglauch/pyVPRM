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
    def __init__(self, vprm_pre=None, met=None, footprint=None,
                 flux_tower=None, ffp_handler, met_keys=[]):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.met_keys = met_keys
        self.footprint= footprint
        self.flux_tower = flux_tower
        self.ffp_handler = ffp_handler
        return

    def get_training_data(self):
        ds = vprm_pre.sat_imgs.sat_img.drop(['scl', 'ndvi']).drop(['time'])
        ds = ds.assign_attrs(crs=ds.rio.crs)
        ds['min_evi'] = vprm_pre.min_max_evi.sat_img['min_evi']
        ds['max_evi'] = vprm_pre.min_max_evi.sat_img['max_evi']
        ds['th'] = vprm_pre.min_max_evi.sat_img['th']
        ds['min_lswi'] =  vprm_pre.min_lswi.sat_img['min_lswi']
        ds['max_lswi'] =  vprm_pre.max_lswi.sat_img['max_lswi']
        flux_tower_keys = ['t2m', 'ssrd', 'ZL', 'FETCH_90',
                          'NEE_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_DT_VUT_REF',
                           'NEE_VUT_REF_QC', 'GPP_NT_VUT_REF',
                           'RECO_NT_VUT_REF']
        for key in flux_tower_keys:
            try:
                ds[key] = xr.DataArray(
                    flux_tower_inst.flux_data[key].values,
                    dims=('datetime_utc',),
                    coords={
                        'datetime_utc': flux_tower_inst.flux_data['datetime_utc']
                    },
                    attrs={'units': 'K', 'long_name': '2m air temperature'}
                )
            except:
                print('Problem with {}'.format(key))
                continue
        
        t0 = np.datetime64(vprm_pre.timestamp_start)
        ds = ds.assign_coords(
            days_since_t0=(
                "datetime_utc",
                ((ds.datetime_utc.data - t0) / np.timedelta64(1, "D")).astype(int)))
        
        if 'GPP_DT_VUT_REF' in list(ds.keys()):
            gpp_key = 'GPP_DT_VUT_REF'
        elif 'GPP_NT_VUT_REF' in list(ds.keys()):
            gpp_key = 'GPP_NT_VUT_REF'
        else:
            print('No GPP variable available')
        
        mask = (
            (ds["NEE_VUT_REF_QC"] < 2) &
            (ds["ZL"] > -1000))
        
        footprint_timestamps = (
            ds["datetime_utc"]
            .where(mask, drop=True))
        
        self.ffp_handler.make_calculation_grid()
        self.ffp_handler.calculate_footprints()
        self.ffp_handler.regrid_calculation_grid_to_satellite_grid(handler.sat_img, base_path)
        
        ds['ffp_footprint'] = self.ffp_handler.footprint_on_satellite_grid['footprint']
        ds['land_cover_map'] = vprm_pre.land_cover_type.sat_img
        
        meteo_vars = {'ssrd': lambda x: x/3600,
                      't2m': lambda x: x - 273.15,
                      'skt': lambda x: x - 273.15,
                      'swvl1': None,
                      'swvl2': None,
                      'e':  lambda x: x * 1e4,
                      'd2m': lambda x: x - 273.15,
                      'stl1': lambda x: x - 273.15,
                      'stl2': lambda x: x - 273.15,}
        
        PAT = 'edh_pat_5279479e4fadb6e2e40eefb6968120720ee447ad535d04ded458097df3d5bfc2757f063e9b05130459593d516b308c79'
        inst = met_data_handler(PAT=PAT, keys=list(meteo_vars.keys()),
                                lat_slice=[flux_tower_inst.lat-0.2, flux_tower_inst.lat+0.2],
                                lon_slice=[flux_tower_inst.lon-0.2, flux_tower_inst.lon+0.2])
        inst.reduce_time(flux_tower_inst.flux_data['datetime_utc'].iloc[0],
                         flux_tower_inst.flux_data['datetime_utc'].iloc[-1])
        
        inst.ds_out = sel_nearest_valid(inst.ds_out, lon, lat) #inst.ds_out.sel({'lon': lon, 'lat': lat}, method='nearest') # .sel({'valid_time': ds['datetime_utc']}, method='nearest')
        for k in list(meteo_vars.keys()):
            if meteo_vars[k] is not None:
                inst.ds_out[k] = meteo_vars[k](inst.ds_out[k])
        
        es = calculate_saturation_vapor_pressure(inst.ds_out['t2m'])
        ea = calculate_actual_vapor_pressure(inst.ds_out['d2m'])
        inst.ds_out['vpd'] =  es - ea
        inst.ds_out['vpd_decorrelated'] = inst.ds_out['vpd']/es
        inst.ds_out['vpd_decorrelated_log'] = np.log(np.maximum(inst.ds_out['vpd'], 0.01)) - np.log(es)
        for key in inst.ds_out.keys():
            print(key)
            ds[key+'_era5'] = inst.ds_out[key].sel({'valid_time': ds['datetime_utc']}, method='nearest')
        
        all_meteo_vars = np.concatenate([['t2m', 'ssrd', 'NEE_VUT_REF', gpp_key, 'RECO_DT_VUT_REF'],
                                        [i+'_era5' for i in inst.ds_out.keys()]])
        meteos = ds[all_meteo_vars].sel({'datetime_utc': ds['t']})
        
        sat_vars = ['lswi','evi', 'nirv', 'ndre']
        ds[sat_vars] = (
            ds[sat_vars]
            .fillna(0.0)
        )
        ds["ffp_footprint"] = ds["ffp_footprint"].fillna(0.0)
        sat_imgs = ds[sat_vars].sel({'time_gap_filled': meteos['days_since_t0']})


    def make_sat_imgs_plot(self, ind_x, ind_y, opath=None):
        from pyVPRM.lib.fancy_plot import *
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        
        timestamp_start_np = np.datetime64(vprm_pre.timestamp_start)  # convert to numpy.datetime64
        days_array = vprm_pre.sat_imgs.sat_img['time_gap_filled'].values  # NumPy array
        
        plt_times = timestamp_start_np + np.timedelta64(1, 'D') * days_array
        
        plt_times_pre_smoothing = timestamp_start_np + np.timedelta64(1, 'D') * handler_pre_smoothing['time']
        # --- Extract values ---
        nirv_pre  = handler_pre_smoothing['nirv'].values[:, ind_x, ind_y]
        evi_pre   = handler_pre_smoothing['evi'].values[:, ind_x, ind_y]
        ndre_pre  = handler_pre_smoothing['ndre'].values[:, ind_x, ind_y]
        evi2_pre  = handler_pre_smoothing['evi2'].values[:, ind_x, ind_y]
        lswi_pre  = handler_pre_smoothing['lswi'].values[:, ind_x, ind_y]
        scl_vals  = handler_pre_smoothing['scl'].values[:, ind_x, ind_y]
        
        nirv_smooth = vprm_pre.sat_imgs.sat_img['nirv'].values[:, ind_x, ind_y]
        evi_smooth  = vprm_pre.sat_imgs.sat_img['evi'].values[:, ind_x, ind_y]
        ndre_smooth = vprm_pre.sat_imgs.sat_img['ndre'].values[:, ind_x, ind_y]
        evi2_smooth = vprm_pre.sat_imgs.sat_img['evi2'].values[:, ind_x, ind_y]
        lswi_smooth = vprm_pre.sat_imgs.sat_img['lswi'].values[:, ind_x, ind_y]
        
        # --- Discrete SCL colormap (0–11) ---
        cmap = plt.cm.get_cmap('tab20', 12)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, 12.5, 1), cmap.N)
        
        # --- Figure with 3 panels ---
        fig = plt.figure(figsize=figsize(1.0, 1.0))
        gs = gridspec.GridSpec(5, 2, width_ratios=[30, 1], wspace=0.05)
        
        axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
        cax = fig.add_subplot(gs[:, 1])  # colorbar spans all rows
        
        indices = [
            ("NIRv", nirv_pre, nirv_smooth),
            ("EVI",  evi_pre,  evi_smooth),
            ("NDRE", ndre_pre, ndre_smooth),
            ("EVI2", evi2_pre, evi2_smooth),
            ("LSWI", lswi_pre, lswi_smooth),
        ]
        
        for i, (ax, (name, raw_vals, smooth_vals)) in enumerate(zip(axes, indices)):
        
            sc = ax.scatter(
                plt_times_pre_smoothing,
                raw_vals,
                c=scl_vals,
                cmap=cmap,
                norm=norm,
                s=5,
                alpha=0.9
            )
        
            ax.plot(
                plt_times,
                smooth_vals,
                color='k',
                lw=1.5
            )
        
            ax.set_ylabel(name)
            # ax.set_ylim(-0.1, 1)
            ax.grid(alpha=0.3)
            # Only show x-axis tick labels for bottom panel
            if i < 3:
                ax.set_xticklabels([])
        
        axes[-1].set_xlabel("Time")
        
        # --- Shared colorbar ---
        cbar = fig.colorbar(sc, cax=cax, ticks=np.arange(12))
        cbar.set_label("SCL class")

        if opath is not None:
            fig.savefig(opath=None,dpi=300, bbox_inches='tight')

    

  
      
