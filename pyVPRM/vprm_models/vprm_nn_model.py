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

    def get_training_data(self, opath=None):
        self.ds = self.vprm_pre.sat_imgs.sat_img.drop(['scl', 'ndvi']).drop(['time'])
        self.ds = ds.assign_attrs(crs=ds.rio.crs)
        self.ds['min_evi'] =self.vprm_pre.min_max_evi.sat_img['min_evi']
        self.ds['max_evi'] = self.vprm_pre.min_max_evi.sat_img['max_evi']
        self.ds['th'] = self.vprm_pre.min_max_evi.sat_img['th']
        self.ds['min_lswi'] =  self.vprm_pre.min_lswi.sat_img['min_lswi']
        self.ds['max_lswi'] =  self.vprm_pre.max_lswi.sat_img['max_lswi']
        flux_tower_keys = ['t2m', 'ssrd', 'ZL', 'FETCH_90',
                          'NEE_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_DT_VUT_REF',
                           'NEE_VUT_REF_QC', 'GPP_NT_VUT_REF',
                           'RECO_NT_VUT_REF']
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
        ds = ds.assign_coords(
            days_since_t0=(
                "datetime_utc",
                ((self.ds.datetime_utc.data - t0) / np.timedelta64(1, "D")).astype(int)))
        
        if 'GPP_DT_VUT_REF' in list(ds.keys()):
            gpp_key = 'GPP_DT_VUT_REF'
        elif 'GPP_NT_VUT_REF' in list(ds.keys()):
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
        
        meteo_vars = {'ssrd': lambda x: x/3600,
                      't2m': lambda x: x - 273.15,
                      'skt': lambda x: x - 273.15,
                      'swvl1': None,
                      'swvl2': None,
                      'e':  lambda x: x * 1e4,
                      'd2m': lambda x: x - 273.15,
                      'stl1': lambda x: x - 273.15,
                      'stl2': lambda x: x - 273.15,}
        
        self.era5_inst.reduce_time(self.flux_tower.flux_data['datetime_utc'].iloc[0],
                         self.flux_tower.flux_data['datetime_utc'].iloc[-1])
        
        self.era5_inst.ds_out = sel_nearest_valid(self.era5_inst.ds_out, lon, lat) 
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
        
        all_meteo_vars = np.concatenate([['t2m', 'ssrd', 'NEE_VUT_REF', gpp_key, 'RECO_DT_VUT_REF'],
                                        [i+'_era5' for i in self.era5_inst.ds_out.keys()]])
        meteos = self.ds[all_meteo_vars].sel({'datetime_utc': self.ds['t']})
        
        sat_vars = ['lswi','evi', 'nirv', 'ndre']
        self.ds[sat_vars] = (
            self.ds[sat_vars]
            .fillna(0.0)
        )
        self.ds["ffp_footprint"] = self.ds["ffp_footprint"].fillna(0.0)
        self.ds.attrs["crs"] = self.ds.attrs["crs"].to_wkt()
        if opath is not None:
            self.ds.to_netcdf(opath)
    
    def make_satellite_animation(self, opath=None)
        from pyproj import Transformer
        lon, lat = self.flux_tower.get_lonlat()
        transformer = Transformer.from_crs(
            "EPSG:4326",
            ds.rio.crs,   # or "EPSG:32632"
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
            im.set_array(ds['evi'].isel(time_gap_filled=frame).values)
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
    
        
    
      
          
