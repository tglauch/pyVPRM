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

class pyvprnn:
    """
    Base class for all pyvprnn models
    """
    def __init__(self, vprm_pre=None, met=None, footprint=None met_keys=[]):
        self.era5_inst = met
        self.vprm_pre = vprm_pre
        self.met_keys = met_keys
        self.footprint= footprint
        return

    def get_training_data(self):


    def make_sat_imgs_plot(self, ind_x, ind_y, opath=None):
        from pyVPRM.lib.fancy_plot import *
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        
        timestamp_start_np = np.datetime64(vprm_inst.timestamp_start)  # convert to numpy.datetime64
        days_array = vprm_inst.sat_imgs.sat_img['time_gap_filled'].values  # NumPy array
        
        plt_times = timestamp_start_np + np.timedelta64(1, 'D') * days_array
        
        plt_times_pre_smoothing = timestamp_start_np + np.timedelta64(1, 'D') * handler_pre_smoothing['time']
        # --- Extract values ---
        nirv_pre  = handler_pre_smoothing['nirv'].values[:, ind_x, ind_y]
        evi_pre   = handler_pre_smoothing['evi'].values[:, ind_x, ind_y]
        ndre_pre  = handler_pre_smoothing['ndre'].values[:, ind_x, ind_y]
        evi2_pre  = handler_pre_smoothing['evi2'].values[:, ind_x, ind_y]
        lswi_pre  = handler_pre_smoothing['lswi'].values[:, ind_x, ind_y]
        scl_vals  = handler_pre_smoothing['scl'].values[:, ind_x, ind_y]
        
        nirv_smooth = vprm_inst.sat_imgs.sat_img['nirv'].values[:, ind_x, ind_y]
        evi_smooth  = vprm_inst.sat_imgs.sat_img['evi'].values[:, ind_x, ind_y]
        ndre_smooth = vprm_inst.sat_imgs.sat_img['ndre'].values[:, ind_x, ind_y]
        evi2_smooth = vprm_inst.sat_imgs.sat_img['evi2'].values[:, ind_x, ind_y]
        lswi_smooth = vprm_inst.sat_imgs.sat_img['lswi'].values[:, ind_x, ind_y]
        
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

    

  
      
