import os
import numpy as np 
import pandas as pd
import xarray as xr
from scipy.special import gamma
from pyVPRM.flux_tower_libs.base_footprint_class import base_footprint_manager


class FFP_footprint_manager(base_footprint_manager):
    def __init__(self, time_stamps, flux_tower_manager, calculation_grid_side_length, calculation_grid_pixels_per_side, era5_instance = None):
        super().__init__(time_stamps, flux_tower_manager, calculation_grid_side_length, calculation_grid_pixels_per_side, era5_instance)
        self.footprint_model = 'FFP'

    def print_parameters(self):
        print('Parameters of model', self.footprint_model)
        print('u')
        print(self.u)
        print('ZL')
        print(self.ZL)
        print('u dir')
        print(self.u_dir)
        print('u*')
        print(self.u_star)
        print('sigma v')
        print(self.sigma_v)
        print('boundary layer h')
        print(self.h_pbl)
        print('z')
        print(self.z)
        print('t')
        print(self.time_stamps)
    
    def calculate_footprints(self):  
        k_karman = 0.4
        #calculate secondary parameters
        X_star = self.X_calculation_grid_rotated/np.expand_dims(self.z, axis =(1,2)) * np.expand_dims((1-self.z/self.h_pbl)*(self.u/self.u_star*k_karman)**-1, axis=(1,2))
        p = np.where(self.ZL<=0, 0.8, 0.55)
        ps1 = np.minimum(1, np.abs(self.ZL)**-1*10**-5+p)

        
        #calculate crosswind_integrated_flux
        a = 1.452
        b = -1.991
        c = 1.462
        d = 0.136
        k_karman = 0.4
        Fy_star = a*(X_star-d)**b*np.exp(-c/(X_star-d))
        crosswind_integrated_flux = Fy_star*np.expand_dims((1-self.z/self.h_pbl)*self.u_star/(self.z*self.u*k_karman), axis=(1,2))
        if not np.all(np.logical_or(np.isnan(crosswind_integrated_flux), crosswind_integrated_flux==0)):
            crosswind_integrated_flux = np.where(np.isnan(crosswind_integrated_flux), 0, crosswind_integrated_flux)
        
        #calculate crosswind_distribution
        a_c = 2.17
        b_c = 1.66
        c_c = 20.0
        sigma_y_star = a_c*(b_c*(X_star)**2/(1+c_c*X_star))**.5
        sigma_y = sigma_y_star*np.expand_dims((self.z*self.sigma_v)/(ps1*self.u_star), axis=(1,2))
        crosswind_distribution = 1/((2*np.pi)**.5*sigma_y) * np.exp(-self.Y_calculation_grid_rotated**2/(2*sigma_y**2))
        crosswind_distribution = np.where(np.isnan(crosswind_distribution), 0, crosswind_distribution)
            
        #calculate footprints
        footprint = crosswind_distribution*crosswind_integrated_flux
        integrated_footprint = np.nansum(footprint, axis=(1,2))*self.side_length_pixel**2

        #calculate peak estimator
        x_peak = 0.87 * self.z * (1-self.z/self.h_pbl)**-1 * (self.u/self.u_star*k_karman)

        #calculate fetch estimator
        x_70 = (-c/np.log(0.7)+d) * self.z * (1-self.z/self.h_pbl)**-1 * (self.u/self.u_star*k_karman)
        x_90 = (-c/np.log(0.9)+d) * self.z * (1-self.z/self.h_pbl)**-1 * (self.u/self.u_star*k_karman)
        
        #store in xarray
        self.footprint_on_calculation_grid  =  xr.Dataset(
            data_vars=dict(
                footprint = (["t", "y", "x"], footprint),
                integrated = (["t"], integrated_footprint),
                x_peak = (["t"], x_peak),
                x_70 = (["t"], x_70),
                x_90 = (["t"], x_90),
                
            ),
            coords=dict(
                t=("t", self.time_stamps), 
                y=("y", self.ys_for_calculation_grid),
                x=("x", self.xs_for_calculation_grid), 
            ),
        )


    def calculate_crosswind_integrated_flux(self):  
        k_karman = 0.4
        #calculate secondary parameters
        X_star = self.xs_for_calculation_grid/np.expand_dims(self.z, axis =(1)) * np.expand_dims((1-self.z/self.h_pbl)*(self.u/self.u_star*k_karman)**-1, axis=(1))
        p = np.where(self.ZL<=0, 0.8, 0.55)
        ps1 = np.minimum(1, np.abs(self.ZL)**-1*10**-5+p)

        
        #calculate crosswind_integrated_flux
        a = 1.452
        b = -1.991
        c = 1.462
        d = 0.136
        k_karman = 0.4
        Fy_star = a*(X_star-d)**b*np.exp(-c/(X_star-d))
        crosswind_integrated_flux = Fy_star*np.expand_dims((1-self.z/self.h_pbl)*self.u_star/(self.z*self.u*k_karman), axis=(1))
        if not np.all(np.logical_or(np.isnan(crosswind_integrated_flux), crosswind_integrated_flux==0)):
            crosswind_integrated_flux = np.where(np.isnan(crosswind_integrated_flux), 0, crosswind_integrated_flux)
        self.crosswind_integrated_flux = crosswind_integrated_flux
