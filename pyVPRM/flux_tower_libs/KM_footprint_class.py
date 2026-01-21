import os
import numpy as np 
import pandas as pd
import xarray as xr
from scipy.special import gamma
from pyVPRM.flux_tower_libs.base_footprint_class import base_footprint_manager

class KM_footprint_manager(base_footprint_manager):
    def __init__(self, time_stamps, flux_tower_manager, calculation_grid_side_length, calculation_grid_pixels_per_side):
        super().__init__(time_stamps, flux_tower_manager, calculation_grid_side_length, calculation_grid_pixels_per_side)
        self.footprint_model = 'KM'

    def print_parameters(self):
        print('Parameters of model', self.footprint_model)
        print('u')
        print(self.u)
        print('L')
        print(self.ZL)
        print('u dir')
        print(self.u_dir)
        print('u*')
        print(self.u_star)
        print('sigma v')
        print(self.sigma_v)
        print('t')
        print(self.time_stamps)


    def calculate_footprints(self):
        #print('calculate KM footprint')
        k_karman = 0.4
        #calculate secondary parameters
        stability_c = np.where(self.ZL > 0, (1+5*self.ZL), (1-16*self.ZL)**(-1/2))       
        stability_m = np.where(self.ZL > 0, (1+5*self.ZL), (1-16*self.ZL)**(-1/4))
        n = np.where(self.ZL > 0, 1/(1+5*self.ZL), (1-24*self.ZL)/(1-16*self.ZL))
        #stability_c = np.where(self.L > 0, (1+5*self.z/self.L), (1-16*self.z/self.L)**(-1/2))       
        #stability_m = np.where(self.L > 0, (1+5*self.z/self.L), (1-16*self.z/self.L)**(-1/4))
        #n = np.where(self.L > 0, 1/(1+5*self.z/self.L), (1-24*self.z/self.L)/(1-16*self.z/self.L))
        kappa = k_karman*self.u_star*self.z/(stability_c*self.z**n)
        m = self.u_star*stability_m/(k_karman*self.u)
        r = 2+m-n
        mu = (1+m)/r
        #U = self.u/self.z**m
        U = self.u_star*stability_m/(k_karman*m*self.z**m)
        xi = U*self.z**r/(r**2*kappa)
        
        #calculate crosswind_integrated_flux
        crosswind_integrated_flux = np.expand_dims(xi**mu/gamma(mu), axis=(1,2)) / self.X_calculation_grid_rotated**np.expand_dims((1+mu), axis=(1,2)) * np.exp(-np.expand_dims(xi, axis=(1,2))/self.X_calculation_grid_rotated) 
        if not np.all(np.logical_or(np.isnan(crosswind_integrated_flux), crosswind_integrated_flux==0)):
            crosswind_integrated_flux = np.where(np.isnan(crosswind_integrated_flux), 0, crosswind_integrated_flux)
            
        #calculate crosswind_distribution
        u_bar = np.expand_dims(gamma(mu)/gamma(1/r)*(r**2*kappa/U)**(m/r)*U, axis=(1,2))*self.X_calculation_grid_rotated**np.expand_dims((m/r), axis=(1,2))
        sigma_y = self.X_calculation_grid_rotated*np.expand_dims(self.sigma_v, axis=(1,2))/u_bar
        crosswind_distribution = 1/((2*np.pi)**.5*sigma_y) * np.exp(-self.Y_calculation_grid_rotated**2/(2*sigma_y**2))
        #if not np.all(np.logical_or(np.isnan(crosswind_distribution), crosswind_distribution==0)):
        crosswind_distribution = np.where(np.isnan(crosswind_distribution), 0, crosswind_distribution)
            
        #calculate footprints:
        footprint = crosswind_distribution*crosswind_integrated_flux
        integrated_footprint = np.nansum(footprint, axis=(1,2))*self.side_length_pixel**2

        #calculate peak estimator
        x_peak = xi/(1+mu)
        
        #store in xarray
        self.footprint_on_calculation_grid  =  xr.Dataset(
            data_vars=dict(
                footprint = (["t", "y", "x"], footprint),
                integrated = (["t"], integrated_footprint),
                x_peak = (["t"], x_peak),
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
        stability_c = np.where(self.ZL > 0, (1+5*self.ZL), (1-16*self.ZL)**(-1/2))       
        stability_m = np.where(self.ZL > 0, (1+5*self.ZL), (1-16*self.ZL)**(-1/4))
        n = np.where(self.ZL > 0, 1/(1+5*self.ZL), (1-24*self.ZL)/(1-16*self.ZL))
        #stability_c = np.where(self.L > 0, (1+5*self.z/self.L), (1-16*self.z/self.L)**(-1/2))       
        #stability_m = np.where(self.L > 0, (1+5*self.z/self.L), (1-16*self.z/self.L)**(-1/4))
        #n = np.where(self.L > 0, 1/(1+5*self.z/self.L), (1-24*self.z/self.L)/(1-16*self.z/self.L))
        kappa = k_karman*self.u_star*self.z/(stability_c*self.z**n)
        m = self.u_star*stability_m/(k_karman*self.u)
        r = 2+m-n
        mu = (1+m)/r
        #U = self.u/self.z**m
        U = self.u_star*stability_m/(k_karman*m*self.z**m)
        xi = U*self.z**r/(r**2*kappa)
        
        #calculate crosswind_integrated_flux
        crosswind_integrated_flux = np.expand_dims(xi**mu/gamma(mu), axis=(1)) / self.xs_for_calculation_grid**np.expand_dims((1+mu), axis=(1)) * np.exp(-np.expand_dims(xi, axis=(1))/self.xs_for_calculation_grid) 
        if not np.all(np.logical_or(np.isnan(crosswind_integrated_flux), crosswind_integrated_flux==0)):
            crosswind_integrated_flux = np.where(np.isnan(crosswind_integrated_flux), 0, crosswind_integrated_flux)
        self.crosswind_integrated_flux = crosswind_integrated_flux    
