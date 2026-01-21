import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.animation as animation
from scipy import interpolate
from pyproj import Transformer
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
import rioxarray as rxr
import xesmf as xe
import cartopy.crs as ccrs
from pyVPRM.lib.functions import get_corners_from_pixel_centers_1D, make_xesmf_grid

class base_footprint_manager:
    def __init__(self, time_stamps, flux_tower_manager, calculation_grid_side_length = 5000, calculation_grid_pixels_per_side = 1000, era5_instance=None):
        #print('make footprint manager')
        self.site_ID = flux_tower_manager.site_name
        self.time_stamps = time_stamps
        if flux_tower_manager is not None:
            flux_tower_data = flux_tower_manager.flux_data.where(flux_tower_manager.flux_data != -9999.) #set invalid values to nan, don't drop
            t_Frame = pd.DataFrame(data = self.time_stamps, columns=['datetime_utc'])
            t_Frame['datetime_utc'] = pd.to_datetime(t_Frame['datetime_utc'], utc = True)
            flux_tower_data['datetime_utc'] = pd.to_datetime(flux_tower_data['datetime_utc'], utc = True)
            #print(t_Frame)
            #print(flux_tower_manager.flux_data.datetime_utc)
            flux_tower_data = pd.merge(flux_tower_data, t_Frame, on='datetime_utc', how='right') #merge on t_frame
            self.time_stamps = flux_tower_data['datetime_utc'].values
            self.z = flux_tower_data['z_footprint'].values 
            self.lon = flux_tower_manager.lon
            self.lat = flux_tower_manager.lat
            self.elev = flux_tower_manager.elev
            self.z_displacement = flux_tower_manager.mean_z_displacement
            if 'MO_LENGTH' in flux_tower_data.columns:
                self.L = flux_tower_data['MO_LENGTH'].values
            if 'ZL' in flux_tower_data.columns:
                self.ZL = flux_tower_data['ZL'].values
            if 'PBLH' in flux_tower_data.columns:
                self.h_pbl = flux_tower_data['PBLH'].values
            elif era5_instance is not None:
                era5_instance.get_data_series(flux_tower_manager.get_lonlat(), 'blh', self.time_stamps)
            else:
                self.h_pbl = np.ones(len(self.time_stamps))*1000
            if 'FETCH_70' in flux_tower_data.columns:
                self.fetch_70 = flux_tower_data['FETCH_70'].values
            else:
                self.fetch_70 = None
            if 'FETCH_80' in flux_tower_data.columns:
                self.fetch_80 = flux_tower_data['FETCH_80'].values
            else:
                self.fetch_80 = None
            if 'FETCH_90' in flux_tower_data.columns:
                self.fetch_90 = flux_tower_data['FETCH_90'].values
            else:
                self.fetch_90 = None
            if 'FETCH_MAX' in flux_tower_data.columns:
                self.fetch_max = flux_tower_data['FETCH_MAX'].values
            else:
                self.fetch_max = None
            self.u = flux_tower_data['WS'].values
            self.u_dir = flux_tower_data['WD'].values
            self.u_star = flux_tower_data['USTAR'].values
            self.sigma_v = flux_tower_data['V_SIGMA'].values
        self.calculation_grid_side_length = calculation_grid_side_length  #in meters
        self.calculation_grid_resolution = calculation_grid_pixels_per_side
        self.side_length_pixel = self.calculation_grid_side_length/self.calculation_grid_resolution
        self.footprint_on_calculation_grid = None
        self.footprint_on_satellite_grid = None   


    

    def make_calculation_grid(self):
        #make a finer grid to calculate the footprints on
        #print('make calculation grid')
        self.xs_for_calculation_grid = np.linspace(-0.5*(self.calculation_grid_side_length-self.side_length_pixel), 0.5*(self.calculation_grid_side_length-self.side_length_pixel), self.calculation_grid_resolution)
        self.ys_for_calculation_grid = np.linspace(-0.5*(self.calculation_grid_side_length-self.side_length_pixel), 0.5*(self.calculation_grid_side_length-self.side_length_pixel), self.calculation_grid_resolution)
        self.X_calculation_grid, self.Y_calculation_grid = np.meshgrid(self.xs_for_calculation_grid, self.ys_for_calculation_grid)
        
        #rotate grid so that the x axis vector is (anti)parallel to the mean wind vector
        self.X_calculation_grid_rotated   =  np.expand_dims(np.cos((-self.u_dir+90)* np.pi/180), axis=(1,2))*np.expand_dims(self.X_calculation_grid, axis=0) + np.expand_dims(np.sin((-self.u_dir+90)* np.pi/180), axis=(1,2))*np.expand_dims(self.Y_calculation_grid, axis=0)
        self.Y_calculation_grid_rotated   = -np.expand_dims(np.sin((-self.u_dir+90)* np.pi/180), axis=(1,2))*np.expand_dims(self.X_calculation_grid, axis=0) + np.expand_dims(np.cos((-self.u_dir+90)* np.pi/180), axis=(1,2))*np.expand_dims(self.Y_calculation_grid, axis=0)



    #def filter_footprints_based_on_integral(self, acceptance_percentage, is_printing_sum = False):
    #    self.footprint_on_calculation_grid = self.footprint_on_calculation_grid.where(self.footprint_on_calculation_grid.integrated > acceptance_percentage, self.footprint_on_calculation_grid, 0)
    #    if is_printing_sum:
    #        print(self.footprint_on_calculation_grid.integrated.values)
            

    
    def regrid_calculation_grid_to_satellite_grid(self, satellite_grid, regridder_file_path): 
        #print('regridd to satellite grid')
        footprint_to_satellite_file_path = os.path.join(regridder_file_path, '{0}_footprint_{1}_px_per_{2}_m_to_sentinel2_{3}_x_{4}_px.nc'.format(self.site_ID, self.calculation_grid_resolution, self.calculation_grid_side_length, len(satellite_grid.x.values), len(satellite_grid.y.values) ))
        my_transformer = Transformer.from_pipeline('+proj=pipeline +step +inv +proj=topocentric +lon_0={0} +lat_0={1} +h_0={2} +step +inv +proj=cart +ellps=WGS84'.format(self.lon, self.lat, self.elev+self.z_displacement))
        xesmf_grid_footprint =  make_xesmf_grid(self.footprint_on_calculation_grid, transformer = my_transformer)
        xesmf_grid_satellite =  make_xesmf_grid(satellite_grid)
        if os.path.exists(footprint_to_satellite_file_path):
            weights = footprint_to_satellite_file_path 
            self.regridder = xe.Regridder(xesmf_grid_footprint, xesmf_grid_satellite, method="conservative_normed", weights = weights)
        else:
            weights = None
            print('Calculating new regridder for footprint')
            self.regridder = xe.Regridder(xesmf_grid_footprint, xesmf_grid_satellite, method="conservative_normed")
            print('Store new regridder')
            self.regridder.to_netcdf(footprint_to_satellite_file_path) 
        self.footprint_on_satellite_grid = self.regridder(self.footprint_on_calculation_grid)
        self.footprint_on_satellite_grid = self.footprint_on_satellite_grid.assign_coords(
                    {
                        "y": satellite_grid.coords["y"].values,
                        "x": satellite_grid.coords["x"].values,
                    }
                )
        summed_footprint = self.footprint_on_satellite_grid.footprint.sum(dim=["x", "y"])
        self.footprint_on_satellite_grid['footprint'] = self.footprint_on_satellite_grid.footprint/summed_footprint.where(summed_footprint!=0)
        self.footprint_on_satellite_grid['integrated'] = self.footprint_on_calculation_grid.integrated
        self.footprint_on_satellite_grid['integrated_on_sat'] = self.footprint_on_satellite_grid.footprint.sum(dim=["x", "y"])
        self.footprint_on_satellite_grid['x_peak'] = self.footprint_on_calculation_grid.x_peak
        if self.footprint_model == 'FFP':
            self.footprint_on_satellite_grid['x_70'] = self.footprint_on_calculation_grid.x_70            
            self.footprint_on_satellite_grid['x_90'] = self.footprint_on_calculation_grid.x_90
    
    def get_radius_of_percentile(self, percentile = 0.9):
        dr = self.xs_for_calculation_grid[1] -self.xs_for_calculation_grid[0]
        print(dr)
        r_percentiles = np.empty(len(self.crosswind_integrated_flux[:, 0]))
        integrated_fluxes = np.zeros(len(self.crosswind_integrated_flux[:, 0]))
        for idx_time in range(len(self.crosswind_integrated_flux[:, 0])):
            for idx_flux, flux in enumerate(self.crosswind_integrated_flux[idx_time, :]):
                integrated_fluxes[idx_time] = integrated_fluxes[idx_time] + flux*dr
                if integrated_fluxes[idx_time] >=percentile:
                    r_percentiles[idx_time] = self.xs_for_calculation_grid[idx_flux]
                    break
        return r_percentiles, integrated_fluxes

    def get_radius_peak(self):
        r_peak = self.xs_for_calculation_grid[np.argmax(self.crosswind_integrated_flux, axis = 1)]
        flux_peak = np.max(self.crosswind_integrated_flux, axis = 1)
        return r_peak, flux_peak
        

    def show_footprints_on_calculation_grid(self, output_image_path):
        #shape should be (t, x, y)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        def update(i):
            fig.suptitle('{0} footprint on calculation grid at {1}, t={2}, u={3}m/s, \n u_dir={4}°, ZL={5}, u_star={6}m/s,  sigma_v={7}m/s'.format(self.footprint_model, self.site_ID, self.time_stamps[i], self.u[i], self.u_dir[i], self.ZL[i], self.u_star[i], self.sigma_v[i]))
            #get next footprint
            footprint = self.footprint_on_calculation_grid.isel(t=i)
            #print('nonregridded sum', np.sum(footprint['KM_footprint'].values))
            ax.clear()
            # replot
            CS = ax.contour(self.X_calculation_grid, self.Y_calculation_grid, footprint['footprint'].values, cmap='spring')
            if not np.isnan(footprint['x_peak']):
                x_peak = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                y_peak = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                ax.scatter(x_peak, y_peak)
                if self.footprint_model == 'FFP':
                    x_70 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    y_70 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    x_90 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    y_90 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    ax.scatter(x_70, y_70)
                    ax.scatter(x_90, y_90)
            ax.arrow(0, 0, -np.sin(self.u_dir[i]* np.pi /180)*20*self.u[i], -np.cos(self.u_dir[i]* np.pi/180)*20*self.u[i], width = 3)
        ani = animation.FuncAnimation(fig, update, frames=len(self.time_stamps), interval=1000)
        ani.save(os.path.join(output_image_path,'{}_{}_footprint_evolution_on_meshgrid.gif'.format(self.site_ID, self.footprint_model)), writer='pillow')


    def show_footprints_on_satellite_grid(self, output_image_path, X, Y):
        #shape should be (t, x, y)
        fig, ax = plt.subplots(figsize=(8,8))
        cax = ax.inset_axes([1.03, 0, 0.1, 1], transform=ax.transAxes)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        def update(i):
            fig.suptitle('{0} footprint on satellite grid at {1}, t={2}, u={3}m/s, \n u_dir={4}°, ZL={5}, u_star={6}m/s,  sigma_v={7}m/s'.format(self.footprint_model, self.site_ID, self.time_stamps[i], self.u[i], self.u_dir[i], self.ZL[i], self.u_star[i], self.sigma_v[i]))
            #get next footprint
            footprint = self.footprint_on_satellite_grid.isel(t=i)
            ax.clear()
            cax.clear()
            # replot
            CS = xr.plot.pcolormesh(footprint['footprint'], x = 'x', y='y', ax=ax, add_colorbar=None, cbar_ax = cax)
            if not np.isnan(footprint['x_peak']):
                x_peak = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                y_peak = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                ax.scatter(x_peak, y_peak)
                if self.footprint_model == 'FFP':
                    x_70 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    y_70 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    x_90 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    y_90 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    ax.scatter(x_70, y_70)
                    ax.scatter(x_90, y_90)
        ani = animation.FuncAnimation(fig, update, frames=len(self.time_stamps), interval=1000)
        ani.save(os.path.join(output_image_path,'{}_{}_footprint_evolution_on_satellite_grid.gif'.format(self.site_ID, self.footprint_model)), writer='pillow')

    def show_regridded_footprints(self, output_image_path):
        #shape should be (t, x, y)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        def update(i):
            fig.suptitle('{0} footprint on satellite grid at {1}, t={2}, u={3}m/s, \n u_dir={4}°, ZL={5}, u_star={6}m/s,  sigma_v={7}m/s'.format(self.footprint_model, self.site_ID, self.time_stamps[i], self.u[i], self.u_dir[i], self.ZL[i], self.u_star[i], self.sigma_v[i]))
            #get next footprint
            footprint = self.footprint_on_satellite_grid.isel(t=i)
            ax.clear()
            # replot
            CS = ax.pcolormesh(footprint['footprint'])
            if not np.isnan(footprint['x_peak']):
                x_peak = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                y_peak = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                ax.scatter(x_peak, y_peak)
                if self.footprint_model == 'FFP':
                    x_70 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    y_70 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    x_90 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    y_90 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    ax.scatter(x_70, y_70)
                    ax.scatter(x_90, y_90)
        ani = animation.FuncAnimation(fig, update, frames=len(self.time_stamps), interval=1000)
        ani.save(os.path.join(output_image_path,'{}_{}_footprint_regridded_evolution.gif'.format(self.site_ID, self.footprint_model)), writer='pillow')


    def show_footprint_percentiles(self, output_image_path, X, Y):
        #shape should be (t, x, y)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        n = 1000
        def update(i):
            fig.suptitle('{0} footprint 68th, 90th, 95th percentile at {1}, t={2}, u={3}m/s, \n u_dir={4}°, ZL={5}, u_star={6}m/s,  sigma_v={7}m/s'.format(self.footprint_model, self.site_ID, self.time_stamps[i], self.u[i], self.u_dir[i], self.ZL[i], self.u_star[i], self.sigma_v[i]))
            
            #get next footprint
            footprint = self.footprint_on_satellite_grid.isel(t=i)
            t = np.linspace(0, footprint['footprint'].values.max(), n)
            integral = ((footprint['footprint'].values >= t[:, None, None]) * footprint['footprint'].values).sum(axis=(1,2))
            f = interpolate.interp1d(integral, t)
            try:
                t_contours = f(np.array([0.95, 0.9, 0.68]))
            except:
                t_contours=None
                #print('nonregridded sum', np.sum(footprint['KM_footprint'].values))
            ax.clear()
            # replot
            CS = ax.contour(X, Y, footprint['footprint'].values, t_contours, cmap='spring')
            if not np.isnan(footprint['x_peak']):
                x_peak = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                y_peak = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_peak']
                ax.scatter(x_peak, y_peak)
                if self.footprint_model == 'FFP':
                    x_70 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    y_70 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_70']
                    x_90 = -np.cos((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    y_90 = np.sin((self.u_dir[i]+90)* np.pi/180)*footprint['x_90']
                    ax.scatter(x_70, y_70)
                    ax.scatter(x_90, y_90)
            ax.arrow(0, 0, -np.sin(self.u_dir[i]* np.pi /180)*20*self.u[i], -np.cos(self.u_dir[i]* np.pi/180)*20*self.u[i], width = 3)
            ax.set_ylim(-1000, 1000)
            ax.set_xlim(-1000, 1000)

        ani = animation.FuncAnimation(fig, update, frames=len(self.time_stamps), interval=1)
        ani.save(os.path.join(output_image_path,'{}_{}_footprint_percentiles.gif'.format(self.site_ID, self.footprint_model)), writer='pillow')

#------------------------------------------------------------------------------------------------------------------------------------------------------


def get_corners_from_pixel_centers_1D(pixel_centers_1D):
    half_pixel_width = np.unique(np.diff(pixel_centers_1D))[0]/2       #get smallest difference between two pixel centers (why not just take first?)
    pixel_corners = pixel_centers_1D - half_pixel_width                #shift from center to top/left corners
    pixel_corners = list(pixel_corners)                                #list for fast appending
    pixel_corners.append(pixel_corners[-1]+2*half_pixel_width)         #add missing lower/right corners
    pixel_corners = np.array(pixel_corners)
    return pixel_corners  


def make_xesmf_grid_from_satellite_image(satellite_image):
    #get x and y coordinates of pixel centers from the satellite image
    x_pixel_centers = satellite_image.x.values
    y_pixel_centers = satellite_image.y.values
    
    #get the pixel corners from the pixel centers
    x_pixel_corners = get_corners_from_pixel_centers_1D(x_pixel_centers)
    y_pixel_corners = get_corners_from_pixel_centers_1D(y_pixel_centers)

    #make meshgrids
    X_center_grid, Y_center_grid = np.meshgrid(x_pixel_centers, y_pixel_centers)
    X_corner_grid, Y_corner_grid = np.meshgrid(x_pixel_corners, y_pixel_corners)
    
    #define transformer to transform from the images crs to the defined crs of WGS84
    my_transformer = Transformer.from_crs(satellite_image.rio.crs,
                        '+proj=longlat +datum=WGS84',
                        always_xy=True)
    
    #apply transformer to the grids
    X_center_grid, Y_center_grid = my_transformer.transform(X_center_grid, Y_center_grid)
    X_corner_grid, Y_corner_grid = my_transformer.transform(X_corner_grid, Y_corner_grid)
    
    #put the grids in one xarray
    pixel_grid = xr.Dataset({"lon": (["y", "x"], X_center_grid,
                          {"units": "degrees_east"}),
                         "lon_b": (["y_b", "x_b"], X_corner_grid,
                         {"units": "degrees_east"}),
                          "lat": (["y", "x"], Y_center_grid,
                          {"units": "degrees_north"}),
                         "lat_b": (["y_b", "x_b"], Y_corner_grid,
                         {"units": "degrees_north"})
                          })
    pixel_grid = pixel_grid.set_coords(['lon', 'lat', "lon_b", "lat_b"])
    return pixel_grid

def make_xesmf_grid_from_x_y(footprint_data, lon, lat, elev):
    #get x and y coordinates of pixel centers from the satellite image
    x_pixel_centers = footprint_data.x.values
    y_pixel_centers = footprint_data.y.values
    
    #get the pixel corners from the pixel centers
    x_pixel_corners = get_corners_from_pixel_centers_1D(x_pixel_centers)
    y_pixel_corners = get_corners_from_pixel_centers_1D(y_pixel_centers)

    #make meshgrids
    X_center_grid, Y_center_grid = np.meshgrid(x_pixel_centers, y_pixel_centers, indexing='ij')
    X_corner_grid, Y_corner_grid = np.meshgrid(x_pixel_corners, y_pixel_corners, indexing='ij')
    
    #define transformer to transform from the images crs to the defined crs of WGS84
    my_transformer = Transformer.from_pipeline('+proj=pipeline +step +inv +proj=topocentric +lon_0={0} +lat_0={1} +h_0={2} +step +inv +proj=cart +ellps=WGS84'.format(lon, lat, elev))
    
    #apply transformer to the grids
    X_center_grid, Y_center_grid = my_transformer.transform(X_center_grid, Y_center_grid, errcheck=True)
    X_corner_grid, Y_corner_grid = my_transformer.transform(X_corner_grid, Y_corner_grid, errcheck=True)
    #print('X center grid', X_center_grid)
    
    #put the grids in one xarray
    pixel_grid = xr.Dataset({"lon": (["y", "x"], X_center_grid,
                          {"units": "degrees_east"}),
                         "lon_b": (["y_b", "x_b"], X_corner_grid,
                         {"units": "degrees_east"}),
                          "lat": (["y", "x"], Y_center_grid,
                          {"units": "degrees_north"}),
                         "lat_b": (["y_b", "x_b"], Y_corner_grid,
                         {"units": "degrees_north"})
                          })
    pixel_grid = pixel_grid.set_coords(['lon', 'lat', "lon_b", "lat_b"])
    return pixel_grid
