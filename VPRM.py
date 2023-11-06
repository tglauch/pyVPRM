import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve()))
import numpy as np
from lib.sat_manager import VIIRS, sentinel2, modis,\
                            copernicus_land_cover_map, satellite_data_manager
from lib.era5_class import ERA5
from scipy.ndimage import uniform_filter
from astropy.convolution import convolve
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import copy
from joblib import Parallel, delayed
import xarray as xr
import scipy
import uuid
import time
from scipy.optimize import curve_fit
import pandas as pd
import datetime
from dateutil import parser
from multiprocessing import Process
import rasterio


def adjust_timestamps(sat_img, start_date, stop_date, timestamp0):
    start_day = start_date.timetuple().tm_yday
    stop_day = stop_date.timetuple().tm_yday    
    if sat_img['timestamps'].ndim == 1:
        for i in range(np.shape(sat_img['timestamps'])[0]):
            this_ts = sat_img['timestamps'][i]
            if np.abs(this_ts - start_day) < np.abs(this_ts - stop_day):
                sat_img['timestamps'][i] = ((start_date + datetime.timedelta(days=float(np.abs(this_ts - start_day)))) - timestamp0).days
            else:
                sat_img['timestamps'][i] = ((stop_date - datetime.timedelta(days=float(np.abs(this_ts - stop_day)))) - timestamp0).days    
    else:
        for this_ts in np.unique(sat_img['timestamps'].values): # this_ts is the day of the year
            if np.isnan(this_ts):
                this_ts = start_day
            if np.abs(this_ts - start_day) < np.abs(this_ts - stop_day):
                sat_img['timestamps'].values[sat_img['timestamps'].values==this_ts] = ((start_date + datetime.timedelta(days=float(np.abs(this_ts - start_day)))) - timestamp0).days
            else:
                sat_img['timestamps'].values[sat_img['timestamps'].values==this_ts] = ((stop_date - datetime.timedelta(days=float(np.abs(this_ts - stop_day)))) - timestamp0).days
            
        
#         for i in range(np.shape(sat_img['timestamps'])[0]):
#             for j in range(np.shape(sat_img['timestamps'])[1]):
#                 this_ts = sat_img['timestamps'][i][j]
#                 if np.abs(this_ts - start_day) < np.abs(this_ts - stop_day):
#                     sat_img['timestamps'][i][j] = ((start_date + datetime.timedelta(days=float(np.abs(this_ts - start_day)))) - timestamp0).days
#                 else:
#                     sat_img['timestamps'][i][j] = ((stop_date - datetime.timedelta(days=float(np.abs(this_ts - stop_day)))) - timestamp0).days
    return


def do_lowess_smoothing(array_to_smooth, xvals=None, timestamps=None, vclass=None,
                        frac=0.25, it=3):
    '''
        Performs lowess smoothing on a 2-D-array, where the first dimension is the time.

            Parameters:
                    array_to_smooth (list): The 2-D-array
                    vclass (list): 1-D list of vegetation classes. Vegetation class 8 (builup) is
                                   skipped in the smoothing process

            Returns:
                    The lowess smoothed array
    '''

    ret = []

    if array_to_smooth.ndim == 1:
        if timestamps is None:
            t_timestamp = np.arange(len(array_to_smooth))
        else:
            t_timestamp = timestamps
        mask = np.isfinite(array_to_smooth)
        print(t_timestamp, array_to_smooth)
        if xvals is None:
            xvals = t_timestamp[mask]
        ret = lowess(array_to_smooth[mask], t_timestamp[mask],
                     is_sorted=True, frac=frac, it=it,
                     xvals=xvals,
                     return_sorted=False)
        return ret
    else:
        if xvals is not None:
            ret_array = np.zeros((len(xvals),
                                  np.shape(array_to_smooth)[1]))
        else:
            ret_array = np.zeros((len(array_to_smooth[:, 0]),
                                  np.shape(array_to_smooth)[1]))
        for j in range(np.shape(array_to_smooth)[1]):
            if timestamps is None:
                t_timestamp = np.arange(len(array_to_smooth[:, j]))
            else:
                if timestamps.ndim == 1:
                    t_timestamp = timestamps
                else:
                    t_timestamp = timestamps[:, j]
            # if vclass!=None:
            #     if vclass[j] == 8: # skip arrays with land type class 8 
            #         continue
            mask = np.isfinite(array_to_smooth[:, j])
            if xvals is None:
                xvals = t_timestamp
            ret_array[:, j] = lowess(array_to_smooth[:, j][mask], t_timestamp[mask],
                                     is_sorted=True, frac=frac, it=it, xvals=xvals,
                                     return_sorted=False)
        return ret_array.T

class vprm: 
    '''
    Class for the  Vegetation Photosynthesis and Respiration Model
    '''
    def __init__(self, land_cover_map=None, verbose=False, n_cpus=1, sites=None):
        '''
            Initialize a class instance

            Parameters:
                    land_cover_map (xarray): A pre calculated map with the land cover types
                    verbose (bool): Set true for additional output when debugging
                    n_cpus: Number of CPUs
                    sites: For fitting. Provide a list of sites.

            Returns:
                    The lowess smoothed array
        '''

        self.sat_imgs = []
        
        self.sites = sites
        if self.sites is not None:
            self.lonlats = [i.get_lonlat() for i in sites]
        self.n_cpus = n_cpus
        self.era5_inst = None
        self.ret_dict = dict()
        self.verbose = verbose
        self.counter = 0
        self.fit_params_dict = None
        self.res = None
        
        self.new = True
        self.timestamps = []
        self.t2m = None
        
        self.target_shape = None
        
        self.sat_img_buffer = dict()
        self.buffer = dict()
        self.buffer['cur_lat'] = None
        self.buffer['cur_lon'] = None
        self.prototype_lat_lon = None
        
        self.land_cover_type = land_cover_map
                                  # land_cover_type: tmin, topt, tmax
        self.temp_coefficients = {1: [0, 20, 40], # Evergreen forest
                                  2: [0, 20, 40], # Deciduous forest
                                  3: [0, 20, 40], # Mixed forest
                                  4: [2, 20, 40], # Shrubland
                                  5: [2, 20, 40], # Savannas
                                  6: [5, 22, 40], # Cropland
                                  7: [2, 18, 40], # Grassland
                                  8: [0, 0, 40]}  # Other

        self.map_copernicus_to_vprm_class = {0: 8, 111: 1, 113: 2,
                                             112:1 , 114:2, 115:3,
                                             116:3, 121:1, 123:2,
                                             122 : 1, 124 : 2,
                                             125 : 3,
                                             126: 5, #This could be the savanna type! Check.
                                             20: 4, 30: 7, 90: 7,
                                             100: 7, 60: 8,
                                             40: 6, 50: 8,
                                             70: 8, 80: 8, 200: 8}
        return
    
    
    def to_wrf_output(self, out_grid, weights_for_regridder=None,
                      regridder_save_path=None, driver='xEMSF'):

        '''
            Generate output in the format that can be used as an input for WRF 

                Parameters:
                        out_grid (dict or xarray): Can be either a dictionary with 1D lats and lons
                                                   or an xarray dataset
                        weights_for_regridder (str): Weights to be used for regridding to the WRF grid
                        regridder_save_path (str): Save path when generating a new regridder
                        driver (str): Either ESMF_RegridWeightGen or xESMF. When setting to ESMF_RegridWeightGen
                                      the ESMF library is called directly

                Returns:
                        Dictionary with a dictinoary of the WRF input arrays
        '''
        
        import xesmf as xe


        src_x = self.sat_imgs.sat_img.coords['x'].values
        src_y = self.sat_imgs.sat_img.coords['y'].values
        X, Y = np.meshgrid(src_x, src_y)
        t = Transformer.from_crs(self.sat_imgs.sat_img.rio.crs,
                                '+proj=longlat +datum=WGS84')
        x_long, y_lat = t.transform(X, Y)
        src_grid = xr.Dataset({"lon": (["y", "x"], x_long ,
                             {"units": "degrees_east"}),
                              "lat": (["y", "x"], y_lat,
                             {"units": "degrees_north"})})
        src_grid = src_grid.set_coords(['lon', 'lat'])
        if isinstance(out_grid, dict):
            ds_out = xr.Dataset(
                 {"lon": (["lon"], out_grid['lons'],
                          {"units": "degrees_east"}),
                  "lat": (["lat"], out_grid['lats'],
                          {"units": "degrees_north"})})
        else:
            ds_out = out_grid
        if weights_for_regridder is None:
            print('Need to generate the weights for the regridder. This can be very slow and memory intensive')
            if driver == 'xEMSF':
                regridder = xe.Regridder(src_grid, ds_out, "bilinear")
                if regridder_save_path is not None:
                    regridder.to_netcdf(regridder_save_path)
            elif driver == 'ESMF_RegridWeightGen':
                if regridder_save_path is None:
                    print('If you use ESMF_RegridWeightGen, a regridder_save_path needs to be given')
                    return
                src_temp_path = os.path.join(os.path.dirname(regridder_save_path), '{}.nc'.format(str(uuid.uuid4())))
                dest_temp_path = os.path.join(os.path.dirname(regridder_save_path), '{}.nc'.format(str(uuid.uuid4())))
                src_grid.to_netcdf(src_temp_path)
                ds_out.to_netcdf(dest_temp_path)
                os.system('mpirun -np {} ESMF_RegridWeightGen --source {} --destination {} --weight {} -m bilinear -r --netcdf4 --no_log --extrap_method nearestd –src_regional –dest_regional '.format(self.n_cpus, src_temp_path, dest_temp_path, regridder_save_path)) # --no_log
                os.remove(src_temp_path) 
                os.remove(dest_temp_path)
                weights_for_regridder = regridder_save_path
            else:
                print('Driver needs to be xEMSF or ESMF_RegridWeightGen' )
        if weights_for_regridder is not None:
            regridder = xe.Regridder(src_grid, ds_out,
                                     "bilinear", weights=weights_for_regridder,
                                     reuse_weights=True)
        veg_inds = np.unique([self.map_copernicus_to_vprm_class[i] 
                              for i in self.map_copernicus_to_vprm_class.keys()])
        veg_inds = np.array(veg_inds, dtype=np.int32)
        dims = list(ds_out.dims.mapping.keys())
        for c, i in enumerate(veg_inds):
            if c == 0:
                t = copy.deepcopy(self.land_cover_type)
                t.sat_img['land_cover_type'].values = np.array(self.land_cover_type.sat_img['land_cover_type'] ==i,
                                                               dtype=float)
                t.sat_img = regridder(t.sat_img)
                t.sat_img = t.sat_img.rename({'land_cover_type': str(i)})
            else:
                t1 = np.array(self.land_cover_type.sat_img['land_cover_type'] ==i, dtype=float)
                t1 = regridder(t1)
                t.sat_img = t.sat_img.assign({str(i): (dims, t1)})
        var_list =  t.sat_img.data_vars
        t.sat_img = xr.concat([t.sat_img[var] for var in var_list], dim='vprm_classes')
        t.sat_img = t.sat_img.assign_coords({'vprm_classes': [int(c) for c in list(var_list)]})
        t.sat_img = t.sat_img.to_dataset(name='vegetation_fraction_map')   
        t.sat_img = t.sat_img.rename({'y': 'south_north', 'x': 'west_east'})
        day_of_the_year = np.array(self.sat_imgs.sat_img[self.time_key].values, dtype=np.int32)
        day_of_the_year += self.timestamp_start.timetuple().tm_yday
        kys = len(self.sat_imgs.sat_img[self.time_key].values)
        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({self.time_key:ky})['evi'].where(self.land_cover_type.sat_img['land_cover_type'].values == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        out_dims = ['vprm_classes', 'time']
        out_dims.extend(dims)
        ds_t_evi = copy.deepcopy(ds_out)
        ds_t_evi = ds_t_evi.assign({'evi': (out_dims, np.moveaxis(final_array, 0 ,1))})
        ds_t_evi = ds_t_evi.assign_coords({"time": day_of_the_year})
        ds_t_evi = ds_t_evi.assign_coords({"vprm_classes": veg_inds})
        ds_t_evi = ds_t_evi.rename({'y': 'south_north', 'x': 'west_east'})        

        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({self.time_key: ky})['lswi'].where(self.land_cover_type.sat_img['land_cover_type'].values == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        ds_t_lswi = copy.deepcopy(ds_out)
        ds_t_lswi = ds_t_lswi.assign({'lswi': (out_dims, np.moveaxis(final_array,0, 1))})
        ds_t_lswi = ds_t_lswi.assign_coords({"time": day_of_the_year})
        ds_t_lswi = ds_t_lswi.assign_coords({"vprm_classes": veg_inds})  
        ds_t_lswi = ds_t_lswi.rename({'y': 'south_north', 'x': 'west_east'})        
        
        out_dims = ['vprm_classes']
        out_dims.extend(dims)
        ds_t_max_evi = copy.deepcopy(ds_out)
        ds_t_max_evi = ds_t_max_evi.assign({'evi_max': (out_dims,
                                                     np.nanmax(ds_t_evi['evi'],axis = 1))})
        ds_t_max_evi = ds_t_max_evi.assign_coords({"vprm_classes": veg_inds})  
        ds_t_max_evi = ds_t_max_evi.rename({'y': 'south_north', 'x': 'west_east'})
        
        ds_t_min_evi = copy.deepcopy(ds_out)
        ds_t_min_evi = ds_t_min_evi.assign({'evi_min': (out_dims,
                                                     np.nanmin(ds_t_evi['evi'],axis = 1))})
        ds_t_min_evi = ds_t_min_evi.assign_coords({"vprm_classes": veg_inds})
        ds_t_min_evi = ds_t_min_evi.rename({'y': 'south_north', 'x': 'west_east'})
        
        ds_t_max_lswi = copy.deepcopy(ds_out)
        ds_t_max_lswi = ds_t_max_lswi.assign({'lswi_max': (out_dims,
                                                     np.nanmax(ds_t_lswi['lswi'],axis = 1))})
        ds_t_max_lswi = ds_t_max_lswi.assign_coords({"vprm_classes": veg_inds})  
        ds_t_max_lswi = ds_t_max_lswi.rename({'y': 'south_north', 'x': 'west_east'})
        
        ds_t_min_lswi = copy.deepcopy(ds_out)
        ds_t_min_lswi = ds_t_min_lswi.assign({'lswi_min': (out_dims,
                                                     np.nanmin(ds_t_lswi['lswi'],axis = 1))})
        ds_t_min_lswi = ds_t_min_lswi.assign_coords({"vprm_classes": veg_inds})
        ds_t_min_lswi = ds_t_min_lswi.rename({'y': 'south_north', 'x': 'west_east'})
        
        ret_dict = {'lswi': ds_t_lswi, 'evi': ds_t_evi, 'veg_fraction': t.sat_img,
                    'lswi_max': ds_t_max_lswi, 'lswi_min': ds_t_min_lswi,
                    'evi_max': ds_t_max_evi, 'evi_min': ds_t_min_evi}
        
        for key in ret_dict.keys():
            ret_dict[key] = ret_dict[key].assign_attrs(title="VPRM input data for WRF: {}".format(key),
                                                       MODIS_version = '061',
                                                       author = 'Dr. Theo Glauch',
                                                       institution1 = 'Heidelberg University',
                                                       institution2='Deutsches Zentrum für Luft- und Raumfahrt (DLR)',
                                                       contact = 'theo.glauch@dlr.de',
                                                       comment = 'Used VPRM classes: 1 Evergreen forest, 2 Deciduous forest, 3 Mixed forest, 4 Shrubland, 5 Trees and grasses, 6 Cropland, 7 Grassland, 8 Barren, Urban and built-up, water, permanent snow and ice')
        return ret_dict

    
    def add_sat_img(self, handler, b_nir=None,
                    b_red=None, b_blue=None,
                    b_swir=None,
                    drop_bands=False, 
                    which_evi=None,
                    timestamp_key=None,
                    smearing=False,
                    mask_bad_pixels=True,
                    mask_clouds=True):
        '''
            Add a new satellite image and calculate EVI and LSWI if desired

                Parameters:
                        handler (satellite_data_manager): The satellite image
                        b_nir (str): Name of the near-infrared band
                        b_red (str): Name of the red band
                        b_blue (str): Name of the blue band
                        b_swir (str): Name of the short-wave infrared band
                        drop_bands (bool): If True drop the raw band information after 
                                           calculation of EVI and LSWI. Saves memory.
                                           Can also be a list of keys to drop.
                        which_evi (str): Either evi or evi2. evi2 does not need a blue band.
                        timestamp_key (float): satellite data key containing a timestamp for each
                                               single pixel - to be used with lowess

                Returns:
                        None
        '''

        evi_params = {'g': 2.5, 'c1': 6., 'c2': 7.5, 'l': 1}
        evi2_params = {'g': 2.5, 'l': 1 , 'c': 2.4}
        if not isinstance(handler, satellite_data_manager):
            print('Satellite image needs to be an object of the sattelite_data_manager class')
            return  
        if which_evi in ['evi', 'evi2']:
            nir = handler.sat_img[b_nir] 
            red = handler.sat_img[b_red]
            swir = handler.sat_img[b_swir] 
            if which_evi=='evi':
                blue = handler.sat_img[b_blue] 
                temp_evi = (evi_params['g'] * ( nir - red)  / (nir + evi_params['c1'] * red - evi_params['c2'] * blue + evi_params['l']))
            elif which_evi=='evi2':
                temp_evi = (evi2_params['g'] * ( nir - red )  / (nir +  evi2_params['c'] * red + evi2_params['l']))
            temp_evi = xr.where((temp_evi<0) | (temp_evi>1) , np.nan, temp_evi)
           # temp_evi[temp_evi<0] = np.nan
            #temp_evi[temp_evi>1] = np.nan
            temp_lswi = ((nir - swir) / (nir + swir))
            temp_lswi = xr.where((temp_lswi<-1) | (temp_lswi>1) , np.nan, temp_lswi)
          #  temp_lswi[temp_lswi<-1] = np.nan
          # temp_lswi[temp_lswi>1] = np.nan
            handler.sat_img['evi'] = temp_evi
            handler.sat_img['lswi'] = temp_lswi
           # handler.sat_img = handler.sat_img.assign({'evi': (['y','x'], temp_evi)})
           # handler.sat_img = handler.sat_img.assign({'lswi': (['y','x'], temp_lswi)}) 
        if timestamp_key is not None:
            handler.sat_img = handler.sat_img.rename({timestamp_key: 'timestamps'})
        if self.sites is not None:
            if smearing:
                handler = self.smearing(sat_img=handler, lonlats=self.lonlats,
                                        keys=['evi', 'lswi'])
            handler.reduce_along_lat_lon(lon=[i[0] for i in self.lonlats],
                                         lat=[i[1] for i in self.lonlats],
                                         new_dim_name='site_names', 
                                         interp_method='nearest')
            handler.sat_img = handler.sat_img.assign_coords({'site_names': [i.get_site_name()
                                                                            for i in self.sites]})
        bands_to_mask = []
        for btm in [b_nir, b_red, b_blue, b_swir]:
            if btm is not None:
                  bands_to_mask.append(btm)
        if mask_bad_pixels:
            if bands_to_mask == []:   
                handler.mask_bad_pixels()
            else:
                handler.mask_bad_pixels(bands_to_mask)
        if mask_clouds:
            if bands_to_mask == []:   
                handler.mask_clouds()
            else:
                handler.mask_clouds(bands_to_mask)
        if drop_bands:
            if isinstance(drop_bands, list):
                drop_keys = drop_bands
                handler.sat_img = handler.sat_img.drop(drop_keys)
            else:
                handler.drop_bands()
        self.sat_imgs.append(handler)  
        self.timestamps.append(handler.get_recording_time())
        return

    
    def smearing(self, size=(3,3), sat_img=None, lonlats=None, keys=None):
        '''
            By default performs a spatial smearing on the list of pre-loaded satellite images.
            If sat_img is given the smearing is performed on that specific image.

                Parameters:
                        size (tuple): The extension of the spatial smoothing
                        lonlats (str): If given the smearing is only performed at the
                                       given lats and lons
                        keys (list): keys for the smoothign of the satellite images
                Returns:
                        None
        '''
        intern_call = False
        if sat_img is None:
            sat_img = self.sat_imgs
        else:
            intern_call = True
            sat_img = [sat_img]
            
        if keys is None:
            keys = list(sat_img[0].sat_img.data_vars)
            
        size = np.expand_dims(np.ones(shape=size)/np.sum(np.ones(shape=size)), 0)
        for img in sat_img:
            if lonlats is None:
                for key in keys:
                    img.sat_img[key][:,:] = convolve(img.sat_img[key].values[:,:,:], kernel=size, mode='nearest')
            else:
                t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                         img.sat_img.rio.crs)
                for ll in lonlats:
                    x, y = t.transform(ll[0], ll[1])
                    x_ind = np.argmin(np.abs(x - img.sat_img.coords['x'].values))
                    y_ind = np.argmin(np.abs(y - img.sat_img.coords['y'].values))
                    for key in keys:
                        img.sat_img[key][y_ind-20 : y_ind+20, x_ind-20 : x_ind+20] = \
                            convolve(img.sat_img[key][y_ind-20 : y_ind+20, x_ind-20 : x_ind+20],
                                           size=size, mode='nearest')
        if intern_call is True:
            return sat_img[0]
        else:
            return
        
    
    def sort_and_merge_by_timestamp(self):
        '''
            Called after adding the satellite images with 'add_sat_img'. Sorts the satellite
            images by timestamp and merges everything to one satellite_data_manager.

                Parameters:

                Returns:
                        None
        '''
        if self.sites is None:
            x_time_y = 0
            for h in self.sat_imgs:
                size_dict = dict(h.sat_img.sizes) 
                prod = np.prod([size_dict[i] for i in size_dict.keys()])
                if prod > x_time_y:
                    biggest = h
                    x_time_y = prod
            self.xs = biggest.sat_img.x.values
            self.ys = biggest.sat_img.y.values
            self.target_shape = (len(self.xs), len(self.ys))
            X, Y = np.meshgrid(self.xs, self.ys)
            t = Transformer.from_crs(biggest.sat_img.rio.crs,
                                    '+proj=longlat +datum=WGS84')
            self.x_long, self.y_lat = t.transform(X, Y) 
            self.prototype = copy.deepcopy(biggest) 
            keys = list(self.prototype.sat_img.keys())
            self.prototype.sat_img = self.prototype.sat_img.drop(keys)
          #  for h in self.sat_imgs:
          #      h.sat_img = h.sat_img.rio.reproject_match(self.prototype.sat_img, nodata=np.nan)
        else:
            self.prototype = copy.deepcopy(self.sat_imgs[0]) 
            keys = list(self.prototype.sat_img.keys()) 
            self.prototype.sat_img = self.prototype.sat_img.drop(keys)

        self.timestamps = np.array(self.timestamps).flatten()
        sort_inds = np.argsort(self.timestamps)
        self.timestamps = np.array([pd.Timestamp(i).to_pydatetime() for i in self.timestamps])[sort_inds]
        self.timestamp_start = self.timestamps[0]
        self.timestamp_end = self.timestamps[-1]
        self.tot_num_days = (self.timestamp_end - self.timestamp_start).days
        print('Loaded data from {} to {}'.format(self.timestamp_start, self.timestamp_end))
        day_steps = [i.days for i in (self.timestamps - self.timestamp_start)]
        if 'timestamps' in list(self.sat_imgs[0].sat_img.keys()):
            for sat_img_handler in self.sat_imgs:
                adjust_timestamps(sat_img_handler.sat_img, sat_img_handler.start_date(),
                                  sat_img_handler.stop_date(), self.timestamp_start)
            # else:
            #     Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(adjust_timestamps)(sat_img_handler.sat_img, sat_img_handler.start_date(), sat_img_handler.stop_date(), self.timestamp_start) for sat_img_handler in self.sat_imgs)
        if len(self.timestamps) == len(self.sat_imgs):
            self.sat_imgs = satellite_data_manager(sat_img = xr.concat([i.sat_img for i in np.array(self.sat_imgs)[sort_inds]], 'time'))
        else:
            self.sat_imgs = self.sat_imgs[0]
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign_coords({"time": day_steps})
        self.time_key = 'time'
        return

    def clip_to_box(self, sat_to_crop):
        bounds = sat_to_crop.sat_img.rio.bounds()
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.rio.clip_box(bounds[0],
                                                                   bounds[1],
                                                                   bounds[2],
                                                                   bounds[3])
        self.xs = self.sat_imgs.sat_img.x.values
        self.ys = self.sat_imgs.sat_img.y.values
        self.target_shape = (len(self.xs), len(self.ys))
        X, Y = np.meshgrid(self.xs, self.ys)
        t = Transformer.from_crs(self.sat_imgs.sat_img.rio.crs,
                                '+proj=longlat +datum=WGS84')
        self.x_long, self.y_lat = t.transform(X, Y) 
        keys = list(self.sat_imgs.sat_img.keys())
        self.prototype = satellite_data_manager(sat_img=self.sat_imgs.sat_img.drop(keys))
        return
    
    def add_land_cover_map(self, land_cover_map, var_name='band_1',
                           save_path=None, filter_size=None):
        '''
            Add the land cover map. Either use a pre-calculated one or do the calculation on the fly.

                Parameters:
                        land_cover_map (str or satimg instance): The input land cover map. 
                                                                 If string, assume it's a pre-generated map
                        var_name (str): Name of the land_cover_band in the xarray dataset
                        save_path (str): Path to save the map. Can be useful for re-using
                        filter_size (int): Number of pixels from which the land cover type is aggregated.
                Returns:
                        None
        '''
        if isinstance(land_cover_map, str):
            print('Load pre-generated land cover map')
            self.land_cover_type = satellite_data_manager(sat_img=land_cover_map)
        else:
            print('Generating satellite data compatible land cover map')
           # land_cover_map.reproject(proj=self.prototype.sat_img.rio.crs.to_proj4())
            if land_cover_map.sat_img.rio.crs.to_proj4() != self.sat_imgs.sat_img.rio.crs.to_proj4():
                print('Projection of land cover map and satellite images need to match. Reproject first.')
                return False
            # bounds = self.sat_imgs.sat_img.rio.bounds()
            # land_cover_map.sat_img = land_cover_map.sat_img.rio.clip_box(bounds[0], bounds[1],
            #                                                              bounds[2], bounds[3])
            for key in self.map_copernicus_to_vprm_class.keys():
                land_cover_map.sat_img[var_name].values[land_cover_map.sat_img[var_name].values==key] = self.map_copernicus_to_vprm_class[key]
            f_array = np.zeros(np.shape(land_cover_map.sat_img[var_name].values), dtype=np.int16)
            count_array = np.zeros(np.shape(land_cover_map.sat_img[var_name].values), dtype=np.int16)
            veg_inds = np.unique([self.map_copernicus_to_vprm_class[i] 
                                  for i in self.map_copernicus_to_vprm_class.keys()])
            if filter_size is None:
                filter_size = int(np.ceil(self.sat_imgs.sat_img.rio.resolution()[0]/land_cover_map.get_resolution()))
                print('Filter size {}:'.format(filter_size))
            for i in veg_inds:
                mask = np.array(land_cover_map.sat_img[var_name].values == i, dtype=np.float)
                ta  = scipy.ndimage.uniform_filter(mask, size=(filter_size, filter_size)) * (filter_size **2)
                f_array[ta>count_array] = i
                count_array[ta>count_array] = ta[ta>count_array]
            f_array[f_array==0]=8 # 8 is Category for nothing | alternatively np.nan?
            land_cover_map.sat_img[var_name].values = f_array
            del ta
            del count_array
            del f_array
            del mask
            t =  land_cover_map.sat_img.sel(x=self.xs, y=self.ys, method="nearest").to_array().values[0]
#            t =  land_cover_map.sat_img.sel(x=self.x_long, y=self.y_lat, method="nearest").to_array().values[0]
            self.land_cover_type = copy.deepcopy(self.prototype)
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.assign({'land_cover_type': (['y','x'], t)}) 
            if save_path is not None:
                 self.land_cover_type.save(save_path)
        return
    
    def calc_min_max_evi_lswi(self):
        '''
            Calculate the minimim and maximum EVI and LSWI
                Parameters:
                        None
                Returns:
                        None
        '''
        self.max_lswi = copy.deepcopy(self.prototype)
        self.min_max_evi = copy.deepcopy(self.prototype)
        shortcut = self.sat_imgs.sat_img
        # if self.sites is None:
        self.max_lswi.sat_img['max_lswi'] = shortcut['lswi'].max(self.time_key, skipna=True)
        self.min_max_evi.sat_img['min_evi'] = shortcut['evi'].min(self.time_key, skipna=True)
        self.min_max_evi.sat_img['max_evi'] = shortcut['evi'].max(self.time_key, skipna=True)
        self.min_max_evi.sat_img['th'] = shortcut['evi'].min(self.time_key, skipna=True) + 0.55 * ( shortcut['evi'].max(self.time_key, skipna=True) - shortcut['evi'].min(self.time_key, skipna=True))
            # self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'min_evi': (['y','x'], np.nanmin(shortcut['evi'], axis=0))})
            # self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'max_evi': (['y','x'], np.nanmax(shortcut['evi'], axis=0))})   
            # self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'th': (['y','x'], np.nanmin(shortcut['evi'], axis=0) + (0.55 * (np.nanmax(shortcut['evi'], axis=0) - np.nanmin(shortcut['evi'], axis=0))))})  
        # else:
        #     self.max_lswi.sat_img = self.max_lswi.sat_img.assign({'max_lswi': (['site_names'], np.nanmax(shortcut['lswi'], axis=0))})
        #     self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'min_evi': (['site_names'], np.nanmin(shortcut['evi'], axis=0))})
        #     self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'max_evi': (['site_names'], np.nanmax(shortcut['evi'], axis=0))})   
        #     self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'th': (['site_names'], np.nanmin(shortcut['evi'], axis=0) + (0.55 * (np.nanmax(shortcut['evi'], axis=0) - np.nanmin(shortcut['evi'], axis=0))))})              
        return
    
    def lowess(self, lonlats=None, keys=None, gap_filled=False, frac=0.25, it=3):
        '''
            Performs the lowess smoothing

                Parameters:
                        lonlats (str): If given the smearing is only performed at the
                                       given lats and lons
                Returns:
                        None
        '''
        
        if keys is None:
            keys = list(self.sat_imgs.sat_img.data_vars)
        if gap_filled:
            xvals = np.arange(self.tot_num_days)    
        else:
            xvals = self.sat_imgs.sat_img['time']
        for k in keys:
            if (not 'B' in k) & (not 'evi' in k) &\
               (not 'lswi' in k) & (not 'timestamps' in k):
                keys.remove(k)
        if self.sites is not None:  # Is flux tower sites are given        
            if 'timestamps' in keys:
                keys.remove('timestamps')
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({key: (['time_gap_filled', 'site_names'], np.array([do_lowess_smoothing(self.sat_imgs.sat_img.sel(site_names=i)[key].values, timestamps=self.sat_imgs.sat_img.sel(site_names=i)['timestamps'].values, xvals=xvals, frac=frac, it=it) for i in self.sat_imgs.sat_img.site_names.values]).T)})      
            else:
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({key: (['time_gap_filled', 'site_names'], np.array([do_lowess_smoothing(self.sat_imgs.sat_img.sel(site_names=i)[key].values, timestamps=self.sat_imgs.sat_img['time'].values, xvals=xvals, frac=frac, it=it) for i in self.sat_imgs.sat_img.site_names.values]).T)})

            
        elif lonlats is None: # If smoothing the entire array
            if 'timestamps' in keys:
                keys.remove('timestamps')
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({key: (['time_gap_filled', 'y', 'x'], np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,:,i].values, timestamps=self.sat_imgs.sat_img['timestamps'][:,:,i].values, xvals=xvals, frac=frac, it=it) for i, x_coord in enumerate(self.xs))).T)})
            else:
                for key in keys:
                    self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({key: (['time_gap_filled', 'y', 'x'], np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,:,i].values, timestamps=self.sat_imgs.sat_img['time'].values, xvals=xvals, frac=frac, it=it) for i, x_coord in enumerate(self.xs))).T)})
                
        else: # If smoothing only specific coordinates
            t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                     self.sat_imgs.sat_img.rio.crs)
            if 'timestamps' in keys:
                keys.remove('timestamps')
                for ll in lonlats:
                    x, y = t.transform(ll[0], ll[1])
                    x_ind = np.argmin(np.abs(x - self.sat_imgs.sat_img.coords['x'].values))
                    y_ind = np.argmin(np.abs(y - self.sat_imgs.sat_img.coords['y'].values))
                    for key in keys:
                        self.sat_imgs.sat_img[key][:, y_ind-10 : y_ind+10, x_ind-10 : x_ind+10] = np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,y_ind-10:y_ind+10, x_ind+i].values, timestamps=self.sat_imgs.sat_img['timestamps'][:,y_ind-10:y_ind+10, x_ind+i].values, xvals=xvals, frac=frac, it=it) for i in np.arange(-10, 10, 1))).T    
            else:
                for ll in lonlats:
                    x, y = t.transform(ll[0], ll[1])
                    x_ind = np.argmin(np.abs(x - self.sat_imgs.sat_img.coords['x'].values))
                    y_ind = np.argmin(np.abs(y - self.sat_imgs.sat_img.coords['y'].values))
                    for key in keys:
                        self.sat_imgs.sat_img[key][:, y_ind-10 : y_ind+10, x_ind-10 : x_ind+10] = np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,y_ind-10:y_ind+10, x_ind+i].values, timestamps=self.sat_imgs.sat_img['time'].values, xvals=xvals, frac=frac, it=it) for i in np.arange(-10, 10, 1))).T 
        
        self.time_key = 'time_gap_filled'
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign_coords({'time_gap_filled': list(xvals)}) 
        return

    def clip_values(self, key, min_val, max_val):
        self.sat_imgs.sat_img[key].values[self.sat_imgs.sat_img[key].values<min_val] = min_val
        self.sat_imgs.sat_img[key].values[self.sat_imgs.sat_img[key].values>max_val] = max_val
        return

    def clip_nans(self, key, val):
        self.sat_imgs.sat_img[key].values[np.isnan(self.sat_imgs.sat_img[key].values)] = val
        return
    
    def load_weather_data(self, hour, day, month, year, era_keys):
        '''
            Load meteorlocial data from the available (on DKRZ's levante) data storage

                Parameters:
                        hour (int): hour in UTC
                        day (int): day in UTC
                        month (int): month in UTC
                        year (int): year in UTC
                        era_keys (list): list of ERA5 variables using the shortNames. 
                                         See https://confluence.ecmwf.int/displau /CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
                Returns:
                        None
        '''
        if self.era5_inst is None:
            if self.verbose:
                print('Init ERA5 Instance')
            self.era5_inst = ERA5(year, month, day, hour, 
                                  keys=era_keys) 
        self.era5_inst.change_date(year=year, month=month,
                                   day=day, hour=hour)
        self.hour = hour
        self.day = day
        self.year = year
        self.month = month
        self.date = '{}-{}-{} {}:00:00'.format(year, month, day, hour)
        return
    
    def get_t_scale(self, lon=None, lat=None,
                    land_cover_type=None, temperature=None):
        '''
            Get VPRM t_scale

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        t_scale array
        '''

        if land_cover_type is not None:
            tmin = self.temp_coefficients[land_cover_type][0]
            topt = self.temp_coefficients[land_cover_type][1]
            tmax = self.temp_coefficients[land_cover_type][2]
            t = temperature
        elif lon is not None:
            tmin = self.t2m.value_at_lonlat(lon, lat, key='tmin', as_array=False).values.flatten()
            topt = self.t2m.value_at_lonlat(lon, lat, key='topt', as_array=False).values.flatten()
            tmax = self.t2m.value_at_lonlat(lon, lat, key='tmax', as_array=False).values.flatten()
            t = self.era5_inst.get_data(lonlat=(lon, lat), key='t2m').values.flatten() - 273.15 # to grad celsius
        else:
            tmin = self.t2m.sat_img['tmin'].values
            topt = self.t2m.sat_img['topt'].values
            tmax = self.t2m.sat_img['tmax'].values
            t = self.era5_inst.get_data(key='t2m').values - 273.15
        ret = (((t - tmin) * (t - tmax)) / ((t-tmin)*(t-tmax) - (t - topt)**2))
        if land_cover_type is not None:
            if (ret<0) | (t<tmin):
                ret= 0
            if t<tmin:
                t = tmin
        else:
            ret[(ret<0) | (t<tmin)] = 0   
            t[t<tmin] = tmin[t<tmin]
        return (t, ret)
        
    def get_p_scale(self, lon=None, lat=None,
                    site_name=None, land_type=None):
        '''
            Get VPRM p_scale for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        p_scale array
        '''

        if (self.new is False) & ('p_scale' in self.buffer.keys()):
            return self.buffer['p_scale']
        lswi = self.get_lswi(lon, lat, site_name)
        evi = self.get_evi(lon, lat, site_name)
        p_scale = ( 1 + lswi ) / 2
        if site_name is not None:
            th = self.min_max_evi.sat_img.sel(site_names=site_name)['th']
        elif lon is not None:
            land_type = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type', interp_method='nearest', as_array=False).values.flatten()
            th = self.min_max_evi.value_at_lonlat(lon, lat, key='th', interp_method='nearest', as_array=False).values.flatten()
        else:
            land_type = self.land_cover_type.sat_img['land_cover_type'].values
            th = self.min_max_evi.sat_img['th'].values
        mask1 = evi > th
        mask2 = land_type == 1
        if site_name is not None:
            if mask1 | mask2:    
                p_scale = 1
        else:
            p_scale[(mask1) | (mask2)] = 1
        self.buffer['p_scale'] = p_scale
        return p_scale 
    
    def get_current_timestamp(self):
        return self.timestamps[self.counter]
    
    def get_par(self, lon=None, lat=None, ssrd=None):
        '''
            Get VPRM par

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        par array
        '''
        if ssrd is not None:
            ret = ssrd / 0.505
        elif lon is not None:
            ret = self.era5_inst.get_data(lonlat=(lon, lat), key='ssrd').values.flatten() / 0.505 / 3600
        else:
            ret = self.era5_inst.get_data(key='ssrd') / 0.505 / 3600
        return ret
    
    def get_w_scale(self, lon=None, lat=None, site_name=None):
        '''
            Get VPRM w_scale

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        w_scale array
        '''

        if (self.new is False) & ('w_scale' in self.buffer.keys()):
            return self.buffer['w_scale']
        lswi = self.get_lswi(lon, lat, site_name)
        if site_name is not None:
            max_lswi = self.max_lswi.sat_img.sel(site_names=site_name)['max_lswi'].values
        elif lon is not None:
            max_lswi = self.max_lswi.value_at_lonlat(lon, lat, key='max_lswi', as_array=False).values.flatten()
        else:
            max_lswi = self.max_lswi.sat_img['max_lswi']
        self.buffer['w_scale'] = (1+lswi)/(1+max_lswi)
        return self.buffer['w_scale']
    
    def get_evi(self, lon=None, lat=None, site_name=None):
        '''
            Get EVI for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        EVI array
        '''

        if (self.new is False) & ('evi' in self.buffer.keys()):
            return self.buffer['evi']
        if site_name is not None:
            self.buffer['evi'] =  float(self.sat_imgs.sat_img.sel(site_names=site_name).isel({self.time_key :self.counter})['evi'].values)
        elif lon is not None:
            self.buffer['evi'] = self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key='evi', isel={self.time_key : self.counter}).values.flatten()
        else:
            self.buffer['evi'] = self.sat_imgs.sat_img['evi'].isel({self.time_key : self.counter}).values
        return self.buffer['evi']
        

    def get_sat_img_values_from_key(self, key, lon=None, lat=None,
                                    counter_range=None):
        '''
            Get EVI for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        EVI array
        '''
        
        if self.new is False:
            return self.buffer[key]
        if counter_range is None:
            select_dict = {self.time_key : self.counter}
        else:
            select_dict = {self.time_key : counter_range}
        
        if lon is not None:     
            self.buffer[key] = self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key=key, isel=select_dict).values.flatten()
        else:
            self.buffer[key] = self.sat_imgs.sat_img[key].isel(select_dict)
        return self.sat_img_buffer[key]
    
    def get_sat_img_values_for_all_keys(self, lon=None, lat=None,
                                        counter_range=None):
        '''
            Get EVI for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        EVI array
        '''
    
        if self.new is False:
            return self.buffer['all_sat_keys']
        
        if counter_range is None:
            select_dict = {self.time_key: self.counter}
        else:
            select_dict = {self.time_key: counter_range}
            
        if lon is not None:     
            self.buffer['all_sat_keys'] = self.sat_imgs.value_at_lonlat(lon, lat, as_array=True, isel=select_dict)
        else:
             self.buffer['all_sat_keys'] = self.sat_imgs.sat_img.isel(select_dict)
        return self.buffer['all_sat_keys']

    def get_lswi(self, lon=None, lat=None, site_name=None):
        '''
            Get LSWI for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        LSWI array
        '''

        if (self.new is False) & ('lswi' in self.buffer.keys()):
            return self.buffer['lswi']
        if site_name is not None:
            self.buffer['lswi'] =  float(self.sat_imgs.sat_img.sel(site_names=site_name).isel({self.time_key:self.counter})['lswi'].values)
        elif lon is not None:
            self.buffer['lswi'] =  self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key='lswi', isel={self.time_key: self.counter}).values.flatten()
        else:
            self.buffer['lswi'] = self.sat_imgs.sat_img['lswi'].isel({self.time_key: self.counter}).values
        return self.buffer['lswi']
   
    def init_temps(self):
        '''
            Initialize an xarray with the min, max, and opt temperatures for photosynthesis

                Parameters:
                Returns:
                        None
        '''
        if self.land_cover_type is None:
            return
        tmin = np.full(np.shape(self.land_cover_type.sat_img['land_cover_type'].values), 0)
        topt = np.full(np.shape(self.land_cover_type.sat_img['land_cover_type'].values), 20)
        tmax = np.full(np.shape(self.land_cover_type.sat_img['land_cover_type'].values), 40)
        veg_inds = np.unique([self.map_copernicus_to_vprm_class[i] 
                              for i in self.map_copernicus_to_vprm_class.keys()])
        for i in veg_inds:
            mask = (self.land_cover_type.sat_img['land_cover_type'].values  == i)
            tmin[mask] = self.temp_coefficients[i][0]
            topt[mask] = self.temp_coefficients[i][1]
            tmax[mask] = self.temp_coefficients[i][2]

        self.t2m = copy.deepcopy(self.prototype)
        self.t2m.sat_img = self.t2m.sat_img.assign({'tmin': (['y','x'], tmin),
                                                    'topt': (['y','x'], topt),
                                                    'tmax': (['y','x'], tmax)})  
        return

    def _set_sat_img_counter(self, datetime_utc):
        counter_new = (datetime_utc - self.timestamp_start).days
        counter_new = np.argmin(np.abs(self.sat_imgs.sat_img[self.time_key].values - counter_new))
        if (counter_new < 0) | (counter_new >= len(self.sat_imgs.sat_img[self.time_key])-1):
            print('No data for {}'.format(datetime_utc))
            self.counter = 0
            return False
        elif counter_new != self.counter:
            self.new = True
            self.counter = counter_new
            return
        else:
            self.new = False
            return # Still same satellite image
            
    def get_neural_network_variables(self, datetime_utc, lat=None, lon=None,
                                     era_variables=['ssrd', 't2m'], regridder_weights=None,
                                     sat_img_keys=None):

        '''
            Get the variables for an neural network based vegetation model

                Parameters:
                    datetime_utc (datetime): The END time of the 1-hour integration period
                    lat (float or list of floats): latitude (optional)
                    lon (float or list of floats): longitude (optional)
                    era_variables (list): ERA5 variables (optional)
                    regridder_weights (str): Path to the pre-computed weights for the ERA5 regridder (optional)
                    sat_img_keys (str): List of data_vars from the satellite images to be used (optional)
                Returns:
                        None
        '''
        
        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year
        self._set_sat_img_counter(datetime_utc)
        if sat_img_keys is None:
            sat_img_keys = list(self.sat_imgs.sat_img.data_vars)
            
        if self.prototype_lat_lon is None:
            src_x = self.prototype.sat_img.coords['x'].values
            src_y = self.prototype.sat_img.coords['y'].values
            X, Y = np.meshgrid(src_x, src_y)
            t = Transformer.from_crs(self.prototype.sat_img.rio.crs,
                                    '+proj=longlat +datum=WGS84')
            x_long, y_lat = t.transform(X, Y)
            self.prototype_lat_lon = xr.Dataset({"lon": (["y", "x"], x_long ,
                                 {"units": "degrees_east"}),
                                  "lat": (["y", "x"], y_lat,
                                 {"units": "degrees_north"})})
            self.prototype_lat_lon = self.prototype_lat_lon.set_coords(['lon', 'lat'])
        self.load_weather_data(hour, day, month,
                               year, era_keys=era_variables)
        if (lat is None) & (lon is None):
                self.era5_inst.regrid(dataset=self.prototype_lat_lon,
                                      weights=regridder_weights,
                                      n_cpus=self.n_cpus)
        ret_dict = dict()
        # sat_inds = np.concatenate([np.arange(self.counter-8, self.counter-2, 3),
        #                            np.arange(self.counter-2, self.counter+1)])
        for_ret_dict = self.get_sat_img_values_for_all_keys(counter_range=self.counter,
                                                            lon=lon, lat=lat)
        for i, key in enumerate(sat_img_keys):
            ret_dict[key] = for_ret_dict[key]
        for key in era_variables:
            if lat is None:
                ret_dict[key] = self.era5_inst.get_data(key=key)
            else:
                ret_dict[key] = self.era5_inst.get_data(lonlat=(lon, lat),
                                                        key=key).values.flatten()
        if lon is not None:
            land_type = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type',
                                                             interp_method='nearest', as_array=False).values.flatten()
        else:
            land_type = self.land_cover_type.sat_img['land_cover_type'].values
        land_type[(land_type != 1) & (land_type != 2) & (land_type != 3) & \
                  (land_type != 4) & (land_type != 6) & (land_type != 7)] = 0
        land_type[~np.isfinite(land_type)] = 0 
        ret_dict['land_cover_type'] = land_type
        return ret_dict           
    
    
    def data_for_fitting(self):
        
        for s in self.sites:
            self.new = True
            site_name  = s.get_site_name()
            ret_dict = dict()
            for k in ['evi', 'Ps', 'par', 
                      'Ts', 'Ws', 'lswi',
                      'tcorr']:
                ret_dict[k] = []
            drop_rows = []
            for index, row in s.get_data().iterrows():
                img_status = self._set_sat_img_counter(row['datetime_utc'])
                if img_status == False:
                    drop_rows.append(index)
                    continue
                ret_dict['evi'].append(self.get_evi(site_name=site_name))
                ret_dict['Ps'].append(self.get_p_scale(site_name=site_name, land_type=s.get_land_type()))
                ret_dict['par'].append(self.get_par(ssrd=row['ssrd']))
                Ts_all = self.get_t_scale(land_cover_type=s.get_land_type(), temperature=row['t2m'])
                ret_dict['Ts'].append(Ts_all[1])
                ret_dict['tcorr'].append(Ts_all[0])
                ret_dict['Ws'].append(self.get_w_scale(site_name=site_name))
                ret_dict['lswi'].append(self.get_lswi(site_name=site_name))
            s.drop_rows_by_index(drop_rows)
            s.add_columns(ret_dict)
        return self.sites
    
    def _get_vprm_variables(self, datetime_utc=None, lat=None, lon=None,
                            add_era_variables=[], regridder_weights=None):

        '''
            Get the variables for the Vegetation Photosynthesis and Respiration Model

                Parameters:
                    datetime_utc (datetime): The time of interest
                    lat (float): A latitude (optional)
                    lon (float): A longitude (optional)
                    add_era_variables (list): Additional era variables for modifications of the VPRM
                    regridder_weights (str): Path to the pre-computed weights for the ERA5 regridder
                    tower_dict (dict): Alternatively to a model meteorology and land cover map also the data from the flux tower can be passed in a dictionary Minimaly required are the variables 't2m', 'ssrd', 'land_cover_type'
                Returns:
                        None
        '''
        
        era_keys = ['ssrd', 't2m']       
        era_keys.extend(add_era_variables)
            
        if (self.t2m is None):
                self.init_temps()
        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year

        img_status = self._set_sat_img_counter(datetime_utc)
        if img_status is False:
            return None

        if (lat != self.buffer['cur_lat']) | (lon != self.buffer['cur_lon']):
            self.new = True # Change in lat lon needs new query from satellite images
            self.buffer['cur_lat'] = lat
            self.buffer['cur_lon'] = lon 

        if len(era_keys) > 0:
            if self.prototype_lat_lon is None:
                src_x = self.prototype.sat_img.coords['x'].values
                src_y = self.prototype.sat_img.coords['y'].values
                X, Y = np.meshgrid(src_x, src_y)
                t = Transformer.from_crs(self.prototype.sat_img.rio.crs,
                                        '+proj=longlat +datum=WGS84')
                x_long, y_lat = t.transform(X, Y)
                self.prototype_lat_lon = xr.Dataset({"lon": (["y", "x"], x_long ,
                                     {"units": "degrees_east"}),
                                      "lat": (["y", "x"], y_lat,
                                     {"units": "degrees_north"})})
                self.prototype_lat_lon = self.prototype_lat_lon.set_coords(['lon', 'lat'])
            self.load_weather_data(hour, day, month,
                                   year, era_keys=era_keys)
            if (lat is None) & (lon is None):
                    self.era5_inst.regrid(dataset=self.prototype_lat_lon,
                                          weights=regridder_weights,
                                          n_cpus=self.n_cpus)

        ret_dict = dict()
        ret_dict['evi'] = self.get_evi(lon, lat)
        ret_dict['Ps'] = self.get_p_scale(lon, lat)
        ret_dict['par'] = self.get_par(lon, lat)
        Ts_all = self.get_t_scale(lon, lat)
        ret_dict['Ts'] = Ts_all[1]
        ret_dict['Ws'] = self.get_w_scale(lon, lat)
        ret_dict['tcorr'] = Ts_all[0]
        if add_era_variables!=[]:
            for i in add_era_variables:
                if lon is not None:
                    ret_dict[i] = self.era5_inst.get_data(lonlat=(lon, lat), key=i).values.flatten()
                else:
                    ret_dict[i] = self.era5_inst.get_data(key=i).values.flatten()
        return ret_dict  
    
    def make_vprm_predictions(self, date, fit_params_dict=None,
                              regridder_weights=None,
                              no_flux_veg_types=[0, 5, 8]):
        '''
            Using the VPRM fit parameters make predictions on the entire satellite image.

                Parameters:
                    date (datetime object): The date for the prediction
                    fit_params_dict (dict) : Dict with fit parameters ('lamb', 'par0', 'alpha', 'beta') 
                                      for the different vegetation types.
                    regridder_weights (str): Path to the weights file for regridding from ERA5 
                                             to the satellite grid
                    no_flux_veg_types (list of ints): flux type ids that get a default GPP/NEE of 0
                                                      (e.g. oceans, deserts...)                     
                Returns:
                        None
        '''
        
        if self.res == None:
            if fit_params_dict is None:
                print('Need to provide a dictionary with the fit parameters')
                return 
            else:
                for i in no_flux_veg_types:
                    fit_params_dict [i] = {'lamb': 0, 'par0': 0,
                                   'alpha': 0, 'beta': 0}
                self.res = copy.deepcopy(self.land_cover_type) 
                keys = list(self.land_cover_type.sat_img.keys())
                self.res.sat_img = self.res.sat_img.drop(keys) 
                t = np.array(self.land_cover_type.sat_img['land_cover_type'].values, dtype=np.float64)
                t_shape = np.shape(t)
                lamb, par0, alpha, beta = copy.deepcopy(t), copy.deepcopy(t), copy.deepcopy(t), copy.deepcopy(t)
                for i in range(t_shape[0]):
                    for j in range(t_shape[1]):
                        if np.isnan(t[i,j]):
                            lamb[i,j], par0[i,j], alpha[i,j], beta[i,j] = np.nan, np.nan, np.nan, np.nan
                        else:
                            lamb[i,j] = fit_params_dict[int(t[i,j])]['lamb']
                            par0[i,j] = fit_params_dict[int(t[i,j])]['par0']
                            alpha[i,j] = fit_params_dict[int(t[i,j])]['alpha']
                            beta[i,j] = fit_params_dict[int(t[i,j])]['beta']
                self.res.sat_img = self.res.sat_img.assign({'lamb': (['y','x'], lamb),
                                                            'par0': (['y','x'], par0),
                                                            'alpha': (['y','x'], alpha),
                                                            'beta': (['y','x'], beta)}) 
        if not os.path.exists(os.path.dirname(regridder_weights)):
            os.makedirs(os.path.dirname(regridder_weights))

        inputs = self._get_vprm_variables(date, regridder_weights=regridder_weights)
        
        if inputs is None:
            return None
        
        ret_res = dict()
        gpp = (self.res.sat_img['lamb'].values * inputs['Ps'] * inputs['Ws'] * inputs['Ts']) * inputs['evi'] * inputs['par'] / (1 + inputs['par']/self.res.sat_img['par0'].values)
        ret_res['gpp'] = gpp
        ret_res['nee'] = -gpp + self.res.sat_img['alpha'].values * inputs['tcorr'] + self.res.sat_img['beta'].values
        return ret_res
        

    def save(self, save_path):
        '''
            Save the LSWI and EVI satellite image. ToDo
        '''
        self.sat_imgs.save(save_path)
        return
    

    def fit_vprm_data(self, data_list, variable_dict, same_length=True):
        '''
            Run a VPRM fit
            Parameters:
                data_list (list): A list of instances from type flux_tower_data
                variable_dict (dict): A dictionary giving the target keys for gpp und respiration                           
                                      i.e. {'gpp': 'GPP_DT_VUT_REF', 'respiration': 'RECO_NT_VUT_REF'}           
                same_length (bool): If true all sites have the same number of input data for the fit.
            Returns:
                A dictionary with the fit parameters
        '''        

        variable_dict = {v: k for k, v in variable_dict.items()}
        fit_dict = dict()
        for i in data_list:
            lt = i.get_land_type()
            if lt in fit_dict.keys():
                fit_dict[lt].append(i)
            else:
                fit_dict[lt] = [i]
        best_fit_params_dict = dict()
        for key in fit_dict.keys():
            min_len = np.min([i.get_len() for i in fit_dict[key]])
            data_for_fit = []
            for s in fit_dict[key]:
                t_data = s.get_data()
                if same_length:
                    if len(t_data) > min_len:
                        inds = np.random.choice(np.arange(len(t_data)), min_len, replace=False)
                        t_data = t_data.iloc[inds]
                data_for_fit.append(t_data.rename(variable_dict, axis=1))
            data_for_fit = pd.concat(data_for_fit)

            best_mse = np.inf    
            for i in range(100):
                func = lambda x, a, b: a * x['tcorr'] + b
                fit_respiration = curve_fit(func, data_for_fit, data_for_fit['respiration'], maxfev=5000,
                                            p0=[np.random.uniform(-0.5, 0.5),
                                                np.random.uniform(-0.5, 0.5)]) 
                mse = np.mean((func(data_for_fit, fit_respiration[0][0], fit_respiration[0][1]) - data_for_fit['respiration'])**2)
                if mse < best_mse:
                    best_mse = mse
                    best_fit_params = fit_respiration
                best_fit_params_dict[key] = {'alpha': best_fit_params[0][0],
                                             'beta': best_fit_params[0][1]}

            #GPP
            best_mse = np.inf
            for i in range(100):  
                func = lambda x, lamb, par0: (lamb * data_for_fit['Ws'] * data_for_fit['Ts'] * data_for_fit['Ps']) * data_for_fit['evi'] * data_for_fit['par'] / (1 + data_for_fit['par']/par0) 
                fit_gpp = curve_fit(func,
                                    data_for_fit, data_for_fit['gpp'], maxfev=5000,
                                    p0=[np.random.uniform(0,1), np.random.uniform(0,1000)]) 
                mse = np.mean((func(data_for_fit, fit_gpp[0][0], fit_gpp[0][1]) - data_for_fit['gpp'])**2)
                if mse < best_mse:
                    best_mse = mse
                    best_fit_params = fit_gpp 
                best_fit_params_dict[key]['lamb'] = best_fit_params[0][0]
                best_fit_params_dict[key]['par0'] = best_fit_params[0][1]


        return best_fit_params_dict

    def is_disjoint(self, this_sat_img):
        bounds = self.prototype.sat_img.rio.transform_bounds(this_sat_img.sat_img.rio.crs)
        dj = rasterio.coords.disjoint_bounds(bounds, this_sat_img.sat_img.rio.bounds())
        return dj

    
    def add_vprm_insts(self, vprm_insts):          

        if isinstance(self.sat_imgs, satellite_data_manager):
            self.sat_imgs.add_tile([v.sat_imgs for v in vprm_insts],
                                    reproject=False)
            self.xs = self.sat_imgs.sat_img.x.values
            self.ys = self.sat_imgs.sat_img.y.values
            self.target_shape = (len(self.xs), len(self.ys))
            X, Y = np.meshgrid(self.xs, self.ys)
            t = Transformer.from_crs(self.sat_imgs.sat_img.rio.crs,
                                    '+proj=longlat +datum=WGS84')
            self.x_long, self.y_lat = t.transform(X, Y) 
            keys = list(self.sat_imgs.sat_img.keys())
            self.prototype = satellite_data_manager(sat_img=self.sat_imgs.sat_img.drop(keys))
            
    
        if self.land_cover_type is not None:
            self.land_cover_type.add_tile([v.land_cover_type for v in vprm_insts],
                                    reproject=False)
        