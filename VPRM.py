import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve()))
import numpy as np
from lib.sat_manager import VIIRS, sentinel2, modis,\
                            copernicus_land_cover_map, satellite_data_manager
from lib.era5_class_new import ERA5
from scipy.ndimage import uniform_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import copy
from joblib import Parallel, delayed
import xarray as xr
import xesmf as xe
import scipy
import uuid
import time

def do_lowess_smoothing(array_to_smooth, vclass=None):
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
    for j in range(np.shape(array_to_smooth)[1]):
        if vclass!=None:
            if vclass[j] == 8: # skip arrays with land type class 8 
                continue
        array_to_smooth[:, j] = lowess(array_to_smooth[:, j], range(len(array_to_smooth[:, j])),
                     is_sorted=True, frac=0.2, it=1, return_sorted=False)
    return array_to_smooth.T

class vprm: 
    '''
    Class for the  Vegetation Photosynthesis and Respiration Model
    '''
    def __init__(self, land_cover_map=None, verbose=False, n_cpus=1):
        '''
            Initialize a class instance

            Parameters:
                    land_cover_map (xarray): A pre calculated map with the land cover types
                    verbose (bool): Set true for additional output when debugging

            Returns:
                    The lowess smoothed array
        '''

        self.sat_imgs = []
        self.n_cpus = n_cpus
        self.era5_inst = None
        self.land_cover_map = land_cover_map
        self.verbose = verbose
        self.counter = 0
        self.evis = None
        self.lswis = None
        self.timestamps = []
        self.res_dict = None
        self.ssrd = None
        self.t2m = None
        self.target_shape = None
        self.prototype_lat_lon = None
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
                                             116:3, 121:5, 123:5,
                                             122 : 5, 124 : 5,
                                             125 : 5, 126: 5, 
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
                os.system('mpirun -np {} ESMF_RegridWeightGen --source {} --destination {} --weight {} -m bilinear -r --64bit_offset  --extrap_method nearestd  --no_log'.format(self.n_cpus, src_temp_path, dest_temp_path, regridder_save_path))
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
                t.sat_img = t.sat_img.assign({str(i): (['lat','lon'], t1)})
        var_list =  t.sat_img.data_vars
        t.sat_img = xr.concat([t.sat_img[var] for var in var_list], dim='vprm_classes')
        t.sat_img = t.sat_img.assign_coords({'vprm_classes': [int(c) for c in list(var_list)]})
        t.sat_img = t.sat_img.to_dataset(name='vegetation_fraction_map')
        
        day_of_the_year = [i.timetuple().tm_yday for i in self.timestamps]
        kys = self.sat_imgs.sat_img.time.size
        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({'time':ky})['evi'].where(self.land_cover_type.sat_img['land_cover_type'].values == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        ds_t_evi = copy.deepcopy(ds_out)
        ds_t_evi = ds_t_evi.assign({'evi': (['time', 'vprm_classes','lat','lon'], final_array)})
        ds_t_evi = ds_t_evi.assign_coords({"time": day_of_the_year})
        ds_t_evi = ds_t_evi.assign_coords({"vprm_classes": veg_inds})
        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({'time': ky})['lswi'].where(self.land_cover_type.sat_img['land_cover_type'].values == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        ds_t_lswi = copy.deepcopy(ds_out)
        ds_t_lswi = ds_t_lswi.assign({'lswi': (['time', 'vprm_classes','lat','lon'], final_array)})
        ds_t_lswi = ds_t_lswi.assign_coords({"time": day_of_the_year})
        ds_t_lswi = ds_t_lswi.assign_coords({"vprm_classes": veg_inds})  
        
        ds_t_max_evi = copy.deepcopy(ds_out)
        ds_t_max_evi = ds_t_max_evi.assign({'evi_max': (['vprm_classes','lat','lon'],
                                                     np.nanmax(ds_t_evi['evi'],axis = 0))})
        ds_t_max_evi = ds_t_max_evi.assign_coords({"vprm_classes": veg_inds})  
        
        ds_t_min_evi = copy.deepcopy(ds_out)
        ds_t_min_evi = ds_t_min_evi.assign({'evi_min': (['vprm_classes','lat','lon'],
                                                     np.nanmin(ds_t_evi['evi'],axis = 0))})
        ds_t_min_evi = ds_t_min_evi.assign_coords({"vprm_classes": veg_inds})
        
        ds_t_max_lswi = copy.deepcopy(ds_out)
        ds_t_max_lswi = ds_t_max_lswi.assign({'lswi_max': (['vprm_classes','lat','lon'],
                                                     np.nanmax(ds_t_lswi['lswi'],axis = 0))})
        ds_t_max_lswi = ds_t_max_lswi.assign_coords({"vprm_classes": veg_inds})  
        
        ds_t_min_lswi = copy.deepcopy(ds_out)
        ds_t_min_lswi = ds_t_min_lswi.assign({'lswi_min': (['vprm_classes','lat','lon'],
                                                     np.nanmin(ds_t_lswi['lswi'],axis = 0))})
        ds_t_min_lswi = ds_t_min_lswi.assign_coords({"vprm_classes": veg_inds})

        ret_dict = {'lswi': ds_t_lswi, 'evi': ds_t_evi, 'veg_fraction': t.sat_img,
                    'lswi_max': ds_t_max_lswi, 'lswi_min': ds_t_min_lswi,
                    'evi_max': ds_t_max_evi, 'evi_min': ds_t_min_evi}
        return ret_dict
    
    def add_sat_img(self, handler, b_nir=None,
                    b_red=None, b_blue=None,
                    b_swir=None,
                    drop_bands=True, 
                    which_evi=None,
                    max_evi=False):
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
                        which_evi (str): Either evi or evi2. evi2 does not need a blue band.
                        max_evi (float): If given, every evi larger than this will be cut.

                Returns:
                        None
        '''

        evi_params = {'g': 2.5, 'c1': 6., 'c2': 7.5, 'l': 1}
        evi2_params = {'g': 2.5, 'l': 1 , 'c': 2.4}
        if not isinstance(handler, satellite_data_manager):
            print('Satellite image needs to be an object of the sattelite_data_manager class')
            return  
        handler.sat_img = handler.sat_img.reindex(y=sorted(list(handler.sat_img.y)))
        handler.sat_img = handler.sat_img.reindex(x=sorted(list(handler.sat_img.x)))
        
        if isinstance(drop_bands, list):
            drop_keys = drop_bands
        else:
            drop_keys = list(handler.sat_img.keys())
            
        if which_evi in ['evi', 'evi2']:
            nir = handler.sat_img[b_nir] 
            red = handler.sat_img[b_red]  
            swir = handler.sat_img[b_swir] 
            if which_evi=='evi':
                blue = handler.sat_img[b_blue] 
                temp_evi = (evi_params['g'] * ( nir - red)  / (nir + evi_params['c1'] * red - evi_params['c2'] * blue + evi_params['l'])).values
            elif which_evi=='evi2':
                temp_evi = (evi2_params['g'] * ( nir - red )  / (nir +  evi2_params['c'] * red + evi2_params['l'])).values 
            temp_lswi = ((nir -  swir) / (nir + swir)).values
            if max_evi is not False:
                temp_evi[temp_evi>max_evi] = max_evi
                temp_evi[~np.isfinite(temp_evi)] = max_evi
            handler.sat_img = handler.sat_img.assign({'evi': (['y','x'], temp_evi)})
            handler.sat_img = handler.sat_img.assign({'lswi': (['y','x'], temp_lswi)}) 
        if drop_bands:
                handler.sat_img = handler.sat_img.drop(drop_keys)
        self.sat_imgs.append(handler)  
        self.timestamps.append(handler.get_recording_time())
        return

    
    def smearing(self, size=(3,3), lonlats=None, keys=None):
        '''
            Performs a spatial smearing on the list of pre-loaded satellite images

                Parameters:
                        size (tuple): The extension of the spatial smoothing
                        lonlats (str): If given the smearing is only performed at the
                                       given lats and lons
                        keys (list): keys for the smoothign of the satellite images
                Returns:
                        None
        '''
        
        if keys is None:
            keys = list(self.sat_imgs[0].sat_img.data_vars)
            
        for img in self.sat_imgs:
            if lonlats is None:
                for key in keys:
                    img.sat_img[key][:,:] = uniform_filter(img.sat_img[key].values[:,:], size=size, mode='nearest')
            else:
                t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                         img.sat_img.rio.crs)
                for ll in lonlats:
                    x, y = t.transform(ll[0], ll[1])
                    x_ind = np.argmin(np.abs(x - img.sat_img.coords['x'].values))
                    y_ind = np.argmin(np.abs(y - img.sat_img.coords['y'].values))
                    for key in keys:
                        img.sat_img[key][y_ind-10 : y_ind+10, x_ind-10 : x_ind+10] = \
                            uniform_filter(img.sat_img[key][y_ind-10 : y_ind+10, x_ind-10 : x_ind+10],
                                           size=size, mode='nearest')

        return

    
    def sort_and_merge_by_timestamp(self):
        '''
            Called after adding the satellite images with 'add_sat_img'. Sorts the satellite
            images by timestamp and merges everything to one satellite_data_manager.

                Parameters:

                Returns:
                        None
        '''
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
        sort_inds = np.argsort(self.timestamps)
        for h in self.sat_imgs:
            h.sat_img = h.sat_img.rio.reproject_match(self.prototype.sat_img, nodata=np.nan)

        self.timestamps = np.array(self.timestamps)[sort_inds]
        day_of_the_year = [i.timetuple().tm_yday for i in self.timestamps]
        self.sat_imgs = satellite_data_manager(sat_img = xr.concat([i.sat_img for i in np.array(self.sat_imgs)[sort_inds]], 'time'))
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign_coords({"time": day_of_the_year})
        return

    
    def add_land_cover_map(self, land_cover_map, var_name='band_1',
                           save_path=None, filter_size=5):
        '''
            Add the land cover map. Either use a pre-calculated one or do the calculation on the fly.

                Parameters:
                        land_cover_map (int): Number of parallel CPUs for the smoothing
                        var_name (str): Name of the band in the output xarray
                        save_path (str): Path to save the map. Can be useful for re-using
                        filter_size (int): Number of pixels from which the land cover type is aggregated.
                Returns:
                        None
        '''
        if isinstance(land_cover_map, str):
            print('Load pre-generated land cover map')
            self.land_cover_type = copernicus_land_cover_map(land_cover_map)
            self.land_cover_type.load()
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.reindex(y=sorted(list(self.land_cover_type.sat_img.y)))
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.reindex(x=sorted(list(self.land_cover_type.sat_img.x)))
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.rename({list(self.land_cover_type.sat_img.keys())[0]: 'land_cover_type'})
 
        else:
            print('Aggregating land cover map on vprm land cover types and projecting on sat image grids. This step takes time and memory')
            land_cover_map.reproject(proj=self.prototype.sat_img.rio.crs.to_proj4())
            for key in self.map_copernicus_to_vprm_class.keys():
                land_cover_map.sat_img[var_name].values[land_cover_map.sat_img[var_name].values==key] = self.map_copernicus_to_vprm_class[key]
            f_array = np.zeros(np.shape(land_cover_map.sat_img[var_name].values))
            count_array = np.zeros(np.shape(land_cover_map.sat_img[var_name].values))
            veg_inds = np.unique([self.map_copernicus_to_vprm_class[i] 
                                  for i in self.map_copernicus_to_vprm_class.keys()])
            if filter_size is not None:
                for i in veg_inds:
                    mask = np.array(land_cover_map.sat_img[var_name].values == i, dtype=float)
                    ta  = scipy.ndimage.uniform_filter(mask, size=(filter_size, filter_size)) * 25
                    f_array[ta>count_array] = i 
                    count_array[ta>count_array] = ta[ta>count_array]
                f_array[f_array==0]=np.nan
                land_cover_map.sat_img[var_name].values = f_array
                del ta
                del count_array
                del f_array
                del mask
            land_cover_map.sat_img = land_cover_map.sat_img.reindex(y=sorted(list(land_cover_map.sat_img.y)))
            land_cover_map.sat_img = land_cover_map.sat_img.reindex(x=sorted(list(land_cover_map.sat_img.x)))
            t =  land_cover_map.sat_img.sel(x=self.xs, y=self.ys, method="nearest").to_array().values[0]
            self.land_cover_type = copy.deepcopy(self.prototype)
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.assign({'land_cover_type': (['y','x'], t)}) 
            if save_path is not None:
                 self.land_cover_type.sat_img.rio.to_raster(save_path)
        return
    
    def calc_min_max_evi_lswi(self):
        '''
            Add the land cover map. Either use a pre-calculated one or do the calculation on the fly.

                Parameters:
                        land_cover_map (int): Number of parallel CPUs for the smoothing
                        var_name (str): Name of the band in the output xarray
                        save_path (str): Path to save the map. Can be useful for re-using
                        filter_size (int): Number of pixels from which the land cover type is aggregated.
                Returns:
                        None
        '''
        self.max_lswi = copy.deepcopy(self.prototype)
        self.min_max_evi = copy.deepcopy(self.prototype)
        shortcut = self.sat_imgs.sat_img
        self.max_lswi.sat_img = self.max_lswi.sat_img.assign({'max_lswi': (['y','x'], np.nanmax(shortcut['lswi'], axis=0))})
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'min_evi': (['y','x'], np.nanmin(shortcut['evi'], axis=0))})
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'max_evi': (['y','x'], np.nanmax(shortcut['evi'], axis=0))})   
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'th': (['y','x'], np.nanmin(shortcut['evi'], axis=0) + (0.55 * (np.nanmax(shortcut['evi'], axis=0) - np.nanmin(shortcut['evi'], axis=0))))})  
        return
    
    def lowess(self, lonlats=None, keys=None):
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
            
        if lonlats is None:
            for key in keys:
                self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({key: (['time', 'y', 'x'], np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,:,i].values) for i, x_coord in enumerate(self.xs))).T)})
                self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({key: (['time', 'y', 'x'], np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,:,i].values) for i, x_coord in enumerate(self.xs))).T)})
        else:
            t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                     self.sat_imgs.sat_img.rio.crs)
            for ll in lonlats:
                x, y = t.transform(ll[0], ll[1])
                x_ind = np.argmin(np.abs(x - self.sat_imgs.sat_img.coords['x'].values))
                y_ind = np.argmin(np.abs(y - self.sat_imgs.sat_img.coords['y'].values))
                for key in keys:
                    self.sat_imgs.sat_img[key][:, y_ind-10 : y_ind+10, x_ind-10 : x_ind+10] = np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,y_ind-10:y_ind+10, x_ind+i].values) for i in np.arange(-10, 10, 1))).T              
                    self.sat_imgs.sat_img[key][:, y_ind-10 : y_ind+10, x_ind-10 : x_ind+10] = np.array(Parallel(n_jobs=self.n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img[key][:,y_ind-10:y_ind+10, x_ind+i].values) for i in np.arange(-10, 10, 1))).T 
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
                                         See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
                Returns:
                        None
        '''
        if self.era5_inst is None:
            if self.verbose:
                print('Init ERA5 Instance')
            self.era5_inst = ERA5(year, month, day, hour, 
                                  keys=era_keys) 
        self.era5_inst.change_date(hour=hour, day=day,
                                   month=month, year=year)
        self.hour = hour
        self.day = day
        self.year = year
        self.month = month
        self.date = '{}-{}-{} {}:00:00'.format(year, month, day, hour)
        return
    
    def get_t_scale(self, lon=None, lat=None):
        '''
            Get VPRM t_scale

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        t_scale array
        '''

        if lon is not None:
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
        ret[t<0] = 0
        t[t<0] = tmin[t<0]
        return (t, ret)
        
    def get_p_scale(self, lon=None, lat=None):
        '''
            Get VPRM p_scale for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        p_scale array
        '''

        lswi = self.get_lswi(lon, lat)
        evi = self.get_evi(lon, lat)
        p_scale = ( 1 + lswi ) / 2
        if lon is not None:
            land_type = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type', as_array=False).values.flatten()
            th = self.min_max_evi.value_at_lonlat(lon, lat, key='th', as_array=False).values.flatten()
        else:
            land_type = self.land_cover_type.sat_img['land_cover_type'].values
            th = self.min_max_evi.sat_img['th'].values
        mask1 = evi > th
        mask2 = land_type == 1
        p_scale[(mask1) | (mask2)] = 1
        return p_scale 
    
    def get_current_timestamp(self):
        return self.timestamps[self.counter]
    
    def get_par(self, lon=None, lat=None):
        '''
            Get VPRM par

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        par array
        '''

        if lon is not None:
            ret = self.era5_inst.get_data(lonlat=(lon, lat), key='ssrd').values.flatten() / 0.505 / 3600
        else:
            ret = self.era5_inst.get_data(key='ssrd')/ 0.505 / 3600
        return ret
    
    def get_w_scale(self, lon=None, lat=None):
        '''
            Get VPRM w_scale

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        w_scale array
        '''

        lswi = self.get_lswi(lon, lat)
        if lon is not None:
            max_lswi = self.max_lswi.value_at_lonlat(lon, lat, key='max_lswi', as_array=False).values.flatten()
        else:
            max_lswi = self.max_lswi.sat_img['max_lswi']
        return (1+lswi)/(1+max_lswi)
    
    def get_evi(self, lon=None, lat=None):
        '''
            Get EVI for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        EVI array
        '''

        if lon is not None:     
            return self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key='evi', isel={'time': self.counter}).values.flatten()
        else:
            return self.sat_imgs.sat_img['evi'].isel({'time': self.counter}).values
        

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
        
        if counter_range is None:
            select_dict = {'time': self.counter}
        else:
            select_dict = {'time': counter_range}
            
        if lon is not None:     
            return self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key=key, isel=select_dict).values.flatten()
        else:
            return self.sat_imgs.sat_img[key].isel(select_dict)
        return
    
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
        
        if counter_range is None:
            select_dict = {'time': self.counter}
        else:
            select_dict = {'time': counter_range}
            
        if lon is not None:     
            return self.sat_imgs.value_at_lonlat(lon, lat, as_array=True, isel=select_dict)
        else:
            return self.sat_imgs.sat_img.isel(select_dict)
        
        return

    def get_lswi(self, lon=None, lat=None):
        '''
            Get LSWI for the current satellite image (see self.counter)

                Parameters:
                        lon (float): longitude
                        lat (float): latitude
                Returns:
                        LSWI array
        '''

        if lon is not None:
            return self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key='lswi', isel={'time': self.counter}).values.flatten()
        else:
            return self.sat_imgs.sat_img['lswi'].isel({'time': self.counter}).values
   
    def init_temps(self):
        '''
            Initialize an xarray with the min, max, and opt temperatures for photosynthesis

                Parameters:
                Returns:
                        None
        '''
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
        self.t2m.sat_img = self.t2m.sat_img.assign({'tmin': (['y','x'], tmin)}) 
        self.t2m.sat_img = self.t2m.sat_img.assign({'topt': (['y','x'], topt)}) 
        self.t2m.sat_img = self.t2m.sat_img.assign({'tmax': (['y','x'], tmax)}) 
        return

    def _set_sat_img_counter(self, datetime_utc):
        if (datetime_utc - self.get_current_timestamp()).days > 4:
            while True:
                self.counter += 1
                if np.abs((datetime_utc - self.get_current_timestamp()).days)<=4:
                    break
                if self.counter >= (len(self.timestamps) - 1):
                    print('No data after {}'.format(datetime_utc))
                    self.counter = len(self.timestamps) -1 
                    return None
        elif (self.get_current_timestamp() - datetime_utc).days > 4:
            while True:
                self.counter -= 1
                if np.abs((datetime_utc - self.get_current_timestamp()).days)<4:
                    break
                if self.counter <= 0:
                    print('No data before {}'.format(datetime_utc))
                    self.counter = 0
                    return None  
        return

    def get_neural_network_variables(self, datetime_utc, lat=None, lon=None,
                                     era_variables=['ssrd', 't2m'], regridder_weights=None,
                                     sat_img_keys=None):

        '''
            Get the variables for an neural network based vegetation model

                Parameters:
                    datetime_utc (datetime): The time of interest
                    lat (float): A latitude (optional)
                    lon (float): A longitude (optional)
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
        sat_inds = np.concatenate([np.arange(self.counter-8, self.counter-2, 3),
                                   np.arange(self.counter-2, self.counter+1)])
        for_ret_dict = self.get_sat_img_values_for_all_keys(counter_range=sat_inds,
                                                            lon=lon, lat=lat)
        for i, key in enumerate(sat_img_keys):
            ret_dict[key] =  for_ret_dict[i]
        for key in era_variables:
            if lat is None:
                ret_dict[key] = self.era5_inst.get_data(key=key)
            else:
                ret_dict[key] = self.era5_inst.get_data(lonlat=(lon, lat),
                                                        key=key).values.flatten()
        if lon is not None:
            land_type = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type', as_array=False).values.flatten()
        else:
            land_type = self.land_cover_type.sat_img['land_cover_type'].values
        ret_dict['land_cover_type'] = land_type
        return ret_dict           
    
    
    def get_vprm_variables(self, datetime_utc, lat=None, lon=None,
                           add_era_variables=[], regridder_weights=None):

        '''
            Get the variables for the Vegetation Photosynthesis and Respiration Model

                Parameters:
                    datetime_utc (datetime): The time of interest
                    lat (float): A latitude (optional)
                    lon (float): A longitude (optional)
                    add_era_variables (list): Additional era variables for modifications of the VPRM
                    regridder_weights (str): Path to the pre-computed weights for the ERA5 regridder
                Returns:
                        None
        '''
        
        if self.t2m is None:
                self.init_temps()
        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year
        self._set_sat_img_counter(datetime_utc)
        
        era_keys = ['ssrd', 't2m']
        era_keys.extend(add_era_variables)
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
        t0 = time.time()
        self.load_weather_data(hour, day, month,
                               year, era_keys=era_keys)
        t1 = time.time()
        print('Time for loading ERA data : {}'.format(t1-t0))
        if (lat is None) & (lon is None):
                t0 = time.time()
                self.era5_inst.regrid(dataset=self.prototype_lat_lon,
                                      weights=regridder_weights,
                                      n_cpus=self.n_cpus)
                t1 = time.time()
                print('Time for regridding {}'.format(t1-t0))
        ret_dict = dict()
        t0=time.time()
        ret_dict['evi'] = self.get_evi(lon, lat)
        ret_dict['Ps'] = self.get_p_scale(lon, lat)
        ret_dict['par'] = self.get_par(lon, lat)
        Ts_all = self.get_t_scale(lon, lat)
        ret_dict['Ts'] = Ts_all[1]
        ret_dict['Ws'] = self.get_w_scale(lon, lat)
        ret_dict['t'] = Ts_all[0]
        t1=time.time()
        print('Time to get vprm parameters {}'.format(t1-t0))
        if add_era_variables!=[]:
            for i in add_era_variables:
                ret_dict[i] = self.era5_inst.get_data(lonlat=(lon, lat), key=i).values.flatten()
        return ret_dict       
    
    def make_vprm_predictions(self, date, res_dict=None, which_flux='NEE',
                              regridder_weights=None):
        '''
            Using the VPRM fit parameters make predictions on the entire satellite image.

                Parameters:
                    date (datetime object): The date for the prediction
                    res_dict (dict) : Dict with fit parameters ('lamb', 'par0', 'alpha', 'beta') 
                                      for the different vegetation types.
                    which_flux (str): Either 'NEE' or 'GPP'
                    regridder_weights (str): Path to the weights file for regridding from ERA5 
                                             to the satellite grid
                Returns:
                        None
        '''
        
        if self.res_dict == None:
            if res_dict is None:
                print('Need to provide a dictionary with the fit parameters')
                return 
            else:
                res_dict[0] = {'lamb': 0, 'par0': 0,
                               'alpha': 0, 'beta': 0}
                res_dict[5] = {'lamb': 0, 'par0': 0,
                               'alpha': 0, 'beta': 0}
                res_dict[8] = {'lamb': 0, 'par0': 0,
                               'alpha': 0, 'beta': 0}
                self.res_dict = res_dict
                self.res = copy.deepcopy(self.land_cover_type) 
                keys = list(self.land_cover_type.sat_img.keys())
                self.res.sat_img = self.res.sat_img.drop(keys) 
                t = self.land_cover_type.sat_img['land_cover_type'].values
                t_shape = np.shape(t)
                lamb = copy.deepcopy(t)
                par0 = copy.deepcopy(t)
                alpha = copy.deepcopy(t)
                beta = copy.deepcopy(t)
                for i in range(t_shape[0]):
                    for j in range(t_shape[1]):
                        if np.isnan(t[i,j]):
                            lamb[i,j] = np.nan
                            par0[i,j] =  np.nan
                            alpha[i,j] =  np.nan
                            beta[i,j] =  np.nan
                        else:
                            lamb[i,j] = res_dict[int(t[i,j])]['lamb']
                            par0[i,j] = res_dict[int(t[i,j])]['par0']
                            alpha[i,j] = res_dict[int(t[i,j])]['alpha']
                            beta[i,j] = res_dict[int(t[i,j])]['beta']
                self.res.sat_img = self.res.sat_img.assign({'lamb': (['y','x'], lamb)}) 
                self.res.sat_img = self.res.sat_img.assign({'par0': (['y','x'], par0)}) 
                self.res.sat_img = self.res.sat_img.assign({'alpha': (['y','x'], alpha)}) 
                self.res.sat_img = self.res.sat_img.assign({'beta': (['y','x'], beta)})

        if not os.path.exists(os.path.dirname(regridder_weights)):
            os.makedirs(os.path.dirname(regridder_weights))

        inputs = self.get_vprm_variables(date, regridder_weights=regridder_weights)
        
        if inputs is None:
            return None
        
        ret_res = dict()
        gpp = (self.res.sat_img['lamb'].values * inputs['Ps'] * inputs['Ws'] * inputs['Ts']) * inputs['evi'] * inputs['par'] / (1 + inputs['par']/self.res.sat_img['par0'].values)
        ret_res['gpp'] = gpp
        ret_res['nee'] = -gpp + self.res.sat_img['alpha'].values * inputs['t'] + self.res.sat_img['beta'].values
        return ret_res
        

    def save(self, base_path, lswi_name=None, evi_name=None):
        '''
            Save the LSWI and EVI satellite image. ToDo
        '''
        return
