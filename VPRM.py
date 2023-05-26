import numpy as np
import sys
import os
sys.path.append('/home/b/b309233/software/CO2KI/harvard_forest/')
sys.path.append('/home/b/b309233/software/SatManager')
from sat_manager import VIIRS, sentinel2, modis, copernicus_land_cover_map, satellite_data_manager
import earthpy.plot as ep
import warnings
import sys
import pandas as pd
warnings.filterwarnings("ignore")

from era5_class_new import ERA5
import pandas as pdt
map_function = lambda lon: (lon + 360) if (lon < 0) else lon
from datetime import datetime
from datetime import date
import yaml
import glob
import time
from datetime import timedelta
from dateutil import parser
from scipy.ndimage import uniform_filter
with open("/home/b/b309233/software/VPRM_preprocessor/logins.yaml", "r") as stream:
    try:
        logins = yaml.safe_load(stream)
    except yaml.YAMLError as ex:
        print(exc)
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import copy
from joblib import Parallel, delayed
import scipy.interpolate as si
import rioxarray as rxr
import xarray as xr
import xesmf as xe
import scipy
import uuid

def do_lowess_smoothing(array_to_smooth, vclass=None, count=None):
    if count!=None:
        print(count)
    ret = []
    for j in range(np.shape(array_to_smooth)[1]):
        if vclass!=None:
            if vclass[j] == 8: # skip arrays with land type class 8 
                continue
        array_to_smooth[:, j] = lowess(array_to_smooth[:, j], range(len(array_to_smooth[:, j])),
                     is_sorted=True, frac=0.2, it=1, return_sorted=False)
    return array_to_smooth.T

class vprm: 
    def __init__(self, land_cover_map=None, verbose=False):

        self.keys = ['t2m', 'ssrd']
        self.sat_imgs = []
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
                      regridder_save_path=None, driver='xEMSF',
                      n_cpus=60):
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
                os.system('mpirun -np {} ESMF_RegridWeightGen --source {} --destination {} --weight {} -m bilinear -r --64bit_offset  --extrap_method nearestd  --no_log'.format(n_cpus, src_temp_path, dest_temp_path, regridder_save_path))
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
        ds_t_evi = ds_t_evi.assign({'evi': (['time', 'vprm_class','lat','lon'], final_array)})
        ds_t_evi = ds_t_evi.assign_coords({"time": day_of_the_year})
        ds_t_evi = ds_t_evi.assign_coords({"vprm_class": veg_inds})
        final_array = []
        for ky in range(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.sat_imgs.sat_img.isel({'time': ky})['lswi'].where(self.land_cover_type.sat_img['land_cover_type'].values == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        ds_t_lswi = copy.deepcopy(ds_out)
        ds_t_lswi = ds_t_lswi.assign({'lswi': (['time', 'vprm_class','lat','lon'], final_array)})
        ds_t_lswi = ds_t_lswi.assign_coords({"time": day_of_the_year})
        ds_t_lswi = ds_t_lswi.assign_coords({"vprm_class": veg_inds})  
        
        ds_t_max_evi = copy.deepcopy(ds_out)
        ds_t_max_evi = ds_t_max_evi.assign({'evi_max': (['vprm_class','lat','lon'],
                                                     np.nanmax(ds_t_evi['evi'],axis = 0))})
        ds_t_max_evi = ds_t_max_evi.assign_coords({"vprm_class": veg_inds})  
        
        ds_t_min_evi = copy.deepcopy(ds_out)
        ds_t_min_evi = ds_t_min_evi.assign({'evi_min': (['vprm_class','lat','lon'],
                                                     np.nanmin(ds_t_evi['evi'],axis = 0))})
        ds_t_min_evi = ds_t_min_evi.assign_coords({"vprm_class": veg_inds})
        
        ds_t_max_lswi = copy.deepcopy(ds_out)
        ds_t_max_lswi = ds_t_max_lswi.assign({'lswi_max': (['vprm_class','lat','lon'],
                                                     np.nanmax(ds_t_lswi['lswi'],axis = 0))})
        ds_t_max_lswi = ds_t_max_lswi.assign_coords({"vprm_class": veg_inds})  
        
        ds_t_min_lswi = copy.deepcopy(ds_out)
        ds_t_min_lswi = ds_t_min_lswi.assign({'lswi_min': (['vprm_class','lat','lon'],
                                                     np.nanmin(ds_t_lswi['lswi'],axis = 0))})
        ds_t_min_lswi = ds_t_min_lswi.assign_coords({"vprm_class": veg_inds})

        ret_dict = {'lswi': ds_t_lswi, 'evi': ds_t_evi, 'veg_fraction': t.sat_img,
                    'lswi_max': ds_t_max_lswi, 'lswi_min': ds_t_min_lswi,
                    'evi_max': ds_t_max_evi, 'evi_min': ds_t_min_evi}
        return ret_dict
    
    def add_sat_img(self, handler, b_nir=None,
                    b_red=None, b_blue=None,
                    b_swir=None,
                    drop_bands=True, 
                    which_evi='evi',
                    smearing=False):
        evi_params = {'g': 2.5, 'c1': 6., 'c2': 7.5, 'l': 1}
        evi2_params = {'g': 2.5, 'l': 1 , 'c': 2.4}
        if not isinstance(handler, satellite_data_manager):
            print('Satellite image needs to be an object of the sattelite_data_manager class')
        else:  
            handler.sat_img = handler.sat_img.reindex(y=sorted(list(handler.sat_img.y)))
            handler.sat_img = handler.sat_img.reindex(x=sorted(list(handler.sat_img.x)))
            nir = handler.sat_img[b_nir] 
            red = handler.sat_img[b_red]  
            swir = handler.sat_img[b_swir] 
            if which_evi=='evi':
                blue = handler.sat_img[b_blue] 
                temp_evi = (evi_params['g'] * ( nir - red)  / (nir + evi_params['c1'] * red - evi_params['c2'] * blue + evi_params['l'])).values
            elif which_evi=='evi2':
                temp_evi = (evi2_params['g'] * ( nir - red )  / (nir +  evi2_params['c'] * red + evi2_params['l'])).values 
            else:
                print('which_evi whould be either evi or evi2')
        temp_lswi = ((nir -  swir) / (nir + swir)).values
        temp_evi[temp_evi>1] = np.nan
        drop_keys = list(handler.sat_img.keys())
        if smearing:
            temp_evi = uniform_filter(temp_evi, size=(3,3), mode='nearest')
            temp_lswi = uniform_filter(temp_lswi, size=(3,3), mode='nearest')
        handler.sat_img = handler.sat_img.assign({'evi': (['y','x'], temp_evi)})
        handler.sat_img = handler.sat_img.assign({'lswi': (['y','x'], temp_lswi)}) 
        if drop_bands:
                handler.sat_img = handler.sat_img.drop(drop_keys)
        self.sat_imgs.append(handler)  
        self.timestamps.append(handler.get_recording_time())
        return
    
    def sort_and_merge_by_timestamp(self):
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
        if isinstance(land_cover_map, str):
            print('Load pre-generated land cover map')
            self.land_cover_type = copernicus_land_cover_map(land_cover_map)
            self.land_cover_type.load()
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.reindex(y=sorted(list(self.land_cover_type.sat_img.y)))
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.reindex(x=sorted(list(self.land_cover_type.sat_img.x)))
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.rename({list(self.land_cover_type.sat_img.keys())[0]: 'land_cover_type'})
 
        else:
            print('Aggregating land cover map on vprm land cover types and projecting on sat image grids. This step may take a lot of time and memory')
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
        self.max_lswi = copy.deepcopy(self.prototype)
        self.min_max_evi = copy.deepcopy(self.prototype)
        shortcut = self.sat_imgs.sat_img
        self.max_lswi.sat_img = self.max_lswi.sat_img.assign({'max_lswi': (['y','x'], np.nanmax(shortcut['lswi'], axis=0))})
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'min_evi': (['y','x'], np.nanmin(shortcut['evi'], axis=0))})
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'max_evi': (['y','x'], np.nanmax(shortcut['evi'], axis=0))})   
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'th': (['y','x'], np.nanmin(shortcut['evi'], axis=0) + (0.55 * (np.nanmax(shortcut['evi'], axis=0) - np.nanmin(shortcut['evi'], axis=0))))})  
        return
    
    def lowess(self, n_cpus=1):
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({'evi': (['time', 'y', 'x'], np.array(Parallel(n_jobs=n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img['evi'][:,:,i].values) for i, x_coord in enumerate(self.xs))).T)})
        self.sat_imgs.sat_img = self.sat_imgs.sat_img.assign({'lswi': (['time', 'y', 'x'], np.array(Parallel(n_jobs=n_cpus, max_nbytes=None)(delayed(do_lowess_smoothing)(self.sat_imgs.sat_img['lswi'][:,:,i].values) for i, x_coord in enumerate(self.xs))).T)})
        return

    def get_vprm_land_class(self, lon=None, lat=None):
        if lon is not None:
            veg_class = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type', as_array=False).values.flatten()
        else:
            veg_class = self.land_cover_type.sat_img['land_cover_type'].values
        return veg_class

    def load_weather_data(self, hour, day, month, year):
        if self.era5_inst is None:
            if self.verbose:
                print('Init ERA5 Instance')
            self.era5_inst = ERA5(year, month, day, hour,  keys=['ssrd', 't2m']) 
        self.era5_inst.change_date(hour=hour, day=day,
                                   month=month, year=year)
        self.hour = hour
        self.day = day
        self.year = year
        self.month = month
        self.date = '{}-{}-{} {}:00:00'.format(year, month, day, hour)
        return
    
    def get_t_scale(self, lon=None, lat=None):
        if lon is not None:
            tmin = self.t2m.value_at_lonlat(lon, lat, key='tmin', as_array=False).values.flatten()
            topt = self.t2m.value_at_lonlat(lon, lat, key='topt', as_array=False).values.flatten()
            tmax = self.t2m.value_at_lonlat(lon, lat, key='tmax', as_array=False).values.flatten()
            t = self.era5_inst.get_data(lonlat=(lon, lat), key='t2m').values.flatten() - 273.15 # to grad celsius
            # ret= ((t - tmin) * (t - tmax)) / ((t-tmin)*(t-tmax) - (t - topt)**2)
            # ret[t<0] = 0  
            # t[t<0] = tmin[t<0]
            # return (t, ret)
        else:
            tmin = self.t2m.sat_img['tmin'].values
            topt = self.t2m.sat_img['topt'].values
            tmax = self.t2m.sat_img['tmax'].values
            t = self.era5_inst.get_data(key='t2m').values  - 273.15
        ret = ((t - tmin) * (t - tmax)) / ((t-tmin)*(t-tmax) - (t - topt)**2)
        ret[t<0] = 0
        t[t<0] = tmin[t<0]
        return (t, ret)
        
    def get_p_scale(self, lon=None, lat=None):
        if lon is not None:
            land_type = self.land_cover_type.value_at_lonlat(lon, lat, key='land_cover_type', as_array=False).values.flatten()
            th = self.min_max_evi.value_at_lonlat(lon, lat, key='th', as_array=False).values.flatten()
            evi = self.get_evi(lon, lat)
            lswi = ( 1 + self.get_lswi(lon, lat) ) / 2
            # if land_type == 1: # Evergreen forest
            #      return 1
            # else:
            #     if evi > th: # (phase1) or (phase3):
            #         return 1
            #     else:
            #         lswi = self.get_lswi(lon, lat)
            #         return (1+lswi)/2
        else:
            land_type = self.land_cover_type.sat_img['land_cover_type'].values 
            lswi = (1 + self.sat_imgs.sat_img['lswi'].isel({'time': self.counter}).values)/2
            th = self.min_max_evi.sat_img['th'].values
            evi = self.get_evi()
        mask1 = evi > th
        mask2 = land_type == 1
        lswi[(mask1) | (mask2)] = 1
        return lswi
    
    def get_current_timestamp(self):
        return self.timestamps[self.counter]
    
    def get_par(self, lon=None, lat=None):
        if lon is not None:
            ret = self.era5_inst.get_data(lonlat=(lon, lat), key='ssrd').values.flatten() / 0.505 / 3600
        else:
            ret = self.era5_inst.get_data(key='ssrd').values / 0.505 / 3600
        return ret
    
    def get_w_scale(self, lon=None, lat=None):
        lswi = self.get_lswi(lon, lat)
        if lon is not None:
            max_lswi = self.max_lswi.value_at_lonlat(lon, lat, key='max_lswi', as_array=False).values.flatten()
        else:
            max_lswi = self.max_lswi.sat_img['max_lswi'].values
        return (1+lswi)/(1+max_lswi)
    
    def get_evi(self, lon=None, lat=None):
        if lon is not None:     
            return self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key='evi', isel={'time': self.counter}).values.flatten()
        else:
            return self.sat_imgs.sat_img['evi'].isel({'time': self.counter}).values
    
    def get_lswi(self, lon=None, lat=None):
        if lon is not None:
            return self.sat_imgs.value_at_lonlat(lon, lat, as_array=False, key='lswi', isel={'time': self.counter}).values.flatten()
        else:
            return self.sat_imgs.sat_img['lswi'].isel({'time': self.counter}).values
   
    def init_meteo(self):
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

    def get_gee_variables(self, datetime_utc, lat=None, lon=None):
        if self.t2m is None:
                self.init_meteo()
        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year

        if (datetime_utc - self.get_current_timestamp()).days > 4:
            while True:
                self.counter += 1
                if np.abs((datetime_utc - self.get_current_timestamp()).days)<4:
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
        self.load_weather_data(hour, day, month, year)
#         if lon is None:
#             dims = self.evis.sat_img.dims
#             tck = self.era5_interpolators['t2m'].tck
#             reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0] # in Kelvin
#             reti = reti.reshape((dims['y'], dims['x']))
#             self.t2m.sat_img = self.t2m.sat_img.assign({'t2m': (['y','x'], reti)})

#             tck = self.era5_interpolators['ssrd'].tck
#             reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0]
#             reti = reti.reshape((dims['y'], dims['x']))
#             self.ssrd.sat_img = self.ssrd.sat_img.assign({'ssrd': (['y','x'], reti)})
        evi = self.get_evi(lon, lat)
        par = self.get_par(lon, lat)
        Ts_all = self.get_t_scale(lon, lat)
        Ts = Ts_all[1]
        t = Ts_all[0]
        Ps = self.get_p_scale(lon, lat)
        Ws = self.get_w_scale(lon, lat)
        return evi, Ps, par, Ts, Ws, t
    
    def get_raw_variables(self, datetime_utc, lat=None, lon=None):
        if self.t2m is None:
                self.init_meteo()
        self.counter = 0
        hour = datetime_utc.hour
        day = datetime_utc.day
        month = datetime_utc.month
        year = datetime_utc.year
        if (datetime_utc - self.get_current_timestamp()).days > 4:
            while True:
                self.counter += 1
                if np.abs((datetime_utc - self.get_current_timestamp()).days)<=4:
                    break
                if self.counter >= (len(self.timestamps) - 1):
                    print('No more data after {}'.format(datetime_utc))
                    self.counter = len(self.timestamps) -1 
                    return None
        elif (self.get_current_timestamp() - datetime_utc).days > 4:
            while True:
                self.counter -= 1
                if np.abs((datetime_utc - self.get_current_timestamp()).days)<=4:
                    break
                if self.counter <= 0:
                    print('No more data before {}'.format(datetime_utc))
                    self.counter = 0
                    return None
        self.load_weather_data(hour, day, month, year)
#         if lon is None:
#             dims = self.evis.sat_img.dims
#             tck = self.era5_interpolators['t2m'].tck
#             reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0] # in Kelvin
#             reti = reti.reshape((dims['y'], dims['x']))
#             self.t2m.sat_img = self.t2m.sat_img.assign({'t2m': (['y','x'], reti)})

#             tck = self.era5_interpolators['ssrd'].tck
#             reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0]
#             reti = reti.reshape((dims['y'], dims['x']))
#             self.ssrd.sat_img = self.ssrd.sat_img.assign({'ssrd': (['y','x'], reti)})
        evi = self.get_evi(lon, lat)
        lswi = self.get_lswi(lon, lat)
        par = self.get_par(lon, lat)
        Ts_all = self.get_t_scale(lon, lat)
        Ts = Ts_all[1]
        t = Ts_all[0]
        if lon is not None:
            t = float(t)
        Ps = self.get_p_scale(lon, lat)
        Ws = self.get_w_scale(lon, lat)
        return evi, lswi, par, Ts, t
    
    def get_gee(self, lat, lon, hour, day, month, year, lamb=None, par0=None):
        if lamb != None:
            self.lamb = lamb
        if par0 != None:
            self.par0 = par0
        evi, par, Ts, Ps, Ws = self.get_gee_variables(lat, lon, hour, day, month, year)
        ret = (self.lamb * Ts * Ws * Ps) * evi * 1 / (1 + par/self.par0) * par # par0, lamb,... from fitting to flux tower data
        return 
    
    def make_predictions(self, date, res_dict=None, which_flux='NEE'):
        if self.res_dict == None:
            if res_dict is None:
                print('Need to provide a dictionary with the fit parameters')
                return 
            else:
                res_dict[0] = [0, 0, 0, 0]
                res_dict[8] = [0, 0, 0, 0]
                res_dict[5] = [0, 0, 0, 0]
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
                            lamb[i,j] = res_dict[int(t[i,j])][0]
                            par0[i,j] = res_dict[int(t[i,j])][1]
                            alpha[i,j] = res_dict[int(t[i,j])][2]
                            beta[i,j] = res_dict[int(t[i,j])][3]
                self.res.sat_img = self.res.sat_img.assign({'lamb': (['y','x'], lamb)}) 
                self.res.sat_img = self.res.sat_img.assign({'par0': (['y','x'], par0)}) 
                self.res.sat_img = self.res.sat_img.assign({'alpha': (['y','x'], alpha)}) 
                self.res.sat_img = self.res.sat_img.assign({'beta': (['y','x'], beta)})       

        inputs = self.get_gee_variables(date)
        if inputs is None:
            return None
        if which_flux == 'GPP':
            ret_res = (self.res.sat_img['lamb'].values * inputs[1] * inputs[4] * inputs[3]) * inputs[0] * inputs[2] / (1 + inputs[2]/self.res.sat_img['par0'].values)
        else:
            ret_res = -(self.res.sat_img['lamb'].values * inputs[1] * inputs[4] * inputs[3]) * inputs[0] * inputs[2] / (1 + inputs[2]/self.res.sat_img['par0'].values) + self.res.sat_img['alpha'].values * inputs[5] + self.res.sat_img['beta'].values
        return ret_res
        

    def save(self, base_path, lswi_name=None, evi_name=None):
        if lswi_name is not None:
            self.lswis.save(os.path.join(base_path, lswi_name))
        if evi_name is not None:
            self.evis.save(os.path.join(base_path, evi_name))
        return
