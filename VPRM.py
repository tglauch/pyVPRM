import numpy as np
import sys
sys.path.append('/home/b/b309233/software/CO2KI/harvard_forest/')
sys.path.append('/home/b/b309233/software/SatManager')
from sat_manager import VIIRS, sentinel2, modis, copernicus_land_cover_map, satellite_data_manager
import earthpy.plot as ep
import warnings
import sys
import pandas as pd
warnings.filterwarnings("ignore")
from era5_class import ERA5
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
with open("logins.yaml", "r") as stream:
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

def do_lowess_smoothing(array_to_smooth, vclass=None, count=None):
    if count!=None:
        print(count)
    ret = []
    for j in range(np.shape(array_to_smooth)[1]):
        if vclass!=None:
            if vclass[j] == 8:
                ret.append(array_to_smooth[:, j])
                continue
        tdata = array_to_smooth[:, j] 
        tmp = lowess(tdata, range(len(tdata)),
            is_sorted=True, frac=0.2, it=1)
        ret.append(tmp[:, 1])
    ret = np.array(ret).T
    return ret

class vprm: 
    def __init__(self, land_cover_map=None, verbose=False):

        self.keys = ['t2m', 'ssrd']
        self.sat_imgs = []
        self.era5_inst = None
        self.era5_interpolators = None  
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
                      savepath=None, regridder_save_path=None):
        src_x = self.evis.sat_img.coords['x'].values
        src_y = self.evis.sat_img.coords['y'].values
        src_grid = xr.Dataset({'x_b': (['y', 'x'], np.meshgrid(src_x, src_y)[0]),
                               'y_b': (['y', 'x'], np.meshgrid(src_x, src_y)[1])})
        t = Transformer.from_crs(self.evis.sat_img.rio.crs,
                                '+proj=longlat +datum=WGS84')
        src_lon, src_lat = t.transform(src_grid.x_b.values, src_grid.y_b.values)
        src_grid['lon'] = (['y', 'x'], src_lon)
        src_grid['lat'] = (['y', 'x'], src_lat)
        src_grid = src_grid.drop_vars(['x_b', 'y_b'])
        ds_out = xr.Dataset(
             {"lon": (["lon"], out_grid['lons'],
                      {"units": "degrees_east"}),
              "lat": (["lat"], out_grid['lats'],
                      {"units": "degrees_north"})})
        if weights_for_regridder is None:
            print('Need to generate the weights for the regridder\
                   this can be very slow and memory intensive')
            regridder = xe.Regridder(src_grid, ds_out, "bilinear")
            if regridder_save_path is not None:
                regridder.to_netcdf(regridder_save_path)
        else:
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
        kys = list(self.evis.sat_img.keys())
        final_array = []
        for c, ky in enumerate(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.evis.sat_img[ky].where(self.land_cover_type.sat_img['land_cover_type'] == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        ds_t_evi = copy.deepcopy(ds_out)
        ds_t_evi = ds_t_evi.assign({'evi': (['time', 'vprm_class','lat','lon'], final_array)})
        ds_t_evi = ds_t_evi.assign_coords({"time": day_of_the_year})
        ds_t_evi = ds_t_evi.assign_coords({"vprm_class": veg_inds})
        kys = list(self.lswis.sat_img.keys())
        final_array = []
        for c, ky in enumerate(kys):
            sub_array = []
            for v in veg_inds:
                tres = self.lswis.sat_img[ky].where(self.land_cover_type.sat_img['land_cover_type'] == v, 0) 
                sub_array.append(regridder(tres.values))
            final_array.append(sub_array)
        ds_t_lswi = copy.deepcopy(ds_out)
        ds_t_lswi = ds_t_lswi.assign({'lswi': (['time', 'vprm_class','lat','lon'], final_array)})
        ds_t_lswi = ds_t_lswi.assign_coords({"time": day_of_the_year})
        ds_t_lswi = ds_t_lswi.assign_coords({"vprm_class": veg_inds})  
        
        ds_t_max_evi = copy.deepcopy(ds_out)
        ds_t_max_evi = ds_t_max_evi.assign({'max_evi': (['vprm_class','lat','lon'],
                                                     np.nanmax(ds_t_evi['evi'],axis = 0).values)})
        ds_t_max_evi = ds_t_max_evi.assign_coords({"vprm_class": veg_inds})  
        
        ds_t_min_evi = copy.deepcopy(ds_out)
        ds_t_min_evi = ds_t_min_evi.assign({'min_evi': (['vprm_class','lat','lon'],
                                                     np.nanmin(ds_t_evi['evi'],axis = 0).values)})
        ds_t_min_evi = ds_t_min_evi.assign_coords({"vprm_class": veg_inds})
        
        ds_t_max_lswi = copy.deepcopy(ds_out)
        ds_t_max_lswi = ds_t_max_lswi.assign({'max_lswi': (['vprm_class','lat','lon'],
                                                     np.nanmax(ds_t_lswi['lswi'],axis = 0).values)})
        ds_t_max_lswi = ds_t_max_lswi.assign_coords({"vprm_class": veg_inds})  
        
        ds_t_min_lswi = copy.deepcopy(ds_out)
        ds_t_min_lswi = ds_t_min_lswi.assign({'min_lswi': (['vprm_class','lat','lon'],
                                                     np.nanmin(ds_t_lswi['lswi'],axis = 0).values)})
        ds_t_min_lswi = ds_t_min_lswi.assign_coords({"vprm_class": veg_inds})

        ret_dict = {'lswi': ds_t_lswi, 'evi': ds_t_evi, 'veg_fraction': t.sat_img,
                    'max_lswi': ds_t_max_lswi, 'min_lswi': ds_t_min_lswi,
                    'max_evi': ds_t_max_evi, 'min_evi': ds_t_min_evi}
        return ret_dict
    
    def add_sat_img(self, handler):
        if self.target_shape is None:
            self.xs = handler.sat_img.x.values
            self.ys = handler.sat_img.y.values
            self.target_shape = (len(self.ys), len(self.xs))
            X, Y = np.meshgrid(self.xs, self.ys)
            t = Transformer.from_crs(handler.sat_img.rio.crs,
                                    '+proj=longlat +datum=WGS84')
            self.x_long, self.y_lat = t.transform(X, Y) 
            self.prototype = copy.deepcopy(handler) 
            keys = list(self.prototype.sat_img.keys())
            self.prototype.sat_img = self.prototype.sat_img.drop(keys)
        if not isinstance(handler, satellite_data_manager):
            print('satelite image needs to be an object of the sattelite_data_manager class')
        else:
            handler.sat_img = handler.sat_img.rio.reproject_match(self.prototype.sat_img)
            print('Reproject Match')
            self.sat_imgs.append(handler)  
        return
    
    def sort_sat_imgs_by_timestamp(self):
        self.sat_imgs = np.array(self.sat_imgs)[np.argsort([i.get_recording_time() 
                                                            for i in self.sat_imgs])]

    def add_land_cover_map(self, land_cover_map, var_name='band_1',
                           save_path=None):
        if isinstance(land_cover_map, str):
            print('Load pre-generated land cover map')
            land_cover_map = copernicus_land_cover_map(land_cover_map)
            land_cover_map.load()
            land_cover_map.reproject(proj=self.sat_imgs[0].sat_img.rio.crs.to_proj4())
            ckeys = list(land_cover_map.sat_img.keys())
            if 'land_cover_type' not in ckeys:
                print('Rename key {} to internally used name {}'.format(ckeys[0], 'land_cover_type'))
                land_cover_map.sat_img = land_cover_map.sat_img.rename({ckeys[0]: 'land_cover_type'})
            self.land_cover_type = land_cover_map
            if np.shape(self.land_cover_type.sat_img['land_cover_type'].shape) != self.target_shape:
                print('Assume that I can select a subset of a land cover map')
                t =  land_cover_map.sat_img.sel(x=self.xs, y=self.ys, method="nearest").to_array().values[0]
                self.land_cover_type = copy.deepcopy(self.prototype)
                self.land_cover_type.sat_img = self.land_cover_type.sat_img.assign({'land_cover_type': (['y','x'], t)}) 
        else:
            print('Aggregating land cover map on vprm land cover types and projecting on \
                   sat image grids. This step may take a lot of time and memory')
            land_cover_map.reproject(proj=self.prototype.sat_img.rio.crs.to_proj4())
            for key in self.map_copernicus_to_vprm_class.keys():
                print(key)
                land_cover_map.sat_img[var_name].values[land_cover_map.sat_img[var_name].values==key] = self.map_copernicus_to_vprm_class[key]
            f_array = np.zeros(np.shape(land_cover_map.sat_img[var_name].values))
            count_array = np.zeros(np.shape(land_cover_map.sat_img[var_name].values))
            veg_inds = np.unique([self.map_copernicus_to_vprm_class[i] 
                                  for i in self.map_copernicus_to_vprm_class.keys()])
            for i in veg_inds:
                print(i)
                mask = np.array(land_cover_map.sat_img[var_name].values == i, dtype=float)
                ta  = scipy.ndimage.uniform_filter(mask, size=(5,5)) * 25
                # print(i, np.sum(mask), np.sum(np.array(ta>=count_array, dtype=int)))
                f_array[ta>count_array] = i 
                count_array[ta>count_array] = ta[ta>count_array]
            f_array[f_array==0]=np.nan
            land_cover_map.sat_img[var_name].values = f_array
            del ta
            del count_array
            del f_array
            del mask
            t =  land_cover_map.sat_img.sel(x=self.xs, y=self.ys, method="nearest").to_array().values[0]
            self.land_cover_type = copy.deepcopy(self.sat_imgs[0]) 
            keys = list(self.sat_imgs[0].sat_img.keys())
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.drop(keys)
            self.land_cover_type.sat_img = self.land_cover_type.sat_img.assign({'land_cover_type': (['y','x'], t)}) 
            if save_path is not None:
                 self.land_cover_type.sat_img.to_netcdf(save_path)
        return
    
    def calc_min_max_evi_lswi(self):
        self.max_lswi = copy.deepcopy(self.prototype)
        self.min_max_evi = copy.deepcopy(self.prototype)
        self.max_lswi.sat_img = self.max_lswi.sat_img.assign({'max_lswi': (['y','x'], np.nanmax(lswis_test, axis=0))})
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'min_evi': (['y','x'], np.nanmin(evis_test, axis=0))})
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'max_evi': (['y','x'], np.nanmax(evis_test, axis=0))})   
        self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'th': (['y','x'], np.nanmin(evis_test, axis=0) + (0.55 * (np.nanmax(evis_test, axis=0) - np.nanmin(evis_test, axis=0))))})  
        return
    
    def _add_variable(self, variable_dict, ind):
        self.sat_imgs[ind].sat_img = self.sat_imgs[ind].sat_img.assign(**variable_dict)
        return
    
    def calc_evis_lswis(self, b_nir, b_red, b_blue, b_swir,
                        which_evi='evi',
                        smearing=True, fill_value_for_inifinite = 0,
                        save_memory = False, do_lowess=True, n_cpus=1):
        evis = []
        lswis = []
        evi_params = {'g': 2.5, 'c1': 6., 'c2': 7.5, 'l': 1}
        evi2_params = {'g': 2.5, 'l': 1 , 'c': 2.4}
        for c, h in enumerate(self.sat_imgs):
            x = h.sat_img
            if self.evis is None:
                self.evis = copy.deepcopy(self.prototype) 
                self.lswis = copy.deepcopy(self.prototype)   
                # self.land_cover_type = copy.deepcopy(self.prototype)
            if which_evi=='evi':
                temp_evi = (evi_params['g'] * ( x[b_nir] - x[b_red] )  / (x[b_nir]+ (evi_params['c1'] * x[b_red] + evi_params['c2'] * x[b_blue]) + evi_params['l'])).values
            elif which_evi=='evi2':
                temp_evi = (evi2_params['g'] * ( x[b_nir] - x[b_red] )  / (x[b_nir] +  evi2_params['c'] * x[b_red] + evi2_params['l'])).values 
            else:
                print('which_evi whould be either evi or evi2')
            temp_lswi = ((x[b_nir] -  x[b_swir]) / (x[b_nir]+ x[b_swir])).values
            temp_evi[~np.isfinite(temp_evi)] = fill_value_for_inifinite
            temp_lswi[~np.isfinite(temp_lswi)] = fill_value_for_inifinite
            if smearing:
                temp_evi = uniform_filter(temp_evi, size=(3,3), mode='nearest')
                temp_lswi = uniform_filter(temp_lswi, size=(3,3), mode='nearest')
            self.timestamps.append(h.get_recording_time())
            evis.append(temp_evi)
            lswis.append(temp_lswi)
        evis = np.array(evis)
        lswis = np.array(lswis)
        t0 = time.time()
        if do_lowess:
            evis_test = np.rollaxis(np.array(Parallel(n_jobs=n_cpus)(delayed(do_lowess_smoothing)(evis[:,:,i]) for i, x_coord in enumerate(self.xs))), 2, start=1).T
            lswis_test = np.rollaxis(np.array(Parallel(n_jobs=n_cpus)(delayed(do_lowess_smoothing)(lswis[:,:,i]) for i, x_coord in enumerate(self.ys))), 2, start=1).T
        else:
            evis_test = evis
            lswis_test = lswis

        # self.max_lswi.sat_img = self.max_lswi.sat_img.assign({'max_lswi': (['y','x'], np.nanmax(lswis_test, axis=0))})
        # self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'min_evi': (['y','x'], np.nanmin(evis_test, axis=0))})
        # self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'max_evi': (['y','x'], np.nanmax(evis_test, axis=0))})   
        # self.min_max_evi.sat_img = self.min_max_evi.sat_img.assign({'th': (['y','x'], np.nanmin(evis_test, axis=0) + (0.55 * (np.nanmax(evis_test, axis=0) - np.nanmin(evis_test, axis=0))))})  

        for c, tevi in enumerate(evis_test):
            self.evis.sat_img = self.evis.sat_img.assign({'evi_{}'.format(c) : (['y','x'], tevi)})
        for c, tlswi in enumerate(lswis_test):
            self.lswis.sat_img = self.lswis.sat_img.assign({'lswi_{}'.format(c) : (['y','x'], tlswi)})
        if save_memory:
            self.sat_imgs = [self.sat_imgs[0]]

        # for i in range(np.shape(t)[0]):
        #     for j in range(np.shape(t)[1]):
        #         if np.isnan(t[i,j]):
        #             continue
        #         tmin[i,j] = self.temp_coefficients[t[i,j]][0]
        #         topt[i,j] = self.temp_coefficients[t[i,j]][1]
        #         tmax[i,j] = self.temp_coefficients[t[i,j]][2] 
        return

    def get_vprm_land_class(self, lon=None, lat=None):
        if lon is not None:
            veg_class = self.land_cover_type.value_at_lonlat(lon, lat, as_array=False)['land_cover_type'].values
        else:
            veg_class = self.land_cover_type.sat_img['land_cover_type'].values
        return veg_class

    def load_weather_data(self, hour, day, month, year):
        t2 = time.time()
        if self.era5_inst is None:
            if self.verbose:
                print('Init ERA5 Instance')
            self.era5_inst = ERA5(keys=self.keys,
                                  year=year, month=month)
            self.year = year
            self.month = month
        if (month != self.month) | (year != self.year):
            if self.verbose:
                print('Init ERA5 Instance')
            del self.era5_inst
            del self.era5_interpolators
            self.era5_inst = ERA5(keys=self.keys,
                                  year=year, month=month)
        self.hour = hour
        self.day = day
        self.year = year
        self.month = month
        self.date = '{}-{}-{} {}:00:00'.format(year, month, day, hour)
        self.era5_interpolators = self.era5_inst.get_all_interpolators(self.day, self.hour)
        if self.verbose:
            print('Time to load interpolators {}s'.format(t1 - t0))
            print('Time to load interpolators and era5 file {}s'.format(t1 - t2))
        return
    
    def get_t_scale(self, lon=None, lat=None):
        if lon is not None:
            tmin = self.t2m.value_at_lonlat(lon, lat, as_array=False)['tmin'].values
            topt = self.t2m.value_at_lonlat(lon, lat, as_array=False)['topt'].values
            tmax = self.t2m.value_at_lonlat(lon, lat, as_array=False)['tmax'].values
            t = float(self.era5_interpolators['t2m'](map_function(lon), lat)) - 273.15 # to grad celsius
            if t<tmin:
                return (tmin, 0)
            return ( t, ((t - tmin) * (t - tmax)) / ((t-tmin)*(t-tmax) - (t - topt)**2) )
        else:
            tmin = self.t2m.sat_img['tmin'].values
            topt = self.t2m.sat_img['topt'].values
            tmax = self.t2m.sat_img['tmax'].values
            t = self.t2m.sat_img['t2m'].values  - 273.15
            ret = ((t - tmin) * (t - tmax)) / ((t-tmin)*(t-tmax) - (t - topt)**2)
            ret[t<0] = 0
            t[t<0] = tmin[t<0]
            return (t, ret)
        
    def get_p_scale(self, lon=None, lat=None):
        if lon is not None:
            land_type =  self.land_cover_type.value_at_lonlat(lon, lat, as_array=False)['land_cover_type'].values
            th = self.min_max_evi.value_at_lonlat(lon, lat, as_array=False)['th'].values
            evi = self.get_evi(lon, lat)
            if land_type == 1: # Evergreen forest
                 return 1
            else:
                if evi > th: # (phase1) or (phase3):
                    return 1
                else:
                    lswi = self.get_lswi(lon, lat)
                    return (1+lswi)/2
        else:
            lswi = (1 + self.lswis.sat_img['lswi_{}'.format(self.counter)].values)/2
            mask1 = self.get_evi() > self.min_max_evi.sat_img['th'].values
            mask2 = self.land_cover_type.sat_img['land_cover_type'].values == 1
            lswi[(mask1) | (mask2)] = 1
            return lswi
    
    def get_current_timestamp(self):
        return self.timestamps[self.counter]
    
    def get_par(self, lon=None, lat=None):
        if lon is not None:
            ret =  float(self.era5_interpolators['ssrd'](map_function(lon), lat)) / 0.505 / 3600
        else:
            ret = self.ssrd.sat_img['ssrd'].values / 0.505 / 3600
        return ret
    
    def get_w_scale(self, lon=None, lat=None):
        lswi = self.get_lswi(lon, lat)
        if lon is not None:
            max_lswi = float(self.max_lswi.value_at_lonlat(lon, lat, as_array=False)['max_lswi'].values)
        else:
            max_lswi = self.max_lswi.sat_img['max_lswi'].values
        return (1+lswi)/(1+max_lswi)
    
    def get_evi(self, lon=None, lat=None):
        if lon is not None:
            return float(self.evis.value_at_lonlat(lon, lat, as_array=False)['evi_{}'.format(self.counter)].values)
        else:
            return self.evis.sat_img['evi_{}'.format(self.counter)].values
    
    def get_lswi(self, lon=None, lat=None):
        if lon is not None:
            return float(self.lswis.value_at_lonlat(lon, lat, as_array=False)['lswi_{}'.format(self.counter)].values)
        else:
            return self.lswis.sat_img['lswi_{}'.format(self.counter)].values
   
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

        self.t2m.sat_img = self.t2m.sat_img.assign({'tmin': (['y','x'], tmin)}) 
        self.t2m.sat_img = self.t2m.sat_img.assign({'topt': (['y','x'], topt)}) 
        self.t2m.sat_img = self.t2m.sat_img.assign({'tmax': (['y','x'], tmax)}) 
        return


    def get_gee_variables(self, datetime_utc, lat=None, lon=None):
        if self.ssrd is None:
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
                    print('No more data after {}'.format(datetime_utc))
                    self.counter = len(self.timestamps) -1 
                    return None
        elif (self.get_current_timestamp() - datetime_utc).days > 4:
            while True:
                self.counter -= 1
                if np.abs((datetime_utc - self.get_current_timestamp()).days)<4:
                    break
                if self.counter <= 0:
                    print('No more data before {}'.format(datetime_utc))
                    self.counter = 0
                    return None
        self.load_weather_data(hour, day, month, year)
        if lon is None:
            dims = self.evis.sat_img.dims
            tck = self.era5_interpolators['t2m'].tck
            reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0] # in Kelvin
            reti = reti.reshape((dims['y'], dims['x']))
            self.t2m.sat_img = self.t2m.sat_img.assign({'t2m': (['y','x'], reti)})

            tck = self.era5_interpolators['ssrd'].tck
            reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0]
            reti = reti.reshape((dims['y'], dims['x']))
            self.ssrd.sat_img = self.ssrd.sat_img.assign({'ssrd': (['y','x'], reti)})
        evi = self.get_evi(lon, lat)
        par = self.get_par(lon, lat)
        Ts_all = self.get_t_scale(lon, lat)
        Ts = Ts_all[1]
        t = Ts_all[0]
        if lon is not None:
            t = float(t)
        Ps = self.get_p_scale(lon, lat)
        Ws = self.get_w_scale(lon, lat)
        return evi, Ps, par, Ts, Ws, t
    
    def get_raw_variables(self, datetime_utc, lat=None, lon=None):
        if self.ssrd is None:
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
        if lon is None:
            dims = self.evis.sat_img.dims
            tck = self.era5_interpolators['t2m'].tck
            reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0] # in Kelvin
            reti = reti.reshape((dims['y'], dims['x']))
            self.t2m.sat_img = self.t2m.sat_img.assign({'t2m': (['y','x'], reti)})

            tck = self.era5_interpolators['ssrd'].tck
            reti = si.dfitpack.bispeu(tck[0], tck[1], tck[2], tck[3], tck[4], self.x_long.flatten(), self.y_lat.flatten())[0]
            reti = reti.reshape((dims['y'], dims['x']))
            self.ssrd.sat_img = self.ssrd.sat_img.assign({'ssrd': (['y','x'], reti)})
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
