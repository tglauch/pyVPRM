#from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import time
from shapely.geometry import Point, Polygon, box
import rasterio as rio
import rioxarray as rxr
from rasterio.plot import plotting_extent
from fancy_plot import *
import matplotlib
import earthpy.plot as ep
import zipfile
import os
import glob
from pyproj import Transformer
import geopandas as gpd
from pymodis import downmodis
import math 
from pyproj import Proj
import shutil 
from affine import Affine
from rioxarray.rioxarray import affine_to_coords
import requests
from requests.auth import HTTPDigestAuth
from rioxarray import merge
import yaml
import warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import LinearSegmentedColormap
from lxml import etree
from datetime import datetime
from rasterio.warp import calculate_default_transform
import h5py

def geodesic_point_buffer(lat, lon, km):
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    t = Transformer.from_crs('+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'.format(lat=lat, lon=lon),
                             '+proj=longlat +datum=WGS84')
    ret = [t.transform(i[0], i[1]) for i in buf.exterior.coords[:]]
    return ret


def make_cmap(vmin, vmax, nbins, cmap_name='Reds'):
    if isinstance(cmap_name, list):
        cmap = LinearSegmentedColormap.from_list('some_name', cmap_name, N=nbins)
    else:
        cmap = mpl.cm.get_cmap(cmap_name)  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]

        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
#    cmap.set_under('blue') #'#d4ebf2')

    # define the bins and normalize
    bounds = np.linspace(vmin, vmax, nbins)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


with open("/home/b/b309233/software/VPRM_preprocessor/logins.yaml", "r") as stream:
    try:
        logins = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
        
class satellite_data_manager:
    # Base class for all satellite images
    
    def __init__(self, datapath=None, sat_image_path=None, sat_img=None):
        self.outpath = datapath
        self.sat_image_path = sat_image_path  
        self.sat_img = sat_img
        self.t = None
        return
    
    def value_at_lonlat(self, lon, lat, as_array=True, key=None, isel={}):
        # get the value of the satellite image at a 
        #certain lon and lat
        
        if self.t is None:
            self.t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                           self.sat_img.rio.crs)
        x_a, y_a = self.t.transform(lon, lat)
        #ret = self.sat_img.sel(x=x_a, y=y_a, method="nearest")
        if key is not None:
            if isinstance(x_a, list):
                ret = self.sat_img[key].isel(isel).interp(x=('z', x_a), y=('z', y_a), method='linear')
            else:
                ret = self.sat_img[key].isel(isel).interp(x=x_a, y=y_a,
                                          method="linear")
        else:
            if isinstance(x_a, list) | isinstance(x_a, np.ndarray) :
                ret = self.sat_img.isel(isel).interp(x=('z', x_a), y=('z', y_a), method='linear')
            else:
                ret = self.sat_img.isel(isel).interp(x=x_a, y=y_a,
                                          method="linear")
        if as_array:
            return ret.to_array().values
        else:
            return ret
    
    def load(self, proj=None, **kwargs):
        # loading using the individual loading functions 
        # of the derived classes

        self.individual_loading(**kwargs)
        if proj is not None:
            self.reproject(proj=proj)     
        self.ext = self.get_plotting_extend()
        self.t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                self.sat_img.rio.crs)
        return
        
    def get_band_names(self):
        return self.keys
        
    def reproject(self, proj):
        # reproject using a pre-defined projection or by passing a
        # projection string
        
        if proj == 'WGS84':
            self.sat_img = self.sat_img.rio.reproject('+proj=longlat +datum=WGS84')#
        elif proj == 'CRS':
            self.sat_img = self.sat_img.rio.reproject(self.sat_img.rio.estimate_utm_crs())   
        else:
            self.sat_img = self.sat_img.rio.reproject(proj)
        try:
            self.proj_dict = self.sat_img.rio.crs.to_dict()
        except Exception as e:
            self.proj_dict = {}
            print(e)
        self.get_plotting_extend()
        self.t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                self.sat_img.rio.crs)
        return
        
    def get_plotting_extend(self):
        # get the corners of the satellite image
        
        return plotting_extent(self.sat_img[list(self.sat_img.data_vars.keys())[0]].values,
                               self.sat_img.rio.transform())
    
    def plot_bands(self, cmaps='Reds', titles=None, save=None):
        # plot the bands of the satellite image
        
        if self.sat_img is None:
            print('No image loaded...yet')
            return

        sh = len(self.keys)
        if titles==None:
            titles = ['B_{}'.format(i) for i in range(sh)] #['Blue', 'Green', 'Red']
        sqrt_sh = int(np.ceil(np.sqrt(sh)))
        fig, axs = plt.subplots(sqrt_sh, sqrt_sh,
                                figsize=(12,9))
        if np.ndim(axs) == 0 :
            axs = np.array([[axs]])
        # print(axs)
        for i in range(len(axs)):
            for j in range(len(axs[i])):
                c = sqrt_sh * i + j
                if c>=len(titles):
                    axs[i,j].remove()
                    continue
                plt_data = self.sat_img[self.keys[c]].values
                vmax = np.percentile(plt_data[np.isfinite(plt_data)], 90)
                if isinstance(cmaps, list):
                    cmap=cmaps[c]
                else:
                    cmap=cmaps
                ep.plot_bands(plt_data, vmax=vmax, # title=titles[c]
                              vmin=0, cmap=cmap, figsize = (7,9), 
                              ax=axs[i,j], extent=self.ext)
                axs[i,j].set_title(titles[c])
        if save is not None:
            fig.savefig(save, dpi=300)
        fig.show()
        return
    
    def plot_rgb(self, which_bands=[2,1,0], str_clip=2,
                 figsize=0.9, save=None):
        #plot an rgb image with the bands given in which_bands
        
        fig, ax = newfig(figsize, ratio=1.0)
        plt_img = np.array([self.sat_img[c].values
                            for c in which_bands])
        ep.plot_rgb(plt_img,
                    ax=ax, stretch=True, # [0, 2, 1] , rgb=which_bands
                    extent=self.ext, str_clip=str_clip,
                    title="Plot of the  data  clipped  to the geometry")
        if save is not None:
            fig.savefig(save, dpi=500)
        fig.show()
  
    def plot_ndvi(self, band1, band2, figsize=0.9,
                  save=None, n_colors=9, vmin=None,
                  vmax=1.0):
        # plot the normalized difference vegetation index
        
        data = self.sat_img
        NDVI =(data[band2].values - data[band1].values)/(data[band2].values + data[band1].values)
        mask = (NDVI < -1) | (NDVI >1) | (np.isnan(NDVI))
        NDVI[mask] = -1
        NDVI_map, NDVI_norm = make_cmap(-0.2, 1, n_colors,
                                        ['white', 'green'])
        NDVI_map.set_under('white') 
        fig, ax = newfig(figsize, ratio=1.0)
        if vmin is None:
            vmin = 1- (1/n_colors)/2
        ep.plot_bands(NDVI, ax=ax, cmap=NDVI_map,
                      extent=self.ext, vmin=vmin, vmax=vmax)
        if save is not None:
            fig.savefig(save, dpi=500)
        fig.show()
   
    def add_tile(self, new_tiles, reproject=False):
        # merge tiles together using the projection of the current satellite image

        if not isinstance(new_tiles, list):
            new_tiles = [new_tiles]
        if not np.all([isinstance(i, satellite_data_manager) for i in new_tiles]):
            print('Can only merge with another instance of a satellite_data_manger')
        if reproject:
            print('Do reprojections')
            to_merge = [i.sat_img.rio.reproject(self.sat_img.rio.crs) for i in new_tiles]
        else:
            to_merge = [i.sat_img for i in new_tiles]
        to_merge.append(self.sat_img)
        print('Merge')
        self.sat_img = merge.merge_datasets(to_merge)
        self.ext = self.get_plotting_extend()
        return
    
    def crop(self, lonlat, radius):
        # crop the satellite images in place using a given radius around a given 
        # longitude and latitude
        
        circle_poly = gpd.GeoSeries(Polygon(geodesic_point_buffer(lonlat[1],
                                                                  lonlat[0],
                                                                  radius)),
                                    crs='WGS 84')
        if not circle_poly.crs == self.sat_img.rio.crs:
            # If the crs is not equal reproject the data
            circle_poly = circle_poly.to_crs(self.sat_img.rio.crs)

        crop_bound_box = [box(*circle_poly.total_bounds)]

        self.sat_img = self.sat_img.rio.clip([box(*circle_poly.total_bounds)],
                                             all_touched=True,
                                             from_disk=False).squeeze()
        return

    def crop_to_number_of_pixels(self, lonlat, num_pixels, key, reproject=False):
        # crop the satellite images in place to a certain number of pixels around
        # the given longitude and latitude
        
        if (self.sat_img.rio.crs != self.sat_img.rio.estimate_utm_crs()) & reproject:
            self.sat_img = self.sat_img.rio.reproject(self_sat.img.rio.estimate_utm_crs())
        t = Transformer.from_crs('+proj=longlat +datum=WGS84',
                                 self.sat_img.rio.crs)
        x_a, y_a = t.transform(lonlat[0], lonlat[1])
        x_ind = np.argmin(np.abs(self.sat_img.x.values - x_a))
        y_ind = np.argmin(np.abs(self.sat_img.y.values - y_a))
        shift = int(np.floor(num_pixels/2.))
        return self.sat_img[key].values[y_ind-shift: y_ind+shift+1, x_ind-shift : x_ind+shift+1]
        
    def individual_loading(self):
        return

    def save(self, save_path):
        # save satellite image to save_path
        
        self.sat_img.to_netcdf(save_path)
        return

    
class earthdata(satellite_data_manager):
    # Classes for everything that can be downloaded from the NASA 
    # earthdata server, especially MODIS and VIIRS

        
    def __init__(self, datapath=None, sat_image_path=None):
        super().__init__(datapath, sat_image_path)
        return

            
    def _init_downloader(self, dest, date,
                         delta, username, lonlat=None, pwd=None, token=None,
                         jpg=False, enddate=None, hv=None):
        # Downloader with interface to the Earthdata server
        
        if hv is not None:
            h = hv[0]
            v = hv[1]
        else:
            h,v = self.lat_lon_to_modis(lonlat[1], lonlat[0])
        day= '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
        tiles = 'h{:02d}v{:02d}'.format(h,v)

        downloader = downmodis.downModis(destinationFolder=dest,
                                         tiles=tiles, today=day,
                                         product=self.product,
                                         path=self.path,
                                         delta=delta, user=username,
                                         enddate=enddate,
                                         password=pwd, token=token,
                                         jpg=jpg)
        
        return downloader
    
    def download(self, date, savepath, username,
                 lonlat=None, pwd = None, token=None,
                 delta = 1, jpg=False, enddate=None,
                 hv=None,rmv_downloads=False):
        
        dest = os.path.join(savepath, 'temp_for_download')
        
        modisDown = self._init_downloader(dest, date, delta, 
                                          username,lonlat,
                                          pwd, token, jpg,
                                          enddate, hv)
        
        # try:
        print('Download {} data'.format(self.sat))
        modisDown.connect()
        modisDown.downloadsAllDay()
        new_files = self.get_files(dest)
        for nf in new_files:
            test_path = os.path.join(savepath, os.path.basename(nf))
            if not os.path.exists(test_path):
                shutil.move(nf, savepath)
        self.sat_image_path = os.path.join(savepath, os.path.basename(new_files[0]))
        if rmv_downloads:
            shutil.rmtree(dest)
        print('Done...')
        # except Exception as e:
        #     print('Data could not be downloaded')
        #     print(e)
        return
    
    def _to_standard_format(self):
        return  
    
    def lat_lon_to_modis(self, lat, lon):
        CELLS = 2400
        VERTICAL_TILES = 18
        HORIZONTAL_TILES = 36
        EARTH_RADIUS = 6371007.181
        EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

        TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
        TILE_HEIGHT = TILE_WIDTH
        CELL_SIZE = TILE_WIDTH / CELLS
        MODIS_GRID = Proj(f'+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext')
        x, y = MODIS_GRID(lon, lat)
        h = (EARTH_WIDTH * .5 + x) / TILE_WIDTH
        v = -(EARTH_WIDTH * .25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
        return int(h), int(v)
    
    
class modis(earthdata):
    #Class to download and load MODIS data
    
    
    def __init__(self, datapath=None, sat_image_path=None):

        super().__init__(datapath, sat_image_path)
        self.use_keys = ['sur_refl_b01', 'sur_refl_b02',
                         'sur_refl_b03', 'sur_refl_b04',
                         'sur_refl_b05', 'sur_refl_b06',
                         'sur_refl_b07', 'sur_refl_qc_500m']
        self.load_kwargs = {'variable': self.use_keys}
        self.sat = 'MODIS'
        self.product = 'MOD09A1.006'
        self.path = "MOLT"

    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(self.sat_image_path, 
                                 masked=True, cache=False).squeeze()
        self.sat_img = self.sat_img[self.use_keys]
        rename_dict = dict()
        for i in self.use_keys:
            rename_dict[i] = i.split('_')[-1].replace('b', 'B')
        self.sat_img = self.sat_img.rename(rename_dict)
        self.keys = np.array(list(self.sat_img.data_vars))
        for key in self.keys:
            self.sat_img[key] = self.sat_img[key] * self.sat_img[key].scale_factor
        self.meta_data = self.sat_img.attrs
        return
    
    def get_files(self, dest):
        return glob.glob(os.path.join(dest, '*.hdf'))
    
    def get_cloud_coverage(self):
        print('PERCENTCLOUDY', self.meta_data['PERCENTCLOUDY'])
        
    def get_recording_time(self):
        date0 = datetime.strptime(self.meta_data['RANGEBEGINNINGDATE'] + 'T' + self.meta_data['RANGEBEGINNINGTIME'] + 'Z',
                                 '%Y-%m-%dT%H:%M:%S.%fZ')
        date1 = datetime.strptime(self.meta_data['RANGEENDINGDATE'] + 'T' + self.meta_data['RANGEENDINGTIME'] + 'Z',
                                 '%Y-%m-%dT%H:%M:%S.%fZ')
        return date0 + (date1 - date0 ) /2

        
class VIIRS(earthdata):
    #Class to download and load VIIRS data

    
    def __init__(self, datapath=None, sat_image_path=None):
        super().__init__(datapath, sat_image_path)
        self.use_keys = []
        self.load_kwargs = {'variable': self.use_keys}
        self.sat = 'VIIRS'
        self.path = "VIIRS"
        self.product = 'VNP09H1.001' #'VNP09GA.001' # 
        self.pixel_size = 463.31271652777775
    
    def get_files(self, dest):
        return glob.glob(os.path.join(dest, '*.h5'))
 
    def set_sat_img(self, ind):
        #implements ones M and L bands are used. Currently only M bands implemented. 
        return

    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(self.sat_image_path, 
                                 masked=True, cache=False)
        f = h5py.File(self.sat_image_path, "r")
        
        fileMetadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].split()  
        fileMetadata = [m.decode('utf-8') for m in fileMetadata]   
        ulc = [i for i in fileMetadata if 'UpperLeftPointMtrs' in i][0]  
        west = float(ulc.split('=(')[-1].replace(')', '').split(',')[0])
        north = float(ulc.split('=(')[-1].replace(')', '').split(',')[1])
        del f
        
        if isinstance(self.sat_img, list):
            self.sat_img = self.sat_img[0].squeeze()
            self.keys = [i for i in list(self.sat_img.data_vars) if '_M' in i]
            self.sat_img = self.sat_img[self.keys]
        else:
            self.sat_img = self.sat_img.squeeze()
            self.keys = self.sat_img.data_vars
        
        crs_str =  'PROJCS["unnamed",\
                    GEOGCS["Unknown datum based upon the custom spheroid", \
                    DATUM["Not specified (based on custom spheroid)", \
                    SPHEROID["Custom spheroid",6371007.181,0]], \
                    PRIMEM["Greenwich",0],\
                    UNIT["degree",0.0174532925199433]],\
                    PROJECTION["Sinusoidal"], \
                    PARAMETER["longitude_of_center",0], \
                    PARAMETER["false_easting",0], \
                    PARAMETER["false_northing",0], \
                    UNIT["Meter",1]]'
        
        # print(west, north)
        rename_dict = dict()
        for i in self.keys:
            rename_dict[i] = i.split('Data_Fields_')[1]
        self.sat_img = self.sat_img.rename(rename_dict)
        self.keys = np.array(list(self.sat_img.data_vars))
        for k in self.keys:
            if ('SurfReflect_I' not in k) & ('SurfReflect_M' not in k):
                continue
            sf = self.sat_img.attrs[[i for i in self.sat_img.attrs if ('scale_factor' in i) & ('err' not in i) & (k in i)][0]]
            self.sat_img[k] = self.sat_img[k] * sf

        transform = Affine(self.pixel_size, 0, west, 0, -self.pixel_size, north)
        # print(transform)
        coords = affine_to_coords(transform, self.sat_img.rio.width, self.sat_img.rio.height)
        self.sat_img.coords["x"] = coords["x"]
        self.sat_img.coords["y"] = coords["y"]
        self.sat_img.rio.write_crs(crs_str, inplace=True)
        self.meta_data = self.sat_img.attrs
        
        # self.sat_img.rio.write_crs(proj, inplace=True)
#        self.sat_img = self.sat_img.to_array()
        return    

    
    def get_cloud_coverage(self):
        print('PercentCloudy', self.meta_data['PercentCloudy'])
        
    
    def get_recording_time(self):    
        date0 = datetime.strptime(self.meta_data['RangeBeginningDate'] + 'T' + self.meta_data['RangeBeginningTime'] + 'Z',
                                 '%Y-%m-%dT%H:%M:%S.%fZ')
        date1 = datetime.strptime(self.meta_data['RangeEndingDate'] + 'T' + self.meta_data['RangeEndingTime'] + 'Z',
                                 '%Y-%m-%dT%H:%M:%S.%fZ')
        return date0 + (date1 - date0 ) /2

        


class proba_v(satellite_data_manager):
    #Class to download and load Proba V data

    def __init__(self, datapath=None, sat_image_path=None):
        super().__init__(datapath, sat_image_path) 
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        self.path = 'PROBA-V_100m'
        self.product = 'S5_TOC_100_m_C1'
        self.server = 'https://www.vito-eodata.be/PDF/datapool/Free_Data/'
        self.pixel_size = 100
# PROBA-V_100m/S5_TOC_100_m_C1/2021/05/06/PV_S5_TOC-20210506_100M_V101/PROBAV_S5_TOC_X14Y00_20210506_100M_V101.HDF5

    def download(self, date0, date1, savepath,
                 username, pwd,
                 lonlat=False, shape=False,
                 cloudcoverpercentage = (0, 100),
                 server='https://apihub.copernicus.eu/apihub',):

        session = requests.Session()
        session.auth = ('theo_g', '%FeRiEn%07')
        url = 'https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/S5_TOC_100_m_C1/2021/09/11/PV_S5_TOC-20210911_100M_V101/PROBAV_S5_TOC_X15Y04_20210911_100M_V101.HDF5'
        auth = session.post(url)
        data = session.get(url)
        with open("/home/b/b309233/temp/PROBAV_S5_TOC_X15Y04_20210911_100M_V101.HDF5", "wb") as f:
            f.write(data.content)
        return
            
    def coord_to_tile(self):
        return
            # https://proba-v.vgt.vito.be/sites/probavvgt/files/Products_User_Manual.pdft

    def individual_loading(self):
        self.sat_img = rxr.open_rasterio(self.sat_image_path, 
                                 masked=True, cache=False)   
        self.keys = np.array(list(self.sat_img.data_vars))
        handler.sat_img.rio.write_crs(handler.sat_img.attrs['MAP_PROJECTION_WKT'],
                              inplace=True)
        
    def get_recording_time(self):
        return

class copernicus_land_cover_map(satellite_data_manager):
    # Class to load the copernicus land cover map 
    # To get the data, download for example from 
    # here: https://lcviewer.vito.be/download

    def __init__(self, sat_image_path):
        super().__init__()   
        self.load_kwargs = {}
        self.sat_image_path = sat_image_path
        
    def individual_loading(self):
        try: 
            self.sat_img = rxr.open_rasterio(self.sat_image_path, 
                             masked=True, band_as_variable=True,
                                            cache=False).squeeze()
        except:
            self.sat_img = rxr.open_rasterio(self.sat_image_path, 
                             masked=True, cache=False).squeeze()
        self.keys = np.array(list(self.sat_img.data_vars))
            
            
class sentinel2(satellite_data_manager):
    # Class to download an load sentinel 2 data
    # Note: Available data on the copernicus hub is 
    # limited
    
    def __init__(self, datapath=None, sat_image_path=None):
        super().__init__(datapath, sat_image_path)
        self.api = False
        self.load_kwargs = {}
        return
        
    def individual_loading(self):
        return
        
    def download(self, date0, date1, savepath,
                 username, pwd,
                 lonlat=False, shape=False,
                 cloudcoverpercentage = (0, 100),
                 server='https://apihub.copernicus.eu/apihub',
                 processinglevel = 'Level-2A',
                 platformname = 'Sentinel-2'):
        
    # Requires an copernicus account
        
        if self.api == False:
            self.api = SentinelAPI(username,
                                   pwd,
                                   'https://apihub.copernicus.eu/apihub')
        if lonlat is not False:
            footprint = (
                    "POINT({} {})".format(lonlat[0], lonlat[1])
                )
        else:
            footprint = shape
            
        products = self.api.query(footprint,
                             date = (date0, date1), # (date(2022, 8, 10), date(2022, 10, 7)),
                             platformname = platformname,
                             processinglevel = processinglevel,
                             cloudcoverpercentage = cloudcoverpercentage)
        self.api.download(list(products.keys())[0],
                          directory_path=savepath)
        meta_data = self.api.to_geodataframe(products)
        self.outpath = self._unzip(savepath, meta_data['title'][0])
        
    def _unzip(self, folder_path, file_name):
        outpath = os.path.join(folder_path, file_name)
        with zipfile.ZipFile(os.path.join(folder_path,
                                          file_name + '.zip'), 'r') as zip_ref:
            zip_ref.extractall(outpath)
        return outpath
    
    def individual_loading(self, bands = 'all', #[1,2,3] 
             resolution = '20m'):
        ifiles =  glob.glob(os.path.join(self.outpath, '**/*_B*_{}*'.format(resolution)),
                                     recursive=True)
        if bands != 'all':
            ifiles = [i for i in ifiles
                      if i.split('_B')[1].split('_')[0] in bands]
        for j, f in enumerate(ifiles):
            band_name = 'B{}'.format(f.split('_B')[1].split('_')[0])
            if j == 0:
                t = rxr.open_rasterio(f,masked=False, parse_coordinates=True, 
                                      band_as_variable=True, cache=False).squeeze()
                t = t.rename({'band_1': band_name})
                crs = t.rio.crs
            else:
                add_data = rxr.open_rasterio(f,masked=False, cache=False,
                                                 band_as_variable=True).squeeze()['band_1']
                t[band_name] = add_data
        t = t.rio.write_crs(crs) #, inplace=True)
        self.sat_img = t
        self.sat_image_path = self.outpath + '.hdf' 
        self.keys = np.array(list(self.sat_img.data_vars)) 
        self.meta_data = dict()
        meta_data_file = etree.parse(glob.glob(os.path.join(self.outpath, '**/MTD_MSIL*.xml'))[0])
        for i in meta_data_file.iter():
            self.meta_data[i.tag] =i.text
        return

    def get_cloud_coverage(self):
        for i in ['CLOUDY_PIXEL_OVER_LAND_PERCENTAGE',
                 'MEDIUM_PROBA_CLOUDS_PERCENTAGE',
                 'HIGH_PROBA_CLOUDS_PERCENTAGE']:
            print(i, self.meta_data[i])
        return
    
    
    def get_recording_time(self):
        return datetime.strptime(self.meta_data['PRODUCT_START_TIME'],
                                 '%Y-%m-%dT%H:%M:%S.%fZ')
    



                                    
