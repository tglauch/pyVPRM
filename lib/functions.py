from pyproj import Proj
import math
import pandas as pd
import pytz
from tzwhere import tzwhere
from dateutil import parser
import numpy as np 
import os

def lat_lon_to_modis(lat, lon):
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



class flux_tower_data:
    # Class to store flux tower data in unique format
    
    def __init__(self, t_start, t_stop, ssrd_key, t2m_key,
                 site_name):
        self.tstart = t_start
        self.tstop = t_stop
        self.t2m_key = t2m_key
        self.ssrd_key = ssrd_key
        self.len = None
        self.site_dict = None
        self.site_name = site_name
        return
   
    def set_land_type(self, lt):
        self.land_cover_type = lt
        return

    def get_utcs(self):
        return self.site_dict[list(self.site_dict.keys())[0]]['flux_data']['datetime_utc'].values

    def get_lonlat(self):
        return (self.lon, self.lat)

    def get_site_name(self):
        return self.site_name
    
    def get_data(self):
        return self.flux_data
    
    def get_len(self):
        return len(self.flux_data)
        
    def get_land_type(self):
        return self.land_cover_type
    
    def drop_rows_by_index(self, indices):
        self.flux_data = self.flux_data.drop(indices)
        
    def add_columns(self, add_dict):
        for i in add_dict.keys():
            self.flux_data[i] = add_dict[i]
        return
        
class fluxnet(flux_tower_data):
    
    def __init__(self, data_path,
                 ssrd_key=None, t2m_key=None, use_vars=None,
                 t_start=None, t_stop=None):
        
        site_name = data_path.split('FLX_')[1].split('_')[0]
        self.data_path = data_path
    
        super().__init__(t_start, t_stop, ssrd_key, t2m_key,
                         site_name)
        
        if use_vars is None:
            self.vars = variables = ['NEE_CUT_REF', 'NEE_VUT_REF', 'NEE_CUT_REF_QC', 'NEE_VUT_REF_QC',
                                    'GPP_NT_VUT_REF', 'GPP_NT_CUT_REF', 'GPP_DT_VUT_REF', 'GPP_DT_CUT_REF',
                                    'TIMESTAMP_START', 'TIMESTAMP_END', 'WD', 'WS', 
                                    'SW_IN_F', 'TA_F', 'USTAR', 'RECO_NT_VUT_REF', 'RECO_DT_VUT_REF',
                                     'TA_F_QC', 'SW_IN_F_QC']
        else:
            self.vars = use_vars
        
        site_info = pd.read_pickle('/home/b/b309233/software/CO2KI/VPRM/fluxnet_sites.pkl')
        self.lat = site_info.loc[site_info['SITE_ID']==site_name]['lat'].values
        self.lon = site_info.loc[site_info['SITE_ID']==site_name]['long'].values
        self.land_cover_type = site_info.loc[site_info['SITE_ID']==site_name]['IGBP'].values
        return

    def add_tower_data(self):
        idata = pd.read_csv(self.data_path, usecols=lambda x: x in self.vars)
        idata.rename({self.ssrd_key: 'ssrd', self.t2m_key: 't2m'}, inplace=True, axis=1)
        tzw = tzwhere.tzwhere()
        timezone_str = tzw.tzNameAt(self.lat, self.lon) 
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse('200001010000') # pick a date that is definitely standard time and not DST 
        datetime_u = []
        for i, row in idata.iterrows():
            datetime_u.append(parser.parse(str(int(row['TIMESTAMP_END'])))  -  timezone.utcoffset(dt))
        datetime_u = np.array(datetime_u)
        idata['datetime_utc'] = datetime_u
        if (self.tstart is not None) & (self.tstop is not None):
            mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
            flux_data = idata[mask]
        else:
            flux_data = idata
        this_len = len(flux_data)
        if this_len < 2:
            print('No data for {} in given time range'.format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print('Data only available for the following years {}'.format(years))
            return False
        else:
            self.flux_data = flux_data
            return True


class icos(flux_tower_data):
    def __init__(self, data_path, ssrd_key=None, t2m_key=None, use_vars=None, t_start=None, t_stop=None):
        
        self.data_path = data_path
        site_name = data_path.split('ICOSETC_')[1].split('_')[0]
        
        super().__init__(t_start, t_stop, ssrd_key, t2m_key,
                         site_name)

        if use_vars is None:
            self.vars = variables = ['NEE_CUT_REF', 'NEE_VUT_REF', 'NEE_CUT_REF_QC', 'NEE_VUT_REF_QC',
                                    'GPP_NT_VUT_REF', 'GPP_NT_CUT_REF', 'GPP_DT_VUT_REF', 'GPP_DT_CUT_REF',
                                    'TIMESTAMP_START', 'TIMESTAMP_END', 'WD', 'WS', 
                                    'SW_IN_F', 'TA_F', 'USTAR', 'RECO_NT_VUT_REF', 'RECO_DT_VUT_REF',
                                     'TA_F_QC', 'SW_IN_F_QC']
        else:
            self.vars = use_vars
            
        site_info = pd.read_csv(os.path.join(os.path.dirname(self.data_path),  'ICOSETC_{}_SITEINFO_L2.csv'.format(self.site_name)),
                                on_bad_lines='skip')
        self.land_cover_type = site_info.loc[site_info['VARIABLE']=='IGBP']['DATAVALUE'].values[0]
        self.lat = float(site_info.loc[site_info['VARIABLE']=='LOCATION_LAT']['DATAVALUE'].values)
        self.lon = float(site_info.loc[site_info['VARIABLE']=='LOCATION_LONG']['DATAVALUE'].values)

        return

    def add_tower_data(self):
        idata = pd.read_csv(self.data_path, usecols=lambda x: x in self.vars,
                            on_bad_lines='skip')
        idata.rename({self.ssrd_key: 'ssrd', self.t2m_key: 't2m'}, inplace=True, axis=1)
        tzw = tzwhere.tzwhere()
        timezone_str = tzw.tzNameAt(self.lat, self.lon) 
        timezone = pytz.timezone(timezone_str)
        dt = parser.parse('200001010000') # pick a date that is definitely standard time and not DST 
        datetime_u = []
        for i, row in idata.iterrows():
            datetime_u.append(parser.parse(str(int(row['TIMESTAMP_END'])))  -  timezone.utcoffset(dt))
        datetime_u = np.array(datetime_u)
        idata['datetime_utc'] = datetime_u
        if (self.tstart is not None) & (self.tstop is not None):
            mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
            flux_data = idata[mask]
        else:
            flux_data = idata
        this_len = len(flux_data)

        if this_len < 2:
            print('No data for {} in given time range'.format(self.site_name))
            years = np.unique([t.year for t in datetime_u])
            print('Data only available for the following years {}'.format(years))
            return False
        else:
            self.flux_data = flux_data
            return True  

        
        