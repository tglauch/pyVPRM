from pyproj import Proj
import math
import pandas as pd
import pytz
from tzwhere import tzwhere
from dateutil import parser
import numpy as np
import os
from timezonefinder import TimezoneFinder
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import xarray as xr
from datetime import datetime
import rioxarray


def parse_wrf_grid_file(file_path, n_chunks=1, chunk_x=1, chunk_y=1):

    t = xr.open_dataset(file_path)

    lats = np.linspace(
        0, np.shape(t["XLAT_M"].values.squeeze())[0], n_chunks + 1, dtype=int
    )

    lons = np.linspace(
        0, np.shape(t["XLONG_M"].values.squeeze())[1], n_chunks + 1, dtype=int
    )

    out_grid = xr.Dataset(
        {
            "lon": (
                ["y", "x"],
                t["XLONG_M"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y], lons[chunk_x - 1] : lons[chunk_x]
                ],
                {"units": "degrees_east"},
            ),
            "lon_b": (
                ["y_b", "x_b"],
                t["XLONG_C"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y] + 1,
                    lons[chunk_x - 1] : lons[chunk_x] + 1,
                ],
                {"units": "degrees_east"},
            ),
            "lat": (
                ["y", "x"],
                t["XLAT_M"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y], lons[chunk_x - 1] : lons[chunk_x]
                ],
                {"units": "degrees_north"},
            ),
            "lat_b": (
                ["y_b", "x_b"],
                t["XLAT_C"].values.squeeze()[
                    lats[chunk_y - 1] : lats[chunk_y] + 1,
                    lons[chunk_x - 1] : lons[chunk_x] + 1,
                ],
                {"units": "degrees_north"},
            ),
        }
    )

    out_grid = out_grid.set_coords(["lon", "lat", "lat_b", "lon_b"])
    out_grid.rio.write_crs("WGS84", inplace=True)
    return out_grid


def make_xesmf_grid(sat_img):

    if isinstance(sat_img, dict):
        src_x = sat_img["lons"]
        src_y = sat_img["lats"]
    else:
        src_x = sat_img.coords["x"].values
        src_y = sat_img.coords["y"].values
    src_x_b = add_corners_to_1d_grid(src_x)
    src_y_b = add_corners_to_1d_grid(src_y)

    X, Y = np.meshgrid(src_x, src_y)
    X_b, Y_b = np.meshgrid(src_x_b, src_y_b)

    if not isinstance(sat_img, dict):
        t = Transformer.from_crs(
            sat_img.rio.crs, "+proj=longlat +datum=WGS84", always_xy=True
        )
        X, Y = t.transform(X, Y)
        X_b, Y_b = t.transform(X_b, Y_b)

    src_grid = xr.Dataset(
        {
            "lon": (["y", "x"], X, {"units": "degrees_east"}),
            "lon_b": (["y_b", "x_b"], X_b, {"units": "degrees_east"}),
            "lat": (["y", "x"], Y, {"units": "degrees_north"}),
            "lat_b": (["y_b", "x_b"], Y_b, {"units": "degrees_north"}),
        }
    )
    src_grid = src_grid.set_coords(["lon", "lat", "lon_b", "lat_b"])
    return src_grid


def to_esmf_grid(sat_img):

    if isinstance(sat_img, dict):
        x = sat_img["lons"]
        y = sat_img["lats"]
    else:
        y = sat_img.coords["y"].values
        x = sat_img.coords["x"].values

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    ny = len(y)
    nx = len(x)
    # make 2D
    y_center = np.broadcast_to(y[:, None], (ny, nx))
    x_center = np.broadcast_to(x[None, :], (ny, nx))

    # compute corner points: must be counterclockwise
    y_corner = np.stack(
        (
            y_center - dy / 2.0,  # SW
            y_center - dy / 2.0,  # SE
            y_center + dy / 2.0,  # NE
            y_center + dy / 2.0,
        ),  # NW
        axis=2,
    )

    x_corner = np.stack(
        (
            x_center - dx / 2.0,  # SW
            x_center + dx / 2.0,  # SE
            x_center + dx / 2.0,  # NE
            x_center - dx / 2.0,
        ),  # NW
        axis=2,
    )

    if not isinstance(sat_img, dict):
        t = Transformer.from_crs(
            sat_img.rio.crs, "+proj=longlat +datum=WGS84", always_xy=True
        )
        x_center, y_center = t.transform(x_center, y_center)
        x_corner, y_corner = t.transform(x_corner, y_corner)
    grid_imask = np.ones((ny, nx), dtype=np.int32)

    # generate output dataset
    dso = xr.Dataset()
    dso["grid_dims"] = xr.DataArray(
        np.array([nx, ny], dtype=np.int32), dims=("grid_rank",)
    )
    dso.grid_dims.encoding = {"dtype": np.int32}

    dso["grid_center_lat"] = xr.DataArray(
        y_center.reshape((-1,)), dims=("grid_size"), attrs={"units": "degrees"}
    )

    dso["grid_center_lon"] = xr.DataArray(
        x_center.reshape((-1,)), dims=("grid_size"), attrs={"units": "degrees"}
    )

    dso["grid_corner_lat"] = xr.DataArray(
        y_corner.reshape((-1, 4)),
        dims=("grid_size", "grid_corners"),
        attrs={"units": "degrees"},
    )
    dso["grid_corner_lon"] = xr.DataArray(
        x_corner.reshape((-1, 4)),
        dims=("grid_size", "grid_corners"),
        attrs={"units": "degrees"},
    )
    dso["grid_imask"] = xr.DataArray(
        grid_imask.reshape((-1,)), dims=("grid_size"), attrs={"units": "unitless"}
    )
    dso.grid_imask.encoding = {"dtype": np.int32}

    # force no '_FillValue' if not specified
    for v in dso.variables:
        if "_FillValue" not in dso[v].encoding:
            dso[v].encoding["_FillValue"] = None

    dso.attrs = {
        "title": f"{ny} x {nx} (lat x lon) grid",
        "created_by": "latlon_to_scrip",
        "date_created": f"{datetime.now()}",
        "conventions": "SCRIP",
    }
    return dso


def do_lowess_smoothing(array_to_smooth, xvals=None, timestamps=None, frac=0.25, it=3):
    ### ToDo: Choose frac adaptively from the data.

    """
    Performs lowess smoothing on a 2-D-array, where the first dimension is the time.

        Parameters:
                array_to_smooth (list): The 2-D-array
        Returns:
                The lowess smoothed array
    """

    ret = []

    if array_to_smooth.ndim == 1:
        if timestamps is None:
            t_timestamp = np.arange(len(array_to_smooth))
        else:
            t_timestamp = timestamps
        mask = np.isfinite(array_to_smooth)
        if xvals is None:
            xvals = t_timestamp
        ret = [np.nan]
        counter = 0
        while counter < 10:
            ret = lowess(
                array_to_smooth[mask],
                t_timestamp[mask],
                is_sorted=True,
                frac=frac + 0.05 * counter,
                it=it,
                xvals=xvals,
                return_sorted=False,
            )
            if not np.all(np.isfinite(ret)):
                #    print('Non finite values for frac: {}. Retry.'.format(frac+0.05*counter))
                counter += 1
            else:
                break
        return ret
    else:
        if xvals is not None:
            ret_array = np.zeros((len(xvals), np.shape(array_to_smooth)[1]))
        else:
            ret_array = np.zeros(
                (len(array_to_smooth[:, 0]), np.shape(array_to_smooth)[1])
            )
        for j in range(np.shape(array_to_smooth)[1]):
            if timestamps is None:
                t_timestamp = np.arange(len(array_to_smooth[:, j]))
            else:
                if timestamps.ndim == 1:
                    t_timestamp = timestamps
                else:
                    t_timestamp = timestamps[:, j]
            mask = np.isfinite(array_to_smooth[:, j])
            if xvals is None:
                xvals = t_timestamp
            lws_res = [np.nan]
            counter = 0
            while counter < 10:
                lws_res = lowess(
                    array_to_smooth[:, j][mask],
                    t_timestamp[mask],
                    is_sorted=True,
                    frac=frac + 0.05 * counter,
                    it=it,
                    xvals=xvals,
                    return_sorted=False,
                )
                if not np.all(np.isfinite(lws_res)):
                    #  print('Non finite values for frac: {}. Retry.'.format(frac+0.05*counter))
                    counter += 1
                else:
                    break
            ret_array[:, j] = lws_res
        return ret_array.T


def lat_lon_to_modis(lat, lon):
    CELLS = 2400
    VERTICAL_TILES = 18
    HORIZONTAL_TILES = 36
    EARTH_RADIUS = 6371007.181
    EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

    TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
    TILE_HEIGHT = TILE_WIDTH
    CELL_SIZE = TILE_WIDTH / CELLS
    MODIS_GRID = Proj(f"+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext")
    x, y = MODIS_GRID(lon, lat)
    h = (EARTH_WIDTH * 0.5 + x) / TILE_WIDTH
    v = -(EARTH_WIDTH * 0.25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
    return int(h), int(v)


def add_corners_to_1d_grid(mids):
    diff = np.unique(np.diff(mids))[0] / 2
    mids = mids - diff
    mids = list(mids)
    mids.append(mids[-1] + 2 * diff)
    mids = np.array(mids)
    return mids


# ----------------------------

from pyproj import Proj
import math
import pandas as pd
import pytz
from tzwhere import tzwhere
from dateutil import parser
import numpy as np
import os
from timezonefinder import TimezoneFinder
from statsmodels.nonparametric.smoothers_lowess import lowess
from pyproj import Transformer
import xarray as xr
from datetime import datetime


# class flux_tower_data:
#     # Class to store flux tower data in unique format

#     def __init__(self, t_start, t_stop, ssrd_key, t2m_key,
#                  site_name):
#         self.tstart = t_start
#         self.tstop = t_stop
#         self.t2m_key = t2m_key
#         self.ssrd_key = ssrd_key
#         self.len = None
#         self.site_dict = None
#         self.site_name = site_name
#         return

#     def set_land_type(self, lt):
#         self.land_cover_type = lt
#         return

#     def get_utcs(self):
#         return self.site_dict[list(self.site_dict.keys())[0]]['flux_data']['datetime_utc'].values

#     def get_lonlat(self):
#         return (self.lon, self.lat)

#     def get_site_name(self):
#         return self.site_name

#     def get_data(self):
#         return self.flux_data

#     def get_len(self):
#         return len(self.flux_data)

#     def get_land_type(self):
#         return self.land_cover_type

#     def drop_rows_by_index(self, indices):
#         self.flux_data = self.flux_data.drop(indices)

#     def add_columns(self, add_dict):
#         for i in add_dict.keys():
#             self.flux_data[i] = add_dict[i]
#         return

# # class fluxnet(flux_tower_data):

# #     def __init__(self, data_path,
# #                  ssrd_key=None, t2m_key=None, use_vars=None,
# #                  t_start=None, t_stop=None):

# #         site_name = data_path.split('FLX_')[1].split('_')[0]
# #         self.data_path = data_path

# #         super().__init__(t_start, t_stop, ssrd_key, t2m_key,
# #                          site_name)

# #         if use_vars is None:
# #             self.vars = variables = ['NEE_CUT_REF', 'NEE_VUT_REF', 'NEE_CUT_REF_QC', 'NEE_VUT_REF_QC',
# #                                     'GPP_NT_VUT_REF', 'GPP_NT_CUT_REF', 'GPP_DT_VUT_REF', 'GPP_DT_CUT_REF',
# #                                     'TIMESTAMP_START', 'TIMESTAMP_END', 'WD', 'WS',
# #                                     'SW_IN_F', 'TA_F', 'USTAR', 'RECO_NT_VUT_REF', 'RECO_DT_VUT_REF',
# #                                      'TA_F_QC', 'SW_IN_F_QC']
# #         else:
# #             self.vars = use_vars

# #         site_info = pd.read_pickle('/home/b/b309233/software/CO2KI/VPRM/fluxnet_sites.pkl')
# #         self.lat = site_info.loc[site_info['SITE_ID']==site_name]['lat'].values
# #         self.lon = site_info.loc[site_info['SITE_ID']==site_name]['long'].values
# #         self.land_cover_type = site_info.loc[site_info['SITE_ID']==site_name]['IGBP'].values
# #         return

# #     def add_tower_data(self):
# #         idata = pd.read_csv(self.data_path, usecols=lambda x: x in self.vars)
# #         idata.rename({self.ssrd_key: 'ssrd', self.t2m_key: 't2m'}, inplace=True, axis=1)
# #         tf = TimezoneFinder()
# #         timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
# #         # tzw = tzwhere.tzwhere()
# #         # timezone_str = tzw.tzNameAt(self.lat, self.lon)
# #         timezone = pytz.timezone(timezone_str)
# #         dt = parser.parse('200001010000') # pick a date that is definitely standard time and not DST
# #         datetime_u = []
# #         for i, row in idata.iterrows():
# #             datetime_u.append(parser.parse(str(int(row['TIMESTAMP_END'])))  -  timezone.utcoffset(dt))
# #         datetime_u = np.array(datetime_u)
# #         idata['datetime_utc'] = datetime_u
# #         if (self.tstart is not None) & (self.tstop is not None):
# #             mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
# #             flux_data = idata[mask]
# #         else:
# #             flux_data = idata
# #         this_len = len(flux_data)
# #         if this_len < 2:
# #             print('No data for {} in given time range'.format(self.site_name))
# #             years = np.unique([t.year for t in datetime_u])
# #             print('Data only available for the following years {}'.format(years))
# #             return False
# #         else:
# #             self.flux_data = flux_data
# #             return True


# class icos(flux_tower_data):
#     def __init__(self, data_path, ssrd_key=None, t2m_key=None, use_vars=None, t_start=None, t_stop=None):

#         self.data_path = data_path
#         site_name = data_path.split('ICOSETC_')[1].split('_')[0]

#         super().__init__(t_start, t_stop, ssrd_key, t2m_key,
#                          site_name)

#         if use_vars is None:
#             self.vars = variables = ['NEE_CUT_REF', 'NEE_VUT_REF', 'NEE_CUT_REF_QC', 'NEE_VUT_REF_QC',
#                                     'GPP_NT_VUT_REF', 'GPP_NT_CUT_REF', 'GPP_DT_VUT_REF', 'GPP_DT_CUT_REF',
#                                     'TIMESTAMP_START', 'TIMESTAMP_END', 'WD', 'WS',
#                                     'SW_IN_F', 'TA_F', 'USTAR', 'RECO_NT_VUT_REF', 'RECO_DT_VUT_REF',
#                                      'TA_F_QC', 'SW_IN_F_QC']
#         else:
#             self.vars = use_vars

#         site_info = pd.read_csv(os.path.join(os.path.dirname(self.data_path),  'ICOSETC_{}_SITEINFO_L2.csv'.format(self.site_name)),
#                                 on_bad_lines='skip')
#         self.land_cover_type = site_info.loc[site_info['VARIABLE']=='IGBP']['DATAVALUE'].values[0]
#         self.lat = float(site_info.loc[site_info['VARIABLE']=='LOCATION_LAT']['DATAVALUE'].values)
#         self.lon = float(site_info.loc[site_info['VARIABLE']=='LOCATION_LONG']['DATAVALUE'].values)

#         return

#     def add_tower_data(self):
#         idata = pd.read_csv(self.data_path, usecols=lambda x: x in self.vars,
#                             on_bad_lines='skip')
#         idata.rename({self.ssrd_key: 'ssrd', self.t2m_key: 't2m'}, inplace=True, axis=1)
#         tf = TimezoneFinder()
#         timezone_str = tf.timezone_at(lng=self.lon, lat=self.lat)
#         #tzw = tzwhere.tzwhere()
#         #timezone_str = tzw.tzNameAt(self.lat, self.lon)
#         timezone = pytz.timezone(timezone_str)
#         dt = parser.parse('200001010000') # pick a date that is definitely standard time and not DST
#         datetime_u = []
#         for i, row in idata.iterrows():
#             datetime_u.append(parser.parse(str(int(row['TIMESTAMP_END'])))  -  timezone.utcoffset(dt))
#         datetime_u = np.array(datetime_u)
#         idata['datetime_utc'] = datetime_u
#         if (self.tstart is not None) & (self.tstop is not None):
#             mask = (datetime_u >= self.tstart) & (datetime_u <= self.tstop)
#             flux_data = idata[mask]
#         else:
#             flux_data = idata
#         this_len = len(flux_data)

#         if this_len < 2:
#             print('No data for {} in given time range'.format(self.site_name))
#             years = np.unique([t.year for t in datetime_u])
#             print('Data only available for the following years {}'.format(years))
#             return False
#         else:
#             self.flux_data = flux_data
#             return True
