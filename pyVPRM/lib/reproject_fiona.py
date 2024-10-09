"""antimeridian-aware geopandas GeoDataFrame reprojection tool using fiona

This is an adaptation of this example from the geopandas documentation:
https://geopandas.org/en/stable/docs/user_guide/reproject_fiona.html#fiona-example

The main user-level function is transform_geodataframe(). Helper functions
crs_to_fiona() and base_transform() are not expected to be called by end-users.
"""

from functools import partial

import fiona
import geopandas as gpd
from fiona.transform import transform_geom
from packaging import version
from pyproj import CRS
from pyproj.enums import WktVersion
from shapely.geometry import mapping, shape


# set up Fiona transformer
def crs_to_fiona(proj_crs):
    """translate a CRS definition to a format fiona can understand

    from https://geopandas.org/en/stable/docs/user_guide/reproject_fiona.html#fiona-example

    helper function for transform_geodataframe()
    """
    proj_crs = CRS.from_user_input(proj_crs)
    if version.parse(fiona.__gdal_version__) < version.parse("3.0.0"):
        fio_crs = proj_crs.to_wkt(WktVersion.WKT1_GDAL)
    else:
        # GDAL 3+ can use WKT2
        fio_crs = proj_crs.to_wkt()
    return fio_crs


def base_transformer(geom, src_crs, dst_crs):
    """transform a geometry from from one CRS to another

    from https://geopandas.org/en/stable/docs/user_guide/reproject_fiona.html#fiona-example

    helper function for transform_geodataframe()
    """
    return shape(
        transform_geom(
            src_crs=crs_to_fiona(src_crs),
            dst_crs=crs_to_fiona(dst_crs),
            geom=mapping(geom),
            antimeridian_cutting=True,
            antimeridian_offset=100.0,
        )
    )


def transform_geodataframe(
    gdf: gpd.GeoDataFrame, src_crs: str, dst_crs: str
) -> gpd.GeoDataFrame:
    forward_transformer = partial(base_transformer, src_crs=gdf.crs, dst_crs=dst_crs)
    with fiona.Env(OGR_ENABLE_PARTIAL_REPROJECTION="YES"):
        gdf_reproj = gdf.set_geometry(
            gdf.geometry.apply(forward_transformer), crs=dst_crs
        )
    return gdf_reproj
