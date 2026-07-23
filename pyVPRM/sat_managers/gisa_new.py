"""Load GISA-new time-of-first-imperviousness rasters."""

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import rioxarray as rxr
import xarray as xr
from rioxarray import merge
from rioxarray.exceptions import NoDataInBounds

from pyVPRM.sat_managers.base_manager import satellite_data_manager


class gisa_new(satellite_data_manager):
    """Manage GISA-new 30 m impervious-surface data.

    Parameters
    ----------
    sat_image_path : str or pathlib.Path or iterable of str or pathlib.Path
        A GISA-new GeoTIFF, a directory containing GISA-new GeoTIFFs, or an
        iterable of GeoTIFF paths.
    target_year : int, default=2021
        Year for the derived binary impervious-surface field. Pixels detected
        as impervious on or before this year are assigned 100 percent.

    Notes
    -----
    GISA-new pixel values record the first year in which a pixel was detected
    as impervious: 1 corresponds to 1985 and 37 corresponds to 2021. A value
    of 0 represents no detected impervious surface.
    """

    first_year = 1985
    last_year = 2021

    def __init__(
        self,
        sat_image_path: Union[str, Path, Iterable[Union[str, Path]]],
        target_year=2021,
    ):
        """Initialise a GISA-new data manager.

        Parameters
        ----------
        sat_image_path : str or pathlib.Path or iterable of str or pathlib.Path
            GISA-new raster path or paths.
        target_year : int, default=2021
            Year used to derive binary impervious-surface percentages.

        Raises
        ------
        ValueError
            If ``target_year`` lies outside the GISA-new temporal coverage.
        """
        if not self.first_year <= target_year <= self.last_year:
            raise ValueError(
                "target_year must be between {} and {}.".format(
                    self.first_year, self.last_year
                )
            )
        super().__init__()
        self.sat_image_path = sat_image_path
        self.target_year = target_year
        self.resolution = 30.0
        self._tiles = []

    def get_resolution(self):
        """Return the nominal GISA-new spatial resolution in metres.

        Returns
        -------
        float
            Nominal raster resolution, 30 metres.
        """
        return self.resolution

    def _resolve_paths(self):
        """Resolve configured GISA-new paths to a sorted non-empty file list.

        Returns
        -------
        list[pathlib.Path]
            GeoTIFF paths to load.

        Raises
        ------
        FileNotFoundError
            If a configured path does not exist or a directory has no GeoTIFFs.
        """
        if isinstance(self.sat_image_path, (str, Path)):
            configured_paths = [Path(self.sat_image_path)]
        else:
            configured_paths = [Path(path) for path in self.sat_image_path]

        paths = []
        for path in configured_paths:
            if path.is_dir():
                paths.extend(sorted(path.glob("*.tif")))
                paths.extend(sorted(path.glob("*.tiff")))
            elif path.is_file():
                paths.append(path)
            else:
                raise FileNotFoundError(f"GISA-new path does not exist: {path}")

        if not paths:
            raise FileNotFoundError("No GISA-new GeoTIFF files were found.")
        return paths

    @staticmethod
    def _unwrap_longitudes(image):
        """Move negative longitudes eastward for antimeridian mosaics.

        Parameters
        ----------
        image : xarray.Dataset
            Geographic GISA-new tile to prepare for mosaicking.

        Returns
        -------
        xarray.Dataset
            Tile with negative longitudes shifted into a 0--360-degree domain.
        """
        x_values = image.x.values
        if np.nanmin(x_values) >= 0.0:
            return image

        shifted_x = np.where(x_values < 0.0, x_values + 360.0, x_values)
        image = image.assign_coords(x=shifted_x)
        return image.rio.write_transform(image.rio.transform(recalc=True))

    def _open_image(self, path):
        """Open one GISA-new tile and derive its ISA variables.

        Parameters
        ----------
        path : pathlib.Path
            GeoTIFF file to open.

        Returns
        -------
        xarray.Dataset
            Lazy spatial data with first-detection year and binary percent
            impervious-surface variables.
        """
        image = rxr.open_rasterio(
            path,
            band_as_variable=True,
            masked=True,
            mask_and_scale=False,
            chunks={"x": 2048, "y": 2048},
            cache=False,
        ).squeeze(drop=True)
        variable_name = next(iter(image.data_vars))
        detection_index = image[variable_name]
        first_detection_year = xr.where(
            detection_index > 0, detection_index + self.first_year - 1, np.nan
        )
        imperviousness = xr.where(
            (detection_index > 0)
            & (detection_index <= self.target_year - self.first_year + 1),
            100.0,
            0.0,
        )
        image = image.drop_vars(variable_name)
        image["impervious_surface_first_year"] = first_detection_year
        image["impervious_surface_first_year"].attrs.update(
            {
                "long_name": "first year of impervious-surface detection",
                "units": "year",
                "valid_range": (self.first_year, self.last_year),
            }
        )
        image["impervious_surface_percentage"] = imperviousness
        image["impervious_surface_percentage"].attrs.update(
            {
                "long_name": "GISA-new binary impervious surface",
                "units": "percent",
                "target_year": self.target_year,
                "valid_range": (0, 100),
            }
        )
        return image

    def individual_loading(self):
        """Open configured GISA-new tiles without building a large mosaic.

        Returns
        -------
        None
            Lazy source tiles are assigned to :attr:`_tiles`. Call
            :meth:`crop_to_polygon` to create :attr:`sat_img` from the
            intersecting tiles.
        """
        self._tiles = [self._open_image(path) for path in self._resolve_paths()]
        self.sat_img = None
        self.keys = np.array(
            ["impervious_surface_first_year", "impervious_surface_percentage"]
        )

    def crop_to_polygon(self, polygon, from_disk=False):
        """Crop GISA-new tiles to a polygon and mosaic intersecting sources.

        Parameters
        ----------
        polygon : geopandas.GeoDataFrame or geopandas.GeoSeries
            Crop geometry with a defined coordinate reference system.
        from_disk : bool, default=False
            Retained for compatibility with the satellite-manager interface.
            GISA-new data are windowed before clipping to avoid full-raster
            reads regardless of this value.

        Returns
        -------
        None
            The cropped raster replaces :attr:`sat_img`.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called.
        rioxarray.exceptions.NoDataInBounds
            If the crop geometry does not intersect a configured tile.
        """
        if not self._tiles:
            raise RuntimeError("Call load() before crop_to_polygon().")

        cropped_images = []
        for tile in self._tiles:
            tile_polygon = polygon
            if tile_polygon.crs != tile.rio.crs:
                tile_polygon = tile_polygon.to_crs(tile.rio.crs)
            try:
                bounded_tile = tile.rio.clip_box(*tile_polygon.total_bounds)
                cropped_images.append(
                    bounded_tile.rio.clip(
                        tile_polygon.geometry,
                        all_touched=True,
                        from_disk=False,
                    ).squeeze()
                )
            except NoDataInBounds:
                continue

        if not cropped_images:
            raise NoDataInBounds("The crop polygon does not intersect any GISA-new tile.")

        all_x_values = np.concatenate([image.x.values for image in cropped_images])
        if np.nanmax(all_x_values) - np.nanmin(all_x_values) > 180.0:
            cropped_images = [
                self._unwrap_longitudes(image) for image in cropped_images
            ]

        if len(cropped_images) == 1:
            self.sat_img = cropped_images[0]
        else:
            self.sat_img = merge.merge_datasets(cropped_images)
