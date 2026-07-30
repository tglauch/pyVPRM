"""Prepare ISA and reference-EVI inputs for UrbanVPRM."""

import numpy as np
import scipy
import xarray as xr

from pyVPRM.VPRM import vprm_preprocessor
from pyVPRM.sat_managers.base_manager import satellite_data_manager


class urban_vprm_preprocessor(vprm_preprocessor):
    """Extend VPRM preprocessing with static UrbanVPRM ancillary inputs.

    Notes
    -----
    This subclass retains the generic VPRM satellite and land-cover workflow.
    It adds only static, urban-specific fields that depend on impervious
    surface area, fractional land cover, and the satellite grid.
    """

    def __init__(self, *args, **kwargs):
        """Initialize generic VPRM preprocessing and UrbanVPRM state.

        Parameters
        ----------
        *args
            Positional arguments accepted by :class:`vprm_preprocessor`.
        **kwargs
            Keyword arguments accepted by :class:`vprm_preprocessor`.
        """
        super().__init__(*args, **kwargs)
        self.impervious_surface_area = None
        self.urban_reference_evi = None

    def add_urban_reference_evi_lookup(
        self,
        vegetated_vprm_classes,
        isa_variable="impervious_surface_percentage",
        zero_tolerance=1e-6,
    ):
        """Build nearest-reference EVI indices for the UrbanVPRM model.

        The static lookup identifies, for each satellite-grid cell, the nearest
        vegetated cell with zero impervious surface. UrbanVPRM can use these
        static candidates directly or search them dynamically for positive
        EVI at the current satellite composite.

        Parameters
        ----------
        vegetated_vprm_classes : iterable of int
            VPRM classes eligible to provide an EVI reference. Water and other
            non-vegetated classes must not be included.
        isa_variable : str, default="impervious_surface_percentage"
            Impervious-surface percentage variable supplied on the satellite
            grid by a compatible preprocessing workflow.
        zero_tolerance : float, default=1e-6
            Absolute tolerance used to identify ISA-zero reference cells.

        Returns
        -------
        None
            A satellite-grid dataset containing nearest-reference ``y`` and
            ``x`` indices, distance in grid units, and the eligible-reference
            mask is stored in :attr:`urban_reference_evi`.

        Raises
        ------
        ValueError
            If ISA or fractional land cover has not been added, requested VPRM
            classes are unavailable, or no eligible reference cell exists.
        """
        if self.impervious_surface_area is None:
            raise ValueError("Impervious-surface data must be added first.")
        if self.land_cover_type is None:
            raise ValueError("Fractional land-cover data must be added first.")
        if isa_variable not in self.impervious_surface_area.sat_img:
            raise ValueError(
                "Impervious-surface data do not contain {!r}.".format(isa_variable)
            )

        land_cover = self.land_cover_type.sat_img
        if "vprm_classes" not in land_cover.dims:
            raise ValueError("Land-cover data must have a vprm_classes dimension.")
        vegetated_vprm_classes = np.asarray(vegetated_vprm_classes, dtype=np.int32)
        available_classes = land_cover.coords["vprm_classes"].values
        missing_classes = np.setdiff1d(vegetated_vprm_classes, available_classes)
        if missing_classes.size:
            raise ValueError(
                "Vegetated VPRM classes are unavailable: {}.".format(
                    ", ".join(str(value) for value in missing_classes)
                )
            )

        isa = self.impervious_surface_area.sat_img[isa_variable]
        if not np.array_equal(isa.x.values, land_cover.x.values) or not np.array_equal(
            isa.y.values, land_cover.y.values
        ):
            raise ValueError("ISA and land-cover data must use the same grid coordinates.")
        if isa.rio.crs != land_cover.rio.crs:
            raise ValueError("ISA and land-cover data must use the same CRS.")

        vegetated_fraction = land_cover.sel(
            vprm_classes=vegetated_vprm_classes
        ).sum("vprm_classes")
        eligible_reference = (
            np.isfinite(isa)
            & np.isclose(isa, 0.0, atol=zero_tolerance)
            & (vegetated_fraction > 0.0)
        )
        eligible_values = np.asarray(eligible_reference.values, dtype=bool)
        if not eligible_values.any():
            raise ValueError("No vegetated ISA-zero reference cells are available.")

        y_resolution = np.abs(np.diff(isa.y.values)).mean()
        x_resolution = np.abs(np.diff(isa.x.values)).mean()
        distances, nearest_indices = scipy.ndimage.distance_transform_edt(
            ~eligible_values,
            sampling=(y_resolution, x_resolution),
            return_indices=True,
        )
        annual_evi = getattr(self, "min_max_evi", None)
        if annual_evi is not None and {
            "min_evi", "max_evi"
        }.issubset(annual_evi.sat_img.data_vars):
            minimum_evi = annual_evi.sat_img["min_evi"]
            maximum_evi = annual_evi.sat_img["max_evi"]
            reference_maximum_evi = np.asarray(maximum_evi.values)[
                nearest_indices[0], nearest_indices[1]
            ]
            reference_minimum_evi = np.asarray(minimum_evi.values)[
                nearest_indices[0], nearest_indices[1]
            ]
        else:
            reference_maximum_evi = np.full(isa.shape, np.nan, dtype=np.float32)
            reference_minimum_evi = np.full(isa.shape, np.nan, dtype=np.float32)
        lookup = xr.Dataset(
            {
                "reference_y_index": (("y", "x"), nearest_indices[0].astype(np.int32)),
                "reference_x_index": (("y", "x"), nearest_indices[1].astype(np.int32)),
                "reference_distance": (("y", "x"), distances.astype(np.float32)),
                "reference_eligible": (("y", "x"), eligible_values),
                "reference_maximum_evi": (
                    ("y", "x"), reference_maximum_evi.astype(np.float32)
                ),
                "reference_minimum_evi": (
                    ("y", "x"), reference_minimum_evi.astype(np.float32)
                ),
            },
            coords={"y": isa.y, "x": isa.x},
        )
        lookup["reference_distance"].attrs.update(
            {
                "long_name": "distance to nearest vegetated zero-impervious reference cell",
                "units": "metre",
            }
        )
        lookup["reference_maximum_evi"].attrs["long_name"] = (
            "annual maximum EVI at selected UrbanVPRM reference cell"
        )
        lookup["reference_minimum_evi"].attrs["long_name"] = (
            "annual minimum EVI at selected UrbanVPRM reference cell"
        )
        lookup.attrs["vegetated_vprm_classes"] = ",".join(
            str(value) for value in vegetated_vprm_classes
        )
        lookup.attrs["reference_candidate_definition"] = (
            "zero ISA and positive fractional coverage of an eligible vegetated VPRM class"
        )
        lookup = lookup.rio.write_crs(isa.rio.crs)
        lookup = lookup.rio.write_transform(isa.rio.transform(recalc=False))
        self.urban_reference_evi = satellite_data_manager(sat_img=lookup)
