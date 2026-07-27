"""UrbanVPRM flux model using ISA, reference EVI, and UHI-adjusted temperature."""

import numpy as np

from pyVPRM.vprm_models.vprm_base_model import vprm_base_model


class vprm_urban_model(vprm_base_model):
    """Apply UrbanVPRM adjustments to selected VPRM land-cover classes.

    Parameters
    ----------
    vprm_pre : urban_vprm_preprocessor
        Preprocessor containing regridded impervious surface area and the
        static nearest-reference EVI lookup.
    met : object
        Meteorology handler accepted by :class:`vprm_base_model`.
    fit_params_dict : dict
        Per-VPRM-class fitted flux parameters.
    urban_vprm_classes : iterable of int, default=(10, 11)
        VPRM classes to which UrbanVPRM adjustments apply. Other classes use
        the unchanged base-VPRM temperature and respiration calculations.
    uhi_temperature_adjustment : callable, optional
        Callable accepting ``(temperature, isa_fraction, datetime_utc)`` and
        returning UHI-adjusted air temperature in degrees Celsius. It is
        required when preparing drivers for an urban class.

    Notes
    -----
    Respiration follows Hardiman et al. (2017) Supplement Eqs. 6--8. The
    caller supplies the UHI parameterization because its coefficients are
    location- and time-specific.
    """

    def __init__(
        self,
        vprm_pre=None,
        met=None,
        fit_params_dict=None,
        urban_vprm_classes=(10, 11),
        uhi_temperature_adjustment=None,
    ):
        """Initialize UrbanVPRM model state.

        Parameters
        ----------
        vprm_pre : urban_vprm_preprocessor
            Preprocessor containing UrbanVPRM ancillary fields.
        met : object
            Meteorology handler accepted by :class:`vprm_base_model`.
        fit_params_dict : dict
            Per-VPRM-class fitted flux parameters.
        urban_vprm_classes : iterable of int, default=(10, 11)
            VPRM classes using UrbanVPRM adjustments.
        uhi_temperature_adjustment : callable, optional
            UHI temperature adjustment callable.
        """
        super().__init__(vprm_pre=vprm_pre, met=met, fit_params_dict=fit_params_dict)
        self.urban_vprm_classes = frozenset(urban_vprm_classes)
        self.uhi_temperature_adjustment = uhi_temperature_adjustment

    def get_isa_fraction(self, lon=None, lat=None):
        """Return impervious-surface area as a fraction from zero to one.

        Parameters
        ----------
        lon, lat : float, optional
            Geographic query location. Omit both to return the complete
            satellite-grid ISA field.

        Returns
        -------
        float or xarray.DataArray
            Impervious-surface fraction on the requested support.

        Raises
        ------
        ValueError
            If the preprocessor lacks percent impervious-surface data or its
            values are outside the expected zero-to-100 range.
        """
        if self.vprm_pre.impervious_surface_area is None:
            raise ValueError("UrbanVPRM requires impervious-surface data.")
        manager = self.vprm_pre.impervious_surface_area
        variable_name = "impervious_surface_percentage"
        if variable_name not in manager.sat_img:
            raise ValueError(
                "Impervious-surface data do not contain {!r}.".format(variable_name)
            )
        if lon is None:
            isa_percent = manager.sat_img[variable_name]
        else:
            isa_percent = manager.value_at_lonlat(
                lon, lat, key=variable_name, as_array=False
            )
        if np.nanmin(isa_percent) < 0 or np.nanmax(isa_percent) > 100:
            raise ValueError("Impervious-surface percentages must be within 0--100.")
        return isa_percent / 100.0

    def _get_reference_evi(self):
        """Gather current and annual-minimum EVI at local reference cells.

        Returns
        -------
        tuple[xarray.DataArray, xarray.DataArray]
            Current reference EVI and annual minimum reference EVI on the
            target satellite grid.

        Raises
        ------
        ValueError
            If the UrbanVPRM preprocessor has not built its reference-EVI
            lookup.
        """
        if getattr(self.vprm_pre, "urban_reference_evi", None) is None:
            raise ValueError("UrbanVPRM requires a reference-EVI lookup.")
        lookup = self.vprm_pre.urban_reference_evi.sat_img
        y_index = lookup["reference_y_index"]
        x_index = lookup["reference_x_index"]
        current_evi = self.vprm_pre.sat_imgs.sat_img["evi"].isel(
            {self.vprm_pre.time_key: self.vprm_pre.counter}
        )
        minimum_evi = self.vprm_pre.min_max_evi.sat_img["min_evi"]
        return (
            current_evi.isel(y=y_index, x=x_index),
            minimum_evi.isel(y=y_index, x=x_index),
        )

    def _get_vprm_variables(
        self,
        land_cover_type,
        datetime_utc=None,
        lat=None,
        lon=None,
        add_era_variables=[],
        regridder_weights=None,
    ):
        """Get VPRM drivers, adding UrbanVPRM fields for urban classes.

        Parameters
        ----------
        land_cover_type : int
            VPRM class identifier to prepare.
        datetime_utc : datetime.datetime
            Driver timestamp.
        lat, lon : float, optional
            Geographic query location. Urban reference-EVI gathering currently
            requires gridded operation, so urban point-mode use is unsupported.
        add_era_variables : list, optional
            Additional ERA variables passed to the base model.
        regridder_weights : str, optional
            Precomputed meteorology regridding weights.

        Returns
        -------
        dict or None
            Base VPRM drivers, plus ISA, reference EVI, and UHI-adjusted
            temperature drivers for urban classes.
        """
        inputs = super()._get_vprm_variables(
            land_cover_type,
            datetime_utc=datetime_utc,
            lat=lat,
            lon=lon,
            add_era_variables=add_era_variables,
            regridder_weights=regridder_weights,
        )
        if inputs is None or land_cover_type not in self.urban_vprm_classes:
            return inputs
        if lon is not None:
            raise ValueError("UrbanVPRM reference EVI currently requires gridded mode.")
        if self.uhi_temperature_adjustment is None:
            raise ValueError(
                "UrbanVPRM requires a uhi_temperature_adjustment callable."
            )

        isa_fraction = self.get_isa_fraction()
        uhi_temperature = self.uhi_temperature_adjustment(
            inputs["tcorr"], isa_fraction, datetime_utc
        )
        temperature_scale = self.get_t_scale(
            lon=lon,
            lat=lat,
            land_cover_type=land_cover_type,
            temperature=uhi_temperature,
        )
        evi_ref, min_evi_ref = self._get_reference_evi()
        inputs.update(
            {
                "ISA": isa_fraction,
                "T_UHI": uhi_temperature,
                "tcorr": temperature_scale[0],
                "Ts": temperature_scale[1],
                "evi_ref": evi_ref,
                "min_evi_ref": min_evi_ref,
            }
        )
        return inputs

    def _calculate_respiration(self, land_cover_fraction, land_cover_type, inputs):
        """Calculate base or Hardiman UrbanVPRM ecosystem respiration.

        Parameters
        ----------
        land_cover_fraction : float, pandas.Series, or xarray.DataArray
            Fractional coverage of the requested class.
        land_cover_type : int
            VPRM class identifier to calculate.
        inputs : dict
            VPRM drivers. Urban classes additionally require ``ISA``,
            ``evi_ref``, and ``min_evi_ref``.

        Returns
        -------
        float, pandas.Series, or xarray.DataArray
            Ecosystem respiration for the requested class.
        """
        if land_cover_type not in self.urban_vprm_classes:
            return super()._calculate_respiration(
                land_cover_fraction, land_cover_type, inputs
            )
        required_keys = {"ISA", "evi", "evi_ref", "min_evi_ref"}
        missing_keys = required_keys.difference(inputs)
        if missing_keys:
            raise ValueError(
                "UrbanVPRM respiration requires inputs: {}.".format(
                    ", ".join(sorted(missing_keys))
                )
            )

        re_initial = super()._calculate_respiration(
            land_cover_fraction, land_cover_type, inputs
        )
        heterotrophic_respiration = (1.0 - inputs["ISA"]) * re_initial / 2.0
        reference_evi = np.maximum(inputs["evi_ref"], 1e-6)
        autotrophic_respiration = (
            (inputs["evi"] + inputs["min_evi_ref"] * inputs["ISA"])
            / reference_evi
            * re_initial
            / 2.0
        )
        return heterotrophic_respiration + autotrophic_respiration
