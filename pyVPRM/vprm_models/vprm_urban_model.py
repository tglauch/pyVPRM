from vprm_base_model import vprm_base_model
from loguru import logger


class vprm_urban_model(vprm_base_model):
    """
    implemement urbanVPRM (Hardiman et al. 2017, Winbourne et al 2021)

    compared to the VPRM base model, urbanVPRM:
    - adjusts the temperature to account for urban heat island effect
    - adjusts respiration according to impervious surface area
    REFERENCES

    Hardiman, B. S., Wang, J. A., Hutyra, L. R., Gately, C. K., Getson, J. M., &
    Friedl, M. A. (2017). Accounting for urban biogenic fluxes in regional
    carbon budgets. Science of The Total Environment, 592, 366–372.
    https://doi.org/10.1016/j.scitotenv.2017.03.028

    Winbourne, J. B., Smith, I. A., Stoynova, H., Kohler, C., Gately, C. K.,
    Logan, B. A., et al. (2022). Quantification of urban forest and grassland
    carbon fluxes using field measurements and a satellite-based model in
    Washington DC/Baltimore area. Journal of Geophysical Research:
    Biogeosciences, 127(1), e2021JG006568. https://doi.org/10.1029/2021JG006568

    """

    def __init__(self, vprm_pre=None, met=None, fit_params_dict=None):
        super().__init__(vprm_pre, met, fit_params_dict)
        return

    def get_ISA(self):
        return self.vprm_pre.impervious_surface_area[
            "impervious_surface_area_percentage"
        ]

    def _get_vprm_variables(
        self,
        land_cover_type,
        datetime_utc=None,
        lat=None,
        lon=None,
        add_era_variables=[],
        regridder_weights=None,
    ):
        ret_dict = super()._get_vprm_variables(
            land_cover_type,
            datetime_utc,
            lat,
            lon,
            add_era_variables,
            regridder_weights,
        )

        logger.warning("T_UHI not yet implemented.  using Ts for T_UHI")
        ret_dict["T_UHI"] = ret_dict["Ts"]

        ret_dict["ISA"] = self.get_ISA()

        return ret_dict

    def get_respiration(self):
        """calculate respiration

        Calculate respiration according to Hardiman et al (2017) supplmemental material SI eqs. 6, 7, 8
        """

        Re_init = np.maximum(
            lcf
            * (
                self.fit_params_dict[i]["alpha"] * inputs["tcorr"]
                + self.fit_params_dict[i]["beta"]
            ),
            0,
        )
