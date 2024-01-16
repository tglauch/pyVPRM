![pyVPRM_logo](https://github.com/tglauch/pyVPRM/assets/29706254/ba2565e6-1434-4a95-8086-936462f8d05d)

# pyVPRM

`pyVPRM` is a software package to calculate the CO2 exchange flux between atmosphere and terrestrial biosphere using the Vegetation Photosynthesis and Respriation Model (VPRM). The implementation is highly flexible and can be run with different satellite products (Sentinel-2, MODIS, VIIRS,...), land cover products (Copernicus Land Cover Service, ESA 10-m World Cover Map) and meteorologies. Through its modular structure it is also easily extendable. 

Among others it can be used for 

1. Fitting the parameters of a VPRM model
2. Making flux predictions
3. Generating input files to run VPRM in the Weather Research and Forecasting Model


# How to use
This code is able to run a VPRM processing for different tasks:
a) get the variables needed to fit an analytical or neural-network based VPRM model for a given set of flux tower data
b) Use the fit parameters in a) to estimate fluxes over time and space
c) Prepare the input for further usage in weather forecast systems like WRF

For all applications you need to download the required land type maps from the Copernicus webpage here: https://lcviewer.vito.be/download, as well as the satellite images.

For a)
Prepare a config file (see for example config.cfg) and set your login data for `https://urs.earthdata.nasa.gov/` in the logins.yaml
Download the Satellite data from MODIS or VIIRS using 'download_satellite_images.py'. For example: `python download_satellite_images.py --config ./config.yaml`
Run the fitting code, for example `python fit_params_draft.py --config ./config.yaml --h 18 --v 4`.
Use the output in the analysis notebook to fit the paramteters and generate plots. See for example `./analysis_notebooks/2012_VPRM_eval.ipynb`

For b)
Prepare a config file (see for example config.cfg)
Run the `make_vprm_predictions.py` code with the config file as argument

For c)
Prepare a config file (see for example config_wrf_prepocessor.yaml)
Run the `vprm_preprocessor.py` code with the config file
