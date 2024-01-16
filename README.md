<img src="https://github.com/tglauch/pyVPRM/assets/29706254/ba2565e6-1434-4a95-8086-936462f8d05d" width=50% height=50%>

# About

`pyVPRM` is a software package to calculate the CO2 exchange flux between atmosphere and terrestrial biosphere using the Vegetation Photosynthesis and Respriation Model (VPRM). The implementation is highly flexible and can be run with different satellite products (Sentinel-2, MODIS, VIIRS,...), land cover products (Copernicus Land Cover Service, ESA 10-m World Cover Map) and meteorologies. Through its modular structure it is also easily extendable. 

Among others it can be used for 

1. Fitting the parameters of a VPRM model
2. Making CO2 flux predictions
3. Generating input files to run VPRM in the Weather Research and Forecasting Model


# How to use
For each calculation the following steps are necessary:
1. Get the necessary satellite data for your region of interest. (If there is not yet an interface for this type of satellite data, implement a new subclass in `./lib/sat_manager_add.py`)
2. Get the land cover maps for your region of interest. (If there is not yet an interface for this land cover map, implement a new subclass in `./lib/sat_manager_add.py` and define the mapping from the land cover classification of your map to the VPRM classes in a config file which is stored in  `./vprm_configs`)
3. Create a config file for your project


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
