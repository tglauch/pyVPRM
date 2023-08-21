# pyVPRM

`pyVPRM` is a software package to run calculations with the Vegetation Photosynthesis and Respriation Model. In addition to the 'classical' implementation it also provides 1.) an interface for different satellite data products 2.) an updated vegetation map handling and 3.) an API for using neural-network-based approaches.

ower data b) Use the fit parameters in a to evaluate fluxes over time c) Prepare the input for further usage in weather forecast systems like WRF

For all applications you need to download the required land type maps from the Copernicus webpage here: https://lcviewer.vito.be/download, as well as the satellite images.

For a)
Prepare a config file (see for example config.cfg) and set your login data for https://urs.earthdata.nasa.gov/ in the logins.yaml
Download the Satellite data from MODIS or VIIRS using 'download_satellite_images.py'. For example: python download_satellite_images.py --config ./config.yaml
Run the fitting code, for example python fit_params_draft.py --config ./config.yaml --h 18 --v 4.
Use the output in the analysis notebook to fit the paramteters and generate plots. See for example ./analysis_notebooks/2012_VPRM_eval.ipynb

For b)
Prepare a config file (see for example config.cfg)
Run the `VPRM_predictions.py` code with the config file as argument

For c)
Prepare a config file (see for example config_wrf_prepocessor.yaml)
Run the VPRM_preprocessor.py code with the config file
