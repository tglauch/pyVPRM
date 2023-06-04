This code is able to run a VPRM processing for different tasks:
a) get the variables needed to fit a VPRM model for a given set of flux tower data
b) Use the fit parameters in a to evaluate fluxes over time
c) Prepare the input for further usage in weather forecast systems like WRF

For a)
1. Prepare a config file (see for example config.cfg) and set your login data for https://urs.earthdata.nasa.gov/ in the logins.yaml 
2. Download the Satellite data from MODIS or VIIRS using 'download_satellite_images.py'. For example: python download_satellite_images.py --config ./config.yaml
3. Run the fitting code, for example python fit_params_draft.py --config ./config.yaml --h 18 --v 4.
4. Use the output in the analysis notebook to fit the paramteters and generate plots. See for example ./analysis_notebooks/2012_VPRM_eval.ipynb

For b) 
  
