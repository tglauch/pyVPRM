<img src="https://github.com/tglauch/pyVPRM/assets/29706254/ba2565e6-1434-4a95-8086-936462f8d05d" width=50% height=50%>

# About

`pyVPRM` is a software package to calculate the CO2 exchange flux between atmosphere and terrestrial biosphere using the Vegetation Photosynthesis and Respriation Model (VPRM). The implementation is highly flexible and can be run with different satellite products (Sentinel-2, MODIS, VIIRS,...), land cover products (Copernicus Land Cover Service, ESA 10-m World Cover Map) and meteorologies. Through its modular structure it is also easily extendable. 

Among others it can be used for 

1. Fitting the parameters of a VPRM model against Eddy-Covariance Fluxtower measurements (e.g. FLUXNET or ICOS)
2. Making CO2 flux predictions for a given region of interest
3. Generating input files to run VPRM in the Weather Research and Forecasting Model (WRF)


# How to use

To install the package simply run 

```
pip install git+https://github.com/tglauch/pyVPRM.git
```

In order to start your own project you need to at least follow theses steps: 

1. Get the necessary satellite data for your region of interest 
2. Get the land cover maps for your region of interest
3. Create a config file for your project
4. Generate your project scripts based on the functions of the vprm class in `VPRM.py`
5. Run the calculations

Remarks: 
- If there is not yet an interface for your satellite data or land cover map, implement a new subclass in `pyVPRM/sat_managers/`
- For new land cover maps you need to additionaly provide a mapping of the land cover classes to the VPRM classes in a config file which is stored in  `pyVPRM/vprm_configs`
- Open access to land cover maps: Copernicus: https://lcviewer.vito.be/2019 | ESA World Cover: https://viewer.esa-worldcover.org
- Open access to satellite data: MODIS,VIIRS: https://e4ftl01.cr.usgs.gov | Sentinel-2: https://scihub.copernicus.eu/

## Examples

In order to get started with ```pyVPRM``` there are a number of example scripts with corresponding `README` and comments available in the `./examples` folder. They come with pre-prepared input data, so you do not need to care about getting the data first. Check it out!

# Modular Structure

The pyVPRM implementation has a modular structure to allow for an easy replacement of satellite images and land cover maps, as well as the meteorologies. The file structure is as follows


```pyVPRM/sat_managers```

The ```satellite_data_manager``` class in this library is the basic data structure for all satellite image and land cover maps calcuations in pyVPRM. It provides function to reproject, transform, merge and crop satellite images. All other classes for specific satellite images or land cover maps, with the respective loading routines, are derived from this base class and implemented in the respective class files in the folder. 


```pyVPRM/meteorologies```
The classes in this folder provide the interface for the satellite data. This will usually strongly depend on the data availability. You'll likely need to make modifications here or implement your own class. All meteorology classes are derived from the base class in ```met_base_class.py```. An example to implement a new meteorology class can be found in ```era5_class_draft.py```.
