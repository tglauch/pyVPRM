[![DOI](https://zenodo.org/badge/626435494.svg)](https://doi.org/10.5281/zenodo.14216613)



<img src="https://github.com/tglauch/pyVPRM/assets/29706254/ba2565e6-1434-4a95-8086-936462f8d05d" width=45% height=50%>

# About


`pyVPRM` is a software package to calculate the CO2 exchange flux between atmosphere and terrestrial biosphere using the Vegetation Photosynthesis and Respriation Model (VPRM). It takes into account both, the primary productivity (GPP) as well as the respiration. The net flux between both of them is the net ecosystem exchange (NEE). The implementation is flexible and can be run with different satellite products (Sentinel-2, MODIS, VIIRS,...), land cover products (Copernicus Land Cover Service, ESA 10-m World Cover Map, MapBiomas) and climate models like the ECMWFS ERA5 Reanalysis. Through its modular structure it is also easily extendable. 

Among others it can be used for 

1. Fitting the parameters of a VPRM model against Eddy-Covariance Fluxtower measurements (e.g. FLUXNET or ICOS)
2. Making CO2 flux predictions for a given region of interest
3. Generating input files to run VPRM in the Weather Research and Forecasting Model (WRF)

In case of any questions please write an E-Mail to theo.glauch@dlr.de.

## Example Net Ecosystem Fluxes generated with pyVPRM

<figure>
<img src="https://github.com/user-attachments/assets/def61395-f90e-421b-9264-f0608161bd7a", height=300pt>
</figure> 

<figure>
<img src="https://github.com/user-attachments/assets/48c412b2-cf5f-4573-9db8-e60909893218", height=300pt>
</figure> 


# How to use

## Installation

In general we recommmend to set up a new virtual environment for the use of ```pyVPRM```, where you install all the required software. You might also want to consider the best practice about the use of conda and pip here (https://www.anaconda.com/blog/using-pip-in-a-conda-environment).

Prerequisites: ```pyVPRM``` requires an installation of the Earth System Modelling Framework (ESMF) and its python interface - ESMFpy - to use all functionalities that include regridding. On many HPCs specialized for Earth System Modelling and Climate Research ESMF is pre-installed. If you need to install it yourself you find instructions on the ESMF Github here: https://github.com/esmf-org. Installation through conda is possible as explained here here https://github.com/conda-forge/esmf-feedstock. It is recommendend to also install NETCDF4 to use all the functionalities of ESMF.

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c conda-forge dask netCDF4
conda install esmf
conda install esmpy
```

Afterwards install ```pyVPRM``` using pip via

```
pip install git+https://github.com/tglauch/pyVPRM.git
```

## Start your own project

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

In order to get started with ```pyVPRM``` there are a number of example scripts with corresponding `README` and comments available in the github respository: https://github.com/tglauch/pyVPRM_examples.git. It contains example scripts for:

- Generating WRF inputs under ``./wrf_preprocessor``
- Generating VPRM fluxes (GPP / NEE): ``./vprm_predictions``
- Fitting VPRM parameters: ``./fit_vprm_parameters``
- Downloading MODIS/VIIRS data using ``pyVPRM``: ``./sat_data_download``

To download the entire example repository run
```
git clone https://github.com/tglauch/pyVPRM_examples.git
```

The repositorty comes with pre-prepared input data, so you do not need to care about getting the data first. Check it out!

# Modular Structure

The pyVPRM implementation has a modular structure to allow for an easy replacement of satellite images and land cover maps, as well as the meteorologies. The file structure is as follows


```pyVPRM/sat_managers```

The ```satellite_data_manager``` class in this library is the basic data structure for all satellite image and land cover maps calcuations in pyVPRM. It provides function to reproject, transform, merge and crop satellite images. All other classes for specific satellite images or land cover maps, with the respective loading routines, are derived from this base class and implemented in the respective class files in the folder. 


```pyVPRM/meteorologies```

The classes in this folder provide the interface for the satellite data. This will usually strongly depend on the data availability. You'll likely need to make modifications here or implement your own class. All meteorology classes are derived from the base class in ```met_base_class.py```. An example to implement a new meteorology class can be found in ```era5_class_draft.py```.
