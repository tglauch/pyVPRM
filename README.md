[![DOI](https://zenodo.org/badge/626435494.svg)](https://doi.org/10.5281/zenodo.14216613)
[![PyPI version](https://img.shields.io/pypi/v/pyVPRM.svg)](https://pypi.org/project/pyVPRM/)
[![License](https://img.shields.io/pypi/l/pyVPRM.svg)](https://pypi.org/project/pyVPRM/)
[![Downloads](https://img.shields.io/pypi/dm/pyVPRM.svg)](https://pypi.org/project/pyVPRM/)
![GitHub stars](https://img.shields.io/github/stars/tglauch/pyVPRM?style=social)
![Python Version](https://img.shields.io/pypi/pyversions/pyVPRM)


<figure>
<img src="https://github.com/tglauch/pyVPRM/assets/29706254/ba2565e6-1434-4a95-8086-936462f8d05d", height=150pt>
</figure> 
  
pyVRPM is a data-driven model to estimate the carbon flux between the atmosphere and the terrestrial biosphere using multi-spectral satellite observations. A description of the model is published in Geoscientific Model Development (GMD) https://gmd.copernicus.org/articles/18/4713/2025/. If you use this package for you scientific work please cite as follows

```
Glauch, T., Marshall, J., Gerbig, C., Botía, S., Gałkowski, M., Vardag, S. N., & Butz, A. (2025). pyVPRM: A next-generation vegetation photosynthesis and respiration model for the post-MODIS era. Geoscientific Model Development, 18(14), 4713–4742. https://doi.org/10.5194/gmd-18-4713-2025
```

or using BibTeX:
```
@Article{gmd-18-4713-2025,
AUTHOR = {Glauch, T. and Marshall, J. and Gerbig, C. and Bot\'{\i}a, S. and Ga{\l}kowski, M. and Vardag, S. N. and Butz, A.},
TITLE = {\textit{pyVPRM}: a next-generation vegetation photosynthesis and respiration model for the post-MODIS era},
JOURNAL = {Geoscientific Model Development},
VOLUME = {18},
YEAR = {2025},
NUMBER = {14},
PAGES = {4713--4742},
URL = {https://gmd.copernicus.org/articles/18/4713/2025/},
DOI = {10.5194/gmd-18-4713-2025}
}
```

In case of any questions please write an E-Mail to theo.glauch@dlr.de. If you enjoy the model leave a :star:.

# Latest Update | 26-01-2026

With the new version - pyVPRM 5.3 - it is possible to replace the lowess filtering with a more stable Kalman filter. An example of the difference for EVI is shown below for the cropland site DE-RuS using Sentinel-2 data for 2022. The new function can be used by calling ```vprm_inst.kalman(...)``` instead of ```vprm_inst.lowess(...)``` for instances of the VPRM preprocessor class. 

<figure>
<img src="https://github.com/user-attachments/assets/8acdfa3f-26e8-4c40-9cd6-8c91002c22a8", height=300pt>
</figure> 

# About

**pyVPRM** is a Python package for estimating the exchange of CO₂ between the atmosphere and the terrestrial biosphere using the **Vegetation Photosynthesis and Respiration Model (VPRM)**.

The model represents both:
- **Gross Primary Productivity (GPP)**  
- **Ecosystem respiration (Reco)**  

The balance between these two components yields the **Net Ecosystem Exchange (NEE)**.

pyVPRM provides a flexible and modular implementation of VPRM, allowing users to combine different data sources for vegetation, land cover, and meteorological forcing.

- Supports multiple **satellite products** (e.g. Sentinel-2, MODIS, VIIRS)
- Compatible with various **land-cover datasets** (e.g. Copernicus Land Cover Service, ESA WorldCover 10 m, MapBiomas)
- Uses standard **meteorological reanalyses**, such as ECMWF ERA5
- Modular design that facilitates **extension and customization**

pyVPRM can be used for, among others:

1. **Parameter optimization** of VPRM against eddy-covariance flux tower observations (e.g. FLUXNET, ICOS)
2. **Regional CO₂ flux estimation and prediction** for user-defined domains
3. **Generation of VPRM input fields** for coupled atmospheric models such as the Weather Research and Forecasting (WRF) model


### Examples of pyVPRM Net Ecosystem Exchange

<figure>
<img src="https://github.com/user-attachments/assets/c099ef1f-6c5a-445c-bc6e-f6697c47b641", height=300pt>
</figure> 

<figure>
<img src="https://github.com/user-attachments/assets/26913805-f188-477a-9a85-08911a165b1e", height=300pt>
</figure> 

# How to use

## Installation

We generally recommend setting up a **dedicated virtual environment** for using `pyVPRM` and installing all required dependencies there.

If you are using **conda**, you may want to follow best practices for mixing `conda` and `pip`. The following blog post provides a good overview:  
https://www.anaconda.com/blog/using-pip-in-a-conda-environment

### Prerequisites

`pyVPRM` requires the **Earth System Modeling Framework (ESMF)** and its Python interface **ESMFpy** for all functionalities that involve regridding.

On many HPC systems specialized for Earth system modeling and climate research, ESMF is already pre-installed. If this is the case, make sure that both `esmf` and `esmpy` are available in your environment.

If you need to install ESMF yourself, you can find installation instructions here:
- ESMF GitHub repository: https://github.com/esmf-org
- Conda-forge ESMF package: https://github.com/conda-forge/esmf-feedstock

To ensure full ESMF functionality, it is also recommended to install **netCDF4**.

### Example conda setup

```bash
conda create -n pyvprm python=3.10
conda activate pyvprm

conda config --add channels conda-forge
conda config --set channel_priority strict

conda install dask netCDF4 esmf esmpy
```

Then install ```pyVPRM``` via pip

```
pip install pyVPRM
```

## Start your Project

To start your own `pyVPRM` project, you typically need to follow these steps:

1. Obtain the required **satellite data** for your region of interest
2. Obtain the corresponding **land-cover maps** for your region of interest
3. Create a **project configuration file**
4. Generate project-specific scripts using the functionality provided by the `VPRM` class in `VPRM.py`
5. Run the calculations

### Remarks

- If no interface exists yet for your satellite product or land-cover dataset, implement a new subclass in  
  `pyVPRM/sat_managers/`
- For new land-cover products, you must additionally provide a **mapping from land-cover classes to VPRM classes**.  
  This mapping is defined in a configuration file stored in `pyVPRM/vprm_configs/`
- Open-access land-cover datasets:
  - **Copernicus Global Land Service**: https://land.copernicus.eu/en/products/global-dynamic-land-cover
  - **ESA WorldCover**:  https://viewer.esa-worldcover.org
- Open-access satellite data:
  - **MODIS, VIIRS**:  https://e4ftl01.cr.usgs.gov
  - **Sentinel-2**:  https://scihub.copernicus.eu


## Examples

To help you get started with `pyVPRM`, we provide a companion repository containing example scripts, each with their own `README` files and in-line comments:

https://github.com/tglauch/pyVPRM_examples.git

The repository includes examples for:

- **Generating WRF input files**: `./wrf_preprocessor`
- **Generating VPRM fluxes (GPP / NEE)**: `./vprm_predictions`
- **Fitting VPRM parameters**: `./fit_vprm_parameters`
- **Downloading MODIS/VIIRS data using `pyVPRM`**: `./sat_data_download`
  
Clone the full example repository with:
```
git clone https://github.com/tglauch/pyVPRM_examples.git
```

The repository comes with pre-prepared input data, so you can run the examples immediately without downloading or preprocessing any datasets.

# Modular Structure

The pyVPRM implementation has a modular structure to allow for an easy replacement of satellite images and land cover maps, as well as the meteorologies. The file structure is as follows


```pyVPRM/sat_managers```

The ```satellite_data_manager``` class in this library is the basic data structure for all satellite image and land cover maps calcuations in pyVPRM. It provides function to reproject, transform, merge and crop satellite images. All other classes for specific satellite images or land cover maps, with the respective loading routines, are derived from this base class and implemented in the respective class files in the folder. 


```pyVPRM/meteorologies```

The classes in this folder provide the interface for the satellite data. This will usually strongly depend on the data availability. You'll likely need to make modifications here or implement your own class. All meteorology classes are derived from the base class in ```met_base_class.py```. An example to implement a new meteorology class can be found in ```era5_class_draft.py```.
