
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14216613.svg)](https://doi.org/10.5281/zenodo.14216613)
[![PyPI version](https://img.shields.io/pypi/v/pyVPRM.svg)](https://pypi.org/project/pyVPRM/)
[![License](https://img.shields.io/pypi/l/pyVPRM.svg)](https://pypi.org/project/pyVPRM/)
[![Downloads](https://img.shields.io/pypi/dm/pyVPRM.svg)](https://pypi.org/project/pyVPRM/)
![GitHub stars](https://img.shields.io/github/stars/tglauch/pyVPRM?style=social)
![Python Version](https://img.shields.io/pypi/pyversions/pyVPRM)

<figure>
<img width="100%" alt="github_logo 001" src="https://github.com/user-attachments/assets/1628353c-802d-4644-8dbc-0a327a72ab24" />
</figure> 

**pyVPRM** and its extension **pyVPRNN** are data-driven models for analyzing and estimating carbon flux exchange between the atmosphere and the terrestrial biosphere — from single flux-tower footprints up to global scale — using multi-spectral satellite observations.

📄 **Paper:** Glauch et al. (2025), *pyVPRM: a next-generation vegetation photosynthesis and respiration model for the post-MODIS era*, [Geoscientific Model Development, 18(14), 4713–4742](https://gmd.copernicus.org/articles/18/4713/2025/). A pyVPRNN paper is in preparation.

🚀 **New here?** Start with the [example repository](https://github.com/tglauch/pyVPRM_examples.git) — it's the fastest way to see the pipeline end to end.

### Citation

If you use this package in your research, please cite:

> Glauch, T., Marshall, J., Gerbig, C., Botía, S., Gałkowski, M., Vardag, S. N., & Butz, A. (2025). pyVPRM: A next-generation vegetation photosynthesis and respiration model for the post-MODIS era. *Geoscientific Model Development*, 18(14), 4713–4742. https://doi.org/10.5194/gmd-18-4713-2025

<details>
<summary>BibTeX</summary>

```bibtex
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
</details>

### Questions?

Open an issue, or reach out directly: **theo.glauch@dlr.de**

---
⭐ If pyVPRM is useful for your work, consider starring the repo — it helps visibility and continued support for the project.

# About

**pyVPRM** is a Python package for estimating CO₂ exchange between the atmosphere and the terrestrial biosphere using the **Vegetation Photosynthesis and Respiration Model (VPRM)**.

VPRM represents two opposing fluxes — **Gross Primary Productivity (GPP)**, the uptake of CO₂ through photosynthesis, and **ecosystem respiration (Reco)**, the release of CO₂ back to the atmosphere. Their balance gives the **Net Ecosystem Exchange (NEE)**, the net flux actually measured at a flux tower or resolved on a model grid.

pyVPRM provides a flexible, modular implementation that lets you mix and match data sources for vegetation, land cover, and meteorological forcing:

- **Satellite products** — Sentinel-2, MODIS, VIIRS, and others
- **Land-cover datasets** — Copernicus Land Cover Service, ESA WorldCover (10 m), MapBiomas
- **Meteorological forcing** — standard reanalyses such as ECMWF ERA5
- **Modular architecture** — built to be extended and customized rather than used only as-is

### What you can do with it

1. **Optimize VPRM parameters** against eddy-covariance flux tower observations (FLUXNET, ICOS, ...)
2. **Estimate and predict CO₂ fluxes** for any user-defined domain, from regional to global scale
3. **Generate VPRM input fields** for coupled atmospheric models such as the Weather Research and Forecasting (WRF) model
4. **Partition measured fluxes into GPP and Reco** using a process-informed neural network approach (**pyVPRNN**)
5. **Interpret partitioned GPP/Reco responses** to meteorological drivers using explainable AI

### Examples

<details>
<summary><strong>pyVPRM NEE in the Amazon Basin</strong> (click to expand)</summary>

<video src="https://github.com/user-attachments/assets/eaa42d58-01ef-4f49-bb5d-9463b310ab02" 
       controls width="500"></video>

</details>

<details>
<summary><strong>pyVPRNN Partitioning at DK-Sor</strong> (click to expand)</summary>

Animated example of pyVPRNN's flux partitioning: modeled GPP, ecosystem respiration, and net ecosystem exchange (NEE) evolve alongside the flux footprint and observed NEE, showing how the model tracks measured fluxes over time.
<video src="https://github.com/user-attachments/assets/93d177b4-fd0e-40dd-83d4-097709618e1c" 
       controls width="500"></video>
</details>

# How to Use

## Installation

We recommend setting up a **dedicated virtual environment** for `pyVPRM` and installing all dependencies there.

If you're using **conda**, it's worth following best practices for mixing `conda` and `pip` — this post from Anaconda gives a good overview: [Using pip in a conda environment](https://www.anaconda.com/blog/using-pip-in-a-conda-environment).

### Prerequisites

`pyVPRM` requires the **Earth System Modeling Framework (ESMF)** and its Python interface, **ESMPy**, for any functionality involving regridding.

Many HPC systems built for Earth system modeling and climate research already have ESMF pre-installed — if so, just confirm that both `esmf` and `esmpy` are available in your environment. If not, you'll need to install it yourself:

- [ESMF GitHub repository](https://github.com/esmf-org)
- [ESMF on conda-forge](https://github.com/conda-forge/esmf-feedstock)

Installing **netCDF4** alongside ESMF is also recommended, for full functionality.

### Example conda setup

```bash
conda create -n pyvprm python=3.14
conda activate pyvprm

conda config --add channels conda-forge
conda config --set channel_priority strict

conda install dask netCDF4 esmf esmpy
```

Then install `pyVPRM` itself via pip:

```bash
pip install pyVPRM
```

> If you're actively developing `pyVPRM` (rather than just using it), install it as an editable clone instead so local edits are picked up immediately without reinstalling.

## Start Your Project

To start your own `pyVPRM` project, you'll typically:

1. Obtain the **satellite data** for your region of interest
2. Obtain the corresponding **land-cover map(s)** for your region of interest
3. Create a **project configuration file**
4. Generate project-specific scripts using the `VPRM` class in `VPRM.py`
5. Run the calculations

### Data Sources

- **Land cover**
  - [Copernicus Global Land Service](https://land.copernicus.eu/en/products/global-dynamic-land-cover)
  - [ESA WorldCover](https://viewer.esa-worldcover.org)
- **Satellite imagery**
  - **MODIS / VIIRS**: [LP DAAC Data Pool](https://e4ftl01.cr.usgs.gov)
  - **Sentinel-2**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (the former Copernicus Open Access Hub / SciHub was permanently retired in November 2023 — use this instead)

### Extending pyVPRM

If no interface exists yet for your satellite product or land-cover dataset:

- Implement a new subclass in `pyVPRM/sat_managers/`
- For a new land-cover product, also provide a **mapping from its land-cover classes to VPRM classes**, defined in a config file under `pyVPRM/vprm_configs/`

## Package Structure

`pyVPRM` follows a modular design: satellite imagery, land cover maps, meteorological data, flux tower datasets, and VPRM model implementations can each be swapped or extended independently. The directory layout reflects this:

---

### `pyVPRM/sat_managers`

Core classes for handling satellite imagery and land cover maps.

`satellite_data_manager` is the base class for all satellite- and land-cover-related data in `pyVPRM`, providing shared functionality — reprojection, transformation, merging, cropping — that every product-specific subclass builds on. Each supported satellite or land-cover product has its own subclass file in this directory.

---

### `pyVPRM/vprm_configs`

Configuration files defining, for each supported land-cover product, the mapping from that product's own land-cover classes to VPRM's internal vegetation classes (plus the associated `tmin`/`topt`/`tmax`/`tlow` temperature parameters per class). Required for any land-cover product you add — see [Extending pyVPRM](#extending-pyvprm) above.

---

### `pyVPRM/meteorologies`

Classes providing the meteorological interface to the model.

Meteorological data handling depends heavily on what's available on your own system — a generic, widely-usable option is the [Destination Earth platform](https://platform.destine.eu/). All meteorology classes inherit from `met_base_class.py`; `era5_class_draft.py` is a worked example for adding a new meteorology source.

---

### `pyVPRM/vprm_models`

The different VPRM model implementations. Each one takes a **VPRM preprocessor** instance and a **meteorology** object as input.

---

### `pyVPRM/flux_tower_libs`

Interfaces to flux tower datasets (FLUXNET, ICOS, etc.), including functionality for computing tower footprints from eddy-covariance measurements.
