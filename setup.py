from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="pyVPRM",
    version="5.3",
    description="Vegetation Photosynthesis and Respiration Model",
    long_description="pyVPRM is a framework for data-driven modelling and interpreting atmosphere-biosphere CO2 fluxes — from eddy-covariance towers to the global scale.",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3.14"
    ],
    keywords="CO2 Biosphere Atmosphere Physics VPRM Partitioning Neural-Networks GPP NEE Respiration",
    url="https://github.com/tglauch/pyVPRM/",
    author="Theo Glauch",
    author_email="theo.glauch@drl.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "xesmf",
        "pyproj",
        "joblib",
        "uuid",
        "pandas",
        "astropy",
        "pyyaml",
        "rasterio",
        "rioxarray",
        "geopandas",
        "h5py",
        "pytz",
        "tzwhere",
        "timezonefinder",
        "pygrib",
        "matplotlib",
        "lxml",
        "requests",
        "statsmodels",
        "loguru",
        "fiona",
        "pykalman",
        "numexpr",
        "cartopy"
    ],
    include_package_data=True,
    zip_safe=False,
)
