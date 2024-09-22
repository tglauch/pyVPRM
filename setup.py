from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="pyVPRM",
    version="3.0",
    description="Vegetation Photosynthesis and Respiration Model",
    long_description="A tool to calculate the CO2 exchange flux between atmosphere and terrestrial biosphere using the Vegetation Photosynthesis and Respiration Model",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    keywords="CO2 Biosphere Atmosphere Physics VPRM",
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
    ],
    include_package_data=True,
    zip_safe=False,
)
