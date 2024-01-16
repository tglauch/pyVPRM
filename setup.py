from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyvprm',
    version='1.0.0',
    description='Vegetation Photosynthesis and Respiration Model',
    long_description='A tool to calculate the CO2 exchange flux between atmosphere and terrestrial biosphere using the Vegetation Photosynthesis and Respiration Model',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ],
    keywords='CO2 Biosphere Atmosphere Physics VPRM',
    url='https://github.com/tkerscher/blast',
    author='Theo Glauch',
    author_email='theo.glauch@drl.de',
    license='MIT',
    packages=['pyvprm'],
    install_requires=[
        'numpy',
        'scipy',
        'xarray',
        'xesmf',],
    entry_points={
        'console_scripts': ['pyvprm=pyvprm.pyvprm:main']
    },
    include_package_data=True,
    zip_safe=False)
