# MCD19A1 MAIAC Downloader

Download MODIS products from NASA LAADS DAAC using SLURM.

## Files
- lads_downloader.py: main downloader class
- product_registry.py: product catalog  
- gen_tasks.py: generates task list
- download_modis_new_parallel.py: parallel download script
- download_modis_new_submit.sh: SLURM submit script

## Usage
1. Edit config.yaml (product, dates, tiles)
2. Create token.txt (NASA Earthdata token)
3. Run: export N=$(python gen_tasks.py)
4. Run: sbatch --array=1-${N}%20 download_modis_new_submit.sh

## Supported products
- MCD19A1.061: MAIAC daily 500m
- MCD43A4.061: BRDF-corrected daily 500m
- MOD09A1.061: Terra 8-day 500m
- MOD13A1.061: Terra 16-day 500m
