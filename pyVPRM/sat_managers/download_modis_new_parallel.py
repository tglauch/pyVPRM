"""
download_modis_new_parallel.py
==============================
Downloads MODIS products from NASA LAADS DAAC for a specific tile and year.
Designed to run as a SLURM array job.

Usage (standalone):
    python download_modis_new_parallel.py \\
        --tile h12v09 \\
        --year 2022 \\
        --token YOUR_EARTHDATA_TOKEN \\
        --output /path/to/output/ \\
        --product MCD19A1.061 \\
        --workers 4

Usage (SLURM array):
    1. Edit config.yaml with product, dates and tiles
    2. Generate task list:
        export N=$(python gen_tasks.py)
    3. Create a SLURM submit script, e.g.:

        #!/bin/bash
        #SBATCH --job-name=modis_dl
        #SBATCH --array=1-N%20
        #SBATCH --time=08:00:00
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=8G

        TASKFILE=tasklist_tile_year.json
        IDX=$((SLURM_ARRAY_TASK_ID - 1))
        TILE=$(jq    -r ".[$IDX].tile"    "$TASKFILE")
        YEAR=$(jq    -r ".[$IDX].year"    "$TASKFILE")
        PRODUCT=$(jq -r ".[$IDX].product" "$TASKFILE")
        TOKEN=$(cat token.txt)

        python download_modis_new_parallel.py \\
            --tile    "${TILE}"    \\
            --year    "${YEAR}"    \\
            --token   "${TOKEN}"   \\
            --output  "/path/to/output/" \\
            --product "${PRODUCT}" \\
            --workers ${SLURM_CPUS_PER_TASK}

    4. Submit:
        sbatch --array=1-${N}%20 your_submit_script.sh

Authentication:
    Get your NASA Earthdata bearer token at:
    https://urs.earthdata.nasa.gov/profile

Requirements:
    - wget available on PATH
    - jq available on PATH
    - pyVPRM installed or on PYTHONPATH
"""

import argparse, os, time
import os, time, sys
import yaml
from datetime import datetime, timedelta, date
sys.path.append("/work/mj0143/b301108/pyVPRM/pyVPRM")
from concurrent.futures import ThreadPoolExecutor, as_completed
from sat_managers.lads_downloader import EarthdataLAADS

p = argparse.ArgumentParser()
p.add_argument("--tile", required=True)
p.add_argument("--year", type=int, required=True)
p.add_argument("--token", required=True)
p.add_argument("--output", required=True)
p.add_argument("--product", required=True)
p.add_argument("--workers", type=int, default=4)
args = p.parse_args()

dl = EarthdataLAADS(product=args.product)

# build list of DOYs for that year (46 DOYs) using your class helper:
start = datetime(args.year,1,1)
end = datetime(args.year,12,31)
doys = [int(doy) for dt, doy in dl._generate_modis_doys(start,end)]

def job_for_doy(doy):
    return dl.download_doy(year=args.year, doy=doy, savepath=os.path.join(args.output,str(args.year)),
                           token=args.token, tile=args.tile, resume=True)

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = { ex.submit(job_for_doy, doy): doy for doy in doys }
    for fut in as_completed(futures):
        doy = futures[fut]
        try:
            files = fut.result()
            # write per-doy success to a local job file or stdout
        except Exception as e:
            # retry or log fail
            pass
