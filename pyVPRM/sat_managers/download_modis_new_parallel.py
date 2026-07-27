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
