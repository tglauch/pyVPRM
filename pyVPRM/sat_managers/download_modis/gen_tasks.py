# Generates tile-year tasks

import json, yaml
import sys
import os
from datetime import datetime
from pyVPRM.sat_managers.lads_downloader import EarthdataLAADS

def generate_tile_year_tasks(config):
    start = datetime.fromisoformat(config["start_date"])
    end   = datetime.fromisoformat(config["end_date"])
    tiles = config["tiles"]
    product = config["product"]

    dl = EarthdataLAADS(product=product)
    # gather unique years in range
    years = sorted({d.year for d, _ in [ (dt, None) for dt,_ in dl._generate_modis_doys(start,end) ]})
    tasks = []
    for year in years:
        for tile in tiles:
            tasks.append({"year": year, "tile": tile, "product": product})
    return tasks

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    tasks = generate_tile_year_tasks(config)
    with open("tasklist_tile_year.json","w") as fo:
        json.dump(tasks, fo, indent=2)
    #print("Wrote tasklist_tile_year.json, total:", len(tasks))
    # Set environment variable N for current shell session
    os.environ["N"] = str(len(tasks))
    print(len(tasks))
