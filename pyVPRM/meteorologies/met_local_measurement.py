import xarray as xr
import glob
import os
import time
import numpy as np
from dateutil import parser
import copy
import datetime
from pyVPRM.meteorologies.met_base_class import met_data_handler_base
import pandas as pd
from loguru import logger

key_map = {"time": "TIMESTAMP_START", "ssrd": "SW_IN", "t2m": "TA"}


class met_data_handler(met_data_handler_base):

    def __init__(self, data_file, names=key_map):
        super().__init__()
        self.names = names
        self.non_time_keys = [
            self.names[i] for i in self.names.keys() if i not in "time"
        ]
        self.data = pd.read_csv(
            data_file, usecols=[self.names[i] for i in self.names.keys()]
        )
        self.data[self.names["time"]] = pd.to_datetime(self.data[self.names["time"]])
        self.data[self.names["t2m"]] = self.data[self.names["t2m"]] + 273.15
        self.data[self.names["ssrd"]] = self.data[self.names["ssrd"]] * 3600
        return

    def get_data(self, lonlat=None, key=None):
        # Return ERA5 data for lonlat if lonlat is not None else return all data.
        # Pick a specific key if key is not None. Return as xarray dataset

        date_int = np.datetime64(
            "{}-{:02d}-{:02d} {:02d}:00:00".format(
                self.year, self.month, self.day, self.hour
            )
        )
        row = self.data.loc[self.data[self.names["time"]] == date_int]
        if key is None:
            return row[self.non_time_keys].values
        else:
            return float(row[self.names[key]].values)


if __name__ == "__main__":
    year = "2000"
    month = 2
    day = 20
    hour = 5  # UTC hour
    era5_handler = ERA5(year, month, day)
    era5_handler.change_date(hour)
    ret = era5_handler.get_data()
    logger.info(ret)
