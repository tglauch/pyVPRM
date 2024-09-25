import xarray as xr
import glob
import os
import time
import numpy as np
from dateutil import parser
import pygrib
import copy
import uuid
import datetime


class met_data_handler_base:
    """
    Base class for all meteorologies
    """

    def __init__(self):
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        return

    def change_date(self, year=None, month=None, day=None, hour=None):
        # Caution: The date as argument corresponds to the END of the ERA5 integration time.

        new_date = False
        if (day != self.day) & (day != None):
            self.day = day
            new_date = True

        if (month != self.month) & (month != None):
            self.month = month
            new_date = True

        if (year != self.year) & (year != None):
            self.year = year
            new_date = True

        if new_date:
            self._init_data_for_day()

        if (hour != self.hour) & (hour != None):
            self.hour = hour
            new_date = True

        if new_date:
            self._load_data_for_hour()
        return

    def _init_data_for_day(self):
        return

    def _load_data_for_hour(self):
        return

    def regrid(
        self,
        lats=None,
        lons=None,
        dataset=None,
        n_cpus=1,
        weights=None,
        overwrite_regridder=False,
    ):
        return
