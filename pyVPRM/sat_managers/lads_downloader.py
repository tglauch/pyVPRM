# lads_downloader.py
import os
import re
import time
import subprocess
import requests
from datetime import timedelta, datetime
from loguru import logger
from pyVPRM.sat_managers.base_manager import satellite_data_manager
from pyVPRM.sat_managers.product_registry import get_product_info   # ← NUEVO


class EarthdataLAADS(satellite_data_manager):
    """
    Downloader for MODIS/VIIRS products hosted by NASA LAADS DAAC.

    Supports:
      - Daily products      (e.g. MCD19A1 MAIAC, MOD09GA)
      - 8-day composites    (e.g. MOD09A1, MYD09A1)
      - 16-day composites   (e.g. MOD13A1)
      - Yearly products     (e.g. MCD12Q1)

    The product cadence is resolved automatically from product_registry.py.
    To add support for a new product, simply add it to the registry.

    Parameters
    ----------
    product : str
        MODIS product + collection, e.g. ``"MCD19A1.061"``.
    """

    def __init__(self, datapath=None, sat_image_path=None,
                 sat_img=None, product="MOD09A1.061"):
        super().__init__(datapath, sat_image_path, sat_img)
        self.product = product

        product_name, _ = self._parse_product_collection()
        info = get_product_info(product_name)          # ← consulta el registry
        self._cadence  = info["cadence"]               # "daily"|"8day"|"16day"|"yearly"
        self._file_ext = info["file_ext"]              # ".hdf"
        self._is_daily = self._cadence == "daily"      # shortcut booleano

        logger.info(
            f"EarthdataLAADS initialized | product={self.product} "
            f"| cadence={self._cadence}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Parsing helpers
    # ─────────────────────────────────────────────────────────────────

    def _parse_product_collection(self):
        """Split 'MCD19A1.061' → ('MCD19A1', '61')."""
        parts = str(self.product).split(".")
        product_name = parts[0]
        try:
            collection = str(int(parts[1])) if len(parts) > 1 else "61"
        except ValueError:
            collection = "61"
        return product_name, collection

    # ─────────────────────────────────────────────────────────────────
    # Date generation  ← lógica central que soporta todos los cadences
    # ─────────────────────────────────────────────────────────────────

    def _generate_modis_doys(self, start, end):
        """
        Yield ``(datetime, doy_str)`` for every valid acquisition date
        in ``[start, end]``, according to the product cadence.

        Cadence rules
        -------------
        daily  → every calendar day
        8day   → (doy - 1) % 8  == 0
        16day  → (doy - 1) % 16 == 0
        yearly → DOY 001 only
        """
        cadence_map = {
            "daily":  1,
            "8day":   8,
            "16day":  16,
            "yearly": 365,   # handled separately below
        }

        cur = start
        while cur <= end:
            doy = cur.timetuple().tm_yday

            if self._cadence == "daily":
                yield cur, f"{doy:03d}"

            elif self._cadence == "8day":
                if (doy - 1) % 8 == 0:
                    yield cur, f"{doy:03d}"

            elif self._cadence == "16day":
                if (doy - 1) % 16 == 0:
                    yield cur, f"{doy:03d}"

            elif self._cadence == "yearly":
                if doy == 1:
                    yield cur, f"{doy:03d}"

            cur += timedelta(days=1)

    # ─────────────────────────────────────────────────────────────────
    # URL helpers
    # ─────────────────────────────────────────────────────────────────

    def build_doy_url(self, year, doy):
        product_name, collection = self._parse_product_collection()
        return (
            f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/"
            f"{collection}/{product_name}/{year}/{doy:03d}/"
        )

    def _list_dir(self, dir_url, token=None, timeout=30):
        headers = {}
        if token:
            headers["Authorization"] = (
                token if token.startswith("Bearer") else f"Bearer {token}"
            )
        try:
            r = requests.get(dir_url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.text
            logger.debug(f"_list_dir: {dir_url} → HTTP {r.status_code}")
            return None
        except Exception as exc:
            logger.exception(f"_list_dir error: {exc}")
            return None

    def list_doy_directory(self, year, doy, token=None, timeout=30):
        url = self.build_doy_url(year, int(doy))
        logger.debug(f"Listing: {url}")
        return self._list_dir(url, token=token, timeout=timeout)

    def find_hdfs_in_html(self, html, year, doy, tile):
        """Extract HDF filenames matching tile from HTML directory listing."""
        if html is None:
            return []
        product_name, _ = self._parse_product_collection()
        doy_str = f"{int(doy):03d}"
        pattern = (
            rf"{re.escape(product_name)}"
            rf"\.A{year}{doy_str}"
            rf"\.{re.escape(tile)}"
            rf"\.[^\"<>\s]*?\.hdf"
        )
        matches = re.findall(pattern, html, re.IGNORECASE)
        seen, unique = set(), []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return unique

    # ─────────────────────────────────────────────────────────────────
    # Download engine
    # ─────────────────────────────────────────────────────────────────

    def _wget_download(self, url, outpath, token):
        os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
        tmp = outpath + ".part"
        cmd = [
            "wget", "-c",
            "--header", f"Authorization: Bearer {token}",
            "-O", tmp,
            url,
        ]
        logger.info("wget → " + " ".join(cmd))
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                logger.error(
                    f"wget failed (rc={proc.returncode}): {proc.stderr.strip()}"
                )
                if os.path.exists(tmp):
                    os.remove(tmp)
                return False
            os.replace(tmp, outpath)
            return True
        except Exception as exc:
            logger.exception(f"_wget_download exception: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False

    def download_doy(self, year, doy, savepath, token=None,
                     tile=None, resume=True, timeout=60, sleep=1):
        """Download all HDF files for year/doy/tile."""
        if isinstance(tile, (tuple, list)):
            tile = f"h{int(tile[0]):02d}v{int(tile[1]):02d}"
        if tile is None:
            raise ValueError("tile must be provided")

        doy = int(doy)
        html = self.list_doy_directory(year, doy, token=token, timeout=timeout)
        if html is None:
            logger.warning(f"Directory unavailable: {self.build_doy_url(year, doy)}")
            return []

        matches = self.find_hdfs_in_html(html, year, doy, tile)
        if not matches:
            logger.info(f"No files found | tile={tile} {year}-{doy:03d}")
            logger.debug(f"HTML snippet:\n{html[:800]}")
            return []

        logger.info(f"Found {len(matches)} file(s): {matches}")
        downloaded = []
        base_url = self.build_doy_url(year, doy)

        for fname in matches:
            outpath = os.path.join(savepath, fname)
            if os.path.exists(outpath):
                logger.info(f"Skipping existing: {outpath}")
                downloaded.append(outpath)
                continue
            ok = self._wget_download(base_url + fname, outpath, token=token)
            if ok:
                logger.success(f"Saved → {outpath}")
                downloaded.append(outpath)
            else:
                logger.error(f"Failed: {base_url + fname}")
            time.sleep(sleep)

        return downloaded

    # ─────────────────────────────────────────────────────────────────
    # pyVPRM compatibility shim
    # ─────────────────────────────────────────────────────────────────

    def _init_downloader(self, dest, date, delta, username,
                         lonlat=None, pwd=None, token=None,
                         jpg=False, enddate=None, hv=None):
        if hv is not None:
            h, v = hv
        elif lonlat is not None:
            h, v = self.lat_lon_to_modis(lonlat[1], lonlat[0])
        else:
            raise ValueError("Either hv or lonlat must be provided")

        if enddate is None:
            enddate = date + timedelta(days=365)

        product_name, collection = self._parse_product_collection()
        return {
            "writeFilePath": dest,
            "tiles":         f"h{int(h):02d}v{int(v):02d}",
            "product":       self.product,
            "product_name":  product_name,
            "collection":    collection,
            "cadence":       self._cadence,      # ← nuevo campo útil
            "token":         token,
            "username":      username,
            "password":      pwd,
            "delta":         delta,
            "start_date":    date,
            "end_date":      enddate,
            "url":           self.build_doy_url(date.year, date.timetuple().tm_yday),
        }