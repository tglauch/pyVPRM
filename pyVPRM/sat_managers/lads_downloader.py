# Script name: lads_downloader.py
# Author:      S. Botía
# Created:     2025-12-09
# Last update: 2025-12-14
# Description: This script downloads MODIS products hosted by the NASA LAADS DAAC. The EarthdataLAADS class provides helper functions to list and download MODIS HDF files for specific tiles and 8-day composite dates using Earthdata Login bearer-token authentication. The implementation is designed to: Integrate with the existing pyVPRM satellite data manager interface, support MODIS 8-day composite products (e.g. MOD09A1), be compatible with SLURM job arrays and use wget for downloads. 
# Requirements:
#      - Python ≥ 3.8
#      - Python packages: requests, loguru, 
#      - System tools: wget (available on PATH)
#      - Project dependencies: pyVPRM (for satellite_data_manager base class)
# Authentication:
#     - Valid NASA Earthdata Login bearer token with access to LAADS DAAC: https://ladsweb.modaps.eosdis.nasa.gov

import os
import re
import time
import subprocess
import requests
from datetime import timedelta, datetime
from loguru import logger
from pyVPRM.sat_managers.base_manager import satellite_data_manager


class EarthdataLAADS(satellite_data_manager):
    """
    Downloader for MODIS products hosted by NASA LAADS DAAC using Earthdata Login bearer-token authentication.
    """
    
    def __init__(self, datapath=None, sat_image_path=None, sat_img=None, product="MOD09A1.061"):
        """
        Initialize the EarthdataLAADS downloader.
        Parameters
        ----------
        datapath : str, optional
            Base path for downloaded data (handled by satellite_data_manager).
        sat_image_path : str, optional
            Path for satellite imagery (handled by satellite_data_manager).
        sat_img : optional
            Satellite image metadata (handled by satellite_data_manager).
        product : str
            MODIS product identifier including collection
            (e.g. "MOD09A1.061").
        """
        super().__init__(datapath, sat_image_path, sat_img)
        self.product = product

    def _parse_product_collection(self):
        """
        Parse the MODIS product string into product name and collection.
        Returns
        -------
        product_name : str
            MODIS product short name.
        collection : str
            MODIS collection number as string.
        """
        try:
            parts = str(self.product).split(".")
            product_name = parts[0]
            collection = str(int(parts[1])) if len(parts) > 1 else "61"
        except Exception:
            product_name = str(self.product)
            collection = "61"
        return product_name, collection

    def _init_downloader(self,dest,date,delta,username,
                         lonlat=None,pwd=None,token=None,
                         jpg=False, enddate=None, hv=None,):
        """
        Initialize a downloader config dictionary. This method has the same structure as the interface of 
        older Earthdata-based downloaders used in pyVPRM for compatibility.
        Parameters
        ----------
        dest : str
            Destination directory for downloaded files.
        date : datetime
            Start date for download.
        delta : timedelta
            Time step (unused here, kept for compatibility).
        username : str
            Earthdata username (not used; token authentication preferred).
        lonlat : tuple(float, float), optional
            Longitude/latitude pair used to compute MODIS tile.
        pwd : str, optional
            Password (unused; token authentication preferred).
        token : str, optional
            Earthdata bearer token.
        jpg : bool
            Placeholder for image download support (not used).
        enddate : datetime, optional
            End date for downloads.
        hv : tuple(int, int), optional
            MODIS tile indices (h, v).
        Returns
        -------
        dict
            Dictionary describing downloader configuration.
        """
        
        if hv is not None:
            h, v = hv
        elif lonlat is not None:
            # Convert lat/lon to MODIS tile indices
            h, v = self.lat_lon_to_modis(lonlat[1], lonlat[0])
        else:
            raise ValueError("Either hv or lonlat must be provided")

        tiles = f"h{int(h):02d}v{int(v):02d}"

        if enddate is None:
            enddate = date + timedelta(days=365)

        product_name, collection = self._parse_product_collection()
        base_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/{product_name}/"

        downloader = {"writeFilePath": dest,
                      "tiles": tiles,
                      "product": self.product,
                      "product_name": product_name,
                      "collection": collection,
                      "token": token,
                      "username": username,
                      "password": pwd,
                      "delta": delta,
                      "start_date": date,
                      "end_date": enddate,
                      "url": base_url,
                     }
        return downloader

    def _generate_modis_doys(self, start, end):
        """
        Generate valid MODIS 8-day composite dates within a time range.
        MODIS 8-day products are defined such that:
        (DOY - 1) % 8 == 0
        Parameters
        ----------
        start : datetime
            Start date.
        end : datetime
            End date.
        Yields
        ------
        datetime
            Date corresponding to the composite.
        str
            Day-of-year (DOY) formatted as zero-padded string.
        Yield datetime and doy string for MODIS 8-day composites inside [start,end].
        """
        
        cur = start
        while cur <= end:
            doy = cur.timetuple().tm_yday
            if (doy - 1) % 8 == 0:
                yield cur, f"{doy:03d}"
            cur += timedelta(days=1)

    def _list_dir(self, dir_url, token=None, timeout=30):
        """
        Retrieve the HTML directory listing from a LAADS directory.

        Parameters
        ----------
        dir_url : str
            URL of the LAADS directory to query.
        token : str, optional
            Earthdata bearer token.
        timeout : int
            HTTP request timeout in seconds.
        Returns
        -------
        str or None
            HTML content if successful, otherwise None.
        """
        headers = {}
        if token:
            headers["Authorization"] = token if token.startswith("Bearer") else f"Bearer {token}"
        try:
            r = requests.get(dir_url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.text
            else:
                logger.debug(f"_list_dir: {dir_url} returned {r.status_code}")
                return None
        except Exception as e:
            logger.exception(f"_list_dir error for {dir_url}: {e}")
            return None


    def list_doy_directory(self, year, doy, token=None, timeout=30):
        """
        Construct and query the LAADS directory for a specific year and DOY.
        Parameters
        ----------
        year : int
            Calendar year.
        doy : int
            Day of year.
        token : str, optional
            Earthdata bearer token.
            
        Returns
        -------
        str or None
            HTML directory listing.
        """
        
        product_name, collection = self._parse_product_collection()
        dir_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/{product_name}/{year}/{doy:03d}/"
        return self._list_dir(dir_url, token=token, timeout=timeout)


    def _wget_download(self, url, outpath, token):
        """
        Download a file using wget with Earthdata authentication.
        Uses a temporary '.part' file in case the download needs to be restarted. This avoids corrupted outputs.
        Parameters
        ----------
        url : str
            File URL.
        outpath : str
            Destination file path.
        token : str
            Earthdata bearer token.
        Returns
        -------
        bool
            True if download succeeded, False otherwise.
        """
        # ensure parent dir exists
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        tmp = outpath + ".part"
        # build command: use -q for quiet or remove to show progress
        cmd = ["wget",
               "-c",
               "--header", 
               f"Authorization: Bearer {token}",
               "-O", 
               tmp,
               url]

        logger.info("WGET CMD: " + " ".join(cmd))

        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                logger.error(f"wget failed (rc={proc.returncode}) for {url}: {proc.stderr.strip()}")
                # cleanup partial if exists
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                return False
            # move tmp -> final
            os.replace(tmp, outpath)
            return True
            
        except Exception as e:
            logger.exception(f"_wget_download exception for {url}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False

    def download_doy(self, year, doy, savepath, token=None, tile=None, resume=True, timeout=60):
        """
        Download all MODIS HDF files for a given year, DOY, and tile.
        Parameters
        ----------
        year : int
            Calendar year.
        doy : int
            Day of year.
        savepath : str
            Output directory.
        token : str
            Earthdata bearer token.
        tile : str or tuple
            MODIS tile identifier (e.g. "h09v09" or (9,9)).
        resume : bool
            Allow resuming partial downloads (handled by wget).
        timeout : int
            Directory listing timeout.
        Returns
        -------
        list[str]
            List of downloaded (or existing) file paths.
        """
        
        # normalize tile
        if isinstance(tile, (tuple, list)):
            tile = f"h{int(tile[0]):02d}v{int(tile[1]):02d}"
        if tile is None:
            raise ValueError("tile must be provided to download_doy (e.g. 'h09v09' or (9,9))")

        html = self.list_doy_directory(year, doy, token=token)
        # html = self.list_doy_directory(year, doy, token=token, timeout=timeout)
        if html is None:
            logger.info(f"Directory not available or empty for {year} DOY {int(doy):03d}")
            return []

        matches = self.find_hdfs_in_html(html, year, doy, tile)
        if not matches:
            logger.info(f"No HDFs found for tile {tile} on {year}-{int(doy):03d}")
            return []

        downloaded = []
        for fname in matches:
            #outdir = os.path.join(savepath, str(year))
            outdir = savepath
            outpath = os.path.join(outdir, fname)
            if os.path.exists(outpath):
                logger.info(f"Skipping existing file: {outpath}")
                downloaded.append(outpath)
                continue

            file_url = self.build_doy_url(year, doy) + fname
            logger.info(f"Downloading {file_url} -> {outpath}")
            ok = self._wget_download(file_url, outpath, token=token)
            if ok:
                downloaded.append(outpath)
            else:
                logger.error(f"Failed to download {file_url}")

            time.sleep(1)  # pause

        return downloaded
        
    def build_doy_url(self, year, doy):
        """
        Return the base URL for a given year and DOY, e.g.,
        https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD09A1/YYYY/DDD/
        """
        product_name, collection = self._parse_product_collection()
        return f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/{product_name}/{year}/{doy:03d}/"

    def find_hdfs_in_html(self, html, year, doy, tile):
        """
        Extract MODIS HDF filenames for a specific tile from a directory listing.
        Parameters
        ----------
        html : str
            HTML directory listing.
        year : int
            Calendar year.
        doy : int
            Day of year.
        tile : str
            MODIS tile identifier.
        Returns
        -------
        list[str]
            List of matching HDF filenames.
        """
        
        if html is None:
            return []
    
        # Build regex pattern
        product_name, _ = self._parse_product_collection()
        pattern = rf"{product_name}\.A{year}{doy:03d}\.{tile}\..*?\.hdf"
        matches = re.findall(pattern, html)
        return matches
