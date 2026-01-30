"""
Module for using digital elevation model (DEM) data
Automatically downloading terrain data from the national map.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pint import Quantity
from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays

from .. import ureg
from ..utils import Singleton
from .config import DEM_SETTINGS
from .downloader import DEM_CATALOG, GeoAsset, get_geospatial_data

# Load and format legend
# This file is manually created
# The category and color from https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description
# Height and gound type is customizable, followed this resource: https://www.pathloss.com/webhelp/terrain_data/terdat_clutter_clutdef.html
NLCDLEG = pd.read_csv(DEM_SETTINGS.NLCDLEG_FILE, skiprows=2)
# Convert RGBA to tuple
NLCDLEG["RGBA"] = [eval(rgba) for rgba in NLCDLEG["RGBA"]]
NLCDLEG["rgbint"] = [tuple([r / 255 for r in rgba]) for rgba in NLCDLEG["RGBA"]]
IDXCLUTTERH = np.zeros(NLCDLEG["Value"].max() + 1)
IDXNLCDCOLOR = np.zeros(NLCDLEG["Value"].max() + 1).tolist()
IDXSIGMA = np.zeros(NLCDLEG["Value"].max() + 1)
IDXER = np.zeros(NLCDLEG["Value"].max() + 1)
IDXRMSSLOPE = np.zeros(NLCDLEG["Value"].max() + 1)
for i in NLCDLEG["Value"].values:
    IDXCLUTTERH[i] = NLCDLEG.loc[NLCDLEG["Value"] == i]["Clutter Height (m)"].item()
    IDXNLCDCOLOR[i] = NLCDLEG.loc[NLCDLEG["Value"] == i]["rgbint"].item()
    IDXSIGMA[i] = NLCDLEG.loc[NLCDLEG["Value"] == i]["Conductivity (S/m)"].item()
    IDXER[i] = NLCDLEG.loc[NLCDLEG["Value"] == i]["Relative Permittivity"].item()
    IDXRMSSLOPE[i] = NLCDLEG.loc[NLCDLEG["Value"] == i]["RMS Slope (m)"].item()


def nlcdcat2clutterh(v):
    return IDXCLUTTERH[int(v)]


def nlcdcat2color(v):
    return IDXNLCDCOLOR[int(v)]


# Utility function to generate a VRT file
def _generate_vrt(vrt_path: Path, file_paths: list[Path]):
    """
    Creates a VRT file from a list of GeoTIFF paths using gdalbuildvrt.
    This is critical for lazy loading of merged data.
    """
    if not file_paths:
        raise ValueError("Cannot build VRT with an empty list of files.")

    # gdalbuildvrt command: -q for quiet, vrt_path is output, followed by input files
    # The list of input files must be passed as strings to subprocess
    command = ["gdalbuildvrt", "-q", str(vrt_path)] + [str(p) for p in file_paths]

    try:
        # Run the command and capture output/errors
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully created VRT: {vrt_path.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"gdalbuildvrt failed with error:\n{e.stderr}")
        raise RuntimeError("Failed to create VRT file.")
    except FileNotFoundError:
        logger.error("gdalbuildvrt command not found. Ensure GDAL is installed and in your PATH.")
        raise FileNotFoundError("GDAL (which includes gdalbuildvrt) is required to create VRTs.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during VRT creation: {e}")
        raise


class _DEM(Singleton):
    """Container class for digital elevation model (DEM) data."""

    def __init__(self, geo_asset: GeoAsset = DEM_CATALOG.USGS30, local_only: bool = True) -> None:
        # Initialize internal data caches to None for lazy loading
        self._data = None
        self._nlcd = None
        self.local_only = local_only
        self.geo_asset = geo_asset

    @property
    def data(self) -> xr.DataArray:
        """GeoTiff data as xr.DataArray."""
        if self._data is None:
            raise ValueError(
                "DEM data has not been loaded. Call load() first, or use an interpolation method."
            )
        return self._data

    @property
    def nlcd(self) -> xr.DataArray:
        """Land clutter data."""
        if self._nlcd is None:
            raise ValueError(
                "NLCD data not loaded. Call load_nlcd() first, or use an interpolation method."
            )
        return self._nlcd

    def clutterh(self, nlcd) -> xr.DataArray:
        """
        Get clutter height from land cover mapping.

        Parameters
        ----------
        nlcd : _type_
            _description_

        Returns:
        -------
        _type_
            _description_
        """
        clutter_h = xr.full_like(nlcd, 1.0)
        clutter_h.data = IDXCLUTTERH[nlcd.values.astype(int)]
        return clutter_h

    @property
    def geotiffs(self):
        """GeoTiff file paths."""
        return self._geotiffs

    def load(self, points: list) -> None:
        """
        Downloads GeoTIFF data from the lat,lon tuples if needed, generates mosaic, and loads into
        memory.
        Loads NLCD data also.
        Files are located in DEM_SETTINGS.DEM_FOLDER.

        Parameters
        ----------
        points : list of tuples
            Contains lat/lon points: [(lat0,lon0), (lat1,lon1),...,(latN,lonN)]
        """
        self._geotiffs = get_geospatial_data(self.geo_asset, points, local_only=self.local_only)
        if self.geotiffs is not None:
            logger.info(f"Loading geotiffs: {self.geotiffs}")
            chunks_dict = {"x": 1024, "y": 1024}

            # Merge GeoTIFFS in memory
            if len(self.geotiffs) == 1:
                data = open_rasterio(self.geotiffs[0], chunks=chunks_dict)
            else:
                try:
                    from osgeo import gdal

                    # Multiple tiles: build a VRT for lazy merging
                    # Use same vrt file for everything and just overwrite
                    vrt_file = DEM_SETTINGS.DEM_FOLDER / "hicsdem.vrt"

                    # Generate VRT
                    _generate_vrt(vrt_file, self.geotiffs)

                    # Open the VRT file for lazy reading
                    data = open_rasterio(vrt_file, masked=True, chunks=chunks_dict)
                except ImportError:
                    logger.info("Merging geotiffs...")
                    # Open each file
                    data_arrays = [open_rasterio(f, chunks=chunks_dict) for f in self.geotiffs]
                    # Merge
                    data = merge_arrays(data_arrays)

            # Rename dimensions
            data = data.rename({"x": "lon", "y": "lat"})
            min_lat = data.lat.min()
            max_lat = data.lat.max()
            min_lon = data.lon.min()
            max_lon = data.lon.max()
            # Change degrees to radians
            data = data.assign_coords(lat=np.deg2rad(data.lat))
            data = data.assign_coords(lon=np.deg2rad(data.lon))
            # Drop dims
            data = data.squeeze("band", drop=True)
            # Ensure data is float64
            data = data.astype(np.float64)

            # Fix auth_code for pyproj
            # https://pyproj4.github.io/pyproj/stable/gotchas.html
            # data.attrs["crs"] = data.attrs["crs"].replace("+init=", "")
            self._data = data

            # Land clutter data
            # Grab data
            try:
                with open_rasterio(DEM_SETTINGS.NLCD_FILE) as r:
                    nlcd = r.rio.clip_box(
                        minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326"
                    ).rio.reproject("EPSG:4326")

                # Flip y to be increasing
                nlcd = nlcd.isel(y=slice(None, None, -1))
                # Rename dimensions
                nlcd = nlcd.rename({"x": "lon", "y": "lat"})
                # Get rid of interpolated data
                nlcd = nlcd.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
                # Change degrees to radians
                nlcd = nlcd.assign_coords(lat=np.deg2rad(nlcd.lat))
                nlcd = nlcd.assign_coords(lon=np.deg2rad(nlcd.lon))
                # Drop dims
                nlcd = nlcd.squeeze("band", drop=True)

                self._nlcd = nlcd
            except:
                pass

    def interp(
        self, lat: xr.DataArray | None = None, lon: xr.DataArray | None = None, **kwargs
    ) -> xr.DataArray:
        """
        Interpolates the loaded geotiff data. Wraps xr.DataArray.interp.
        Enables data validation and ensure correct data is loaded.
        """
        # Check inputs
        if not isinstance(lat, xr.DataArray) or not isinstance(lon, xr.DataArray):
            raise ValueError("Inputs must be a DataArray")
        # Copy inputs before proceeding
        lats = lat.copy()
        lons = lon.copy()

        # Use bounds_error kwarg on scipy.interp to raise an error if extrapolating
        try:
            interp_dat = self.data.interp(
                lat=lats, lon=lons, **kwargs, kwargs=dict(bounds_error=True)
            )
        except ValueError:
            # Catch the error and load in data based off of points
            # Create lat,lon point list
            if isinstance(lats.data, Quantity):
                lapts = lats.data.to("radian").magnitude.copy()
            else:
                lapts = lats.data.copy()
            if isinstance(lons.data, Quantity):
                lopts = lons.data.to("radian").magnitude.copy()
            else:
                lopts = lons.data.copy()
            points = [
                (la.item(), lo.item())
                for la, lo in zip(
                    np.rad2deg(lapts.ravel()), np.rad2deg(lopts.ravel()), strict=False
                )
            ]

            # Load geotiff
            self.load(points)

            # Interpolate
            interp_dat = self.data.interp(lat=lats, lon=lons, **kwargs)

        return interp_dat

    def interp_nlcd(
        self,
        lat: xr.DataArray | None = None,
        lon: xr.DataArray | None = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Interpolates the loaded geotiff data. Wraps xr.DataArray.interp.
        Enables data validation and ensure correct data is loaded.
        """
        # Check inputs
        if not isinstance(lat, xr.DataArray) or not isinstance(lon, xr.DataArray):
            raise ValueError("Inputs must be a DataArray")
        # Copy inputs before proceeding
        lats = lat.copy()
        lons = lon.copy()

        # Use bounds_error kwarg on scipy.interp to raise an error if extrapolating
        try:
            interp_dat = self.nlcd.interp(
                lat=lats,
                lon=lons,
                method="nearest",
                **kwargs,
                kwargs=dict(bounds_error=True, fill_value=0),
            )

        except ValueError:
            # Catch the error and load in data based off of points
            # Create lat,lon point list
            if isinstance(lats.data, Quantity):
                lapts = lats.data.to("radian").magnitude.copy()
            else:
                lapts = lats.data.copy()
            if isinstance(lons.data, Quantity):
                lopts = lons.data.to("radian").magnitude.copy()
            else:
                lopts = lons.data.copy()
            points = [
                (la.item(), lo.item())
                for la, lo in zip(
                    np.rad2deg(lapts.ravel()), np.rad2deg(lopts.ravel()), strict=False
                )
            ]

            # Load NLCD
            self.load(points)

            # Interpolate
            interp_dat = self.nlcd.interp(
                lat=lats, lon=lons, method="nearest", **kwargs, kwargs=dict(fill_value=0)
            )

        return interp_dat

    def interp_clutter(
        self,
        lat: xr.DataArray | None = None,
        lon: xr.DataArray | None = None,
        **kwargs,
    ) -> xr.DataArray:
        nlcd = self.interp_nlcd(lat=lat, lon=lon, **kwargs)
        clutterh = self.clutterh(nlcd)

        # Mask first and last five points to help with ITM interpolation
        clutterh.loc[dict(distance=clutterh.distance[:5])] = 0
        clutterh.loc[dict(distance=clutterh.distance[-5:])] = 0

        return clutterh, nlcd


# Initialize DEM
DEM = _DEM()
