"""
Module for using digital elevation model (DEM) data
Automatically downloading terrain data from the national map.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pint import Quantity
from rasterio.enums import Resampling
from rioxarray import open_rasterio

from ..utils import Singleton
from .config import DEM_SETTINGS
from .downloader import DEM_CATALOG, GeoAsset, get_geospatial_data
from .setup_assets import generate_rf_csv

# For linting
if TYPE_CHECKING:
    import xarray as xr

    # from pint import Quantity
    from rasterio.enums import Resampling


# Utility function to generate a VRT file
def _generate_vrt(
    vrt_path: Path, file_paths: list[Path], dst_crs: str, resample_alg: str = "nearest"
):
    """
    Creates a reprojected VRT by first building a mosaic and then warping it.
    This ensures all tiles are included and the projection is correct.
    """
    if not file_paths:
        raise ValueError("Cannot build VRT with an empty list of files.")

    input_files = [str(p) for p in file_paths]
    # We use a temporary VRT to hold the mosaic structure
    temp_mosaic_vrt = vrt_path.with_suffix(".mosaic.vrt")
    from osgeo import gdal

    try:
        # 1. Create a standard mosaic VRT on disk.
        # This writes the actual file paths into the XML.
        gdal.BuildVRT(str(temp_mosaic_vrt), input_files)

        # 2. Warp the mosaic VRT to create the final reprojected VRT.
        # This will now reference temp_mosaic_vrt by path, not memory handle.
        options = gdal.WarpOptions(
            format="VRT",
            dstSRS=dst_crs,
            resampleAlg="nearest",
            outputBounds=None,
            multithread=True,
            errorThreshold=0,  # Disable coordinate approximation
            warpOptions=["SRC_METHOD=NO_GEOTRANSFORM"],  # Ensure pixel center alignment
        )

        ds = gdal.Warp(str(vrt_path), str(temp_mosaic_vrt), options=options)

        if ds is None:
            raise RuntimeError("GDAL Warp failed.")

        # 3. Clean up: Close the dataset to flush to disk
        ds = None
        logger.info(f"VRT created: {vrt_path}")

    except Exception as e:
        logger.error(f"VRT generation failed: {e}")
        raise


def load_geotiffs(
    geotiffs, vrt_suffix, dst_crs, minmax_latlon=None, resampling: Resampling | None = None
):
    if resampling is None:
        resampling = Resampling.nearest

    chunks_dict = {"x": 1024, "y": 1024}
    # Create a unique VRT for this specific data type
    # Place it in a .vrt subfolder to keep the main folder clean
    vrt_dir = DEM_SETTINGS.DEM_FOLDER / ".vrt"
    vrt_dir.mkdir(exist_ok=True)
    vrt_file = vrt_dir / f"hics_{vrt_suffix}.vrt"

    try:
        from osgeo import gdal

        # Multiple tiles: build a VRT for lazy merging
        # Use same vrt file for everything and just overwrite
        # Generate VRT
        _generate_vrt(vrt_file, geotiffs, dst_crs, resample_alg=resampling.name.lower())

        # Open the VRT file for lazy reading
        data = open_rasterio(vrt_file, masked=True, chunks=chunks_dict)
    except ImportError:
        from rioxarray.merge import merge_arrays

        logger.info("Merging geotiffs...")
        # Open each file
        data_arrays = [open_rasterio(f, chunks=chunks_dict) for f in geotiffs]
        # Merge
        data = merge_arrays(data_arrays)
        # Reproject if needed
        if data.rio.crs != dst_crs:
            logger.info(f"Reprojecting clipped area to EPSG:4326 using {resampling.name}")
            data = data.rio.reproject("EPSG:4326", resampling=resampling)
    # Clip
    if minmax_latlon is not None:
        data = data.rio.clip_box(
            minx=minmax_latlon[0][0],
            miny=minmax_latlon[1][0],
            maxx=minmax_latlon[0][1],
            maxy=minmax_latlon[1][1],
            crs=dst_crs,
        )

    # Rename dimensions
    data = data.rename({"x": "lon", "y": "lat"})

    # Change degrees to radians
    data = data.assign_coords(lat=np.deg2rad(data.lat))
    data = data.assign_coords(lon=np.deg2rad(data.lon))
    # Drop dims
    data = data.squeeze("band", drop=True)
    # Ensure data is float64
    data = data.astype(np.float64)

    return data


class _DEM(Singleton):
    """Container class for digital elevation model (DEM) data."""

    def __init__(
        self,
        geo_asset: GeoAsset = DEM_CATALOG.USGS30,
        lc_asset: GeoAsset = DEM_CATALOG.LULCv02,
        local_only: bool = True,
    ) -> None:
        # Initialize internal data caches to None for lazy loading
        self._data = None
        self._nlcd = None
        self.local_only = local_only
        self.geo_asset = geo_asset
        self.lc_asset = lc_asset
        # Lazy-loaded legend attributes
        self._nlcd_leg = None
        self._lookup_tables = {}

    def _ensure_legend_loaded(self):
        """Lazy-loads the NLCD legend and builds lookup tables only when needed."""
        if self._nlcd_leg is not None:
            return

        leg_file = DEM_SETTINGS.NLCDLEG_FOLDER / f"{self.lc_asset.collection}.csv"
        if not leg_file.exists():
            logger.info(f"Generating legend file: {leg_file}")
            generate_rf_csv(leg_file.stem)

        logger.info("Loading NLCD Legend and building lookup tables...")
        df = pd.read_csv(leg_file)

        # Process colors
        df["RGBA"] = df["RGBA"].apply(eval)
        df["rgbint"] = df["RGBA"].apply(lambda rgba: tuple(r / 255 for r in rgba))

        # Build Lookup Tables (LUTs) based on max NLCD value
        max_val = df["Value"].max()
        clutter_h = np.zeros(max_val + 1)
        nlcd_color = [(0, 0, 0, 0)] * (max_val + 1)
        sigma = np.zeros(max_val + 1)
        er = np.zeros(max_val + 1)
        rms_slope = np.zeros(max_val + 1)
        for _, row in df.iterrows():
            idx = int(row["Value"])
            clutter_h[idx] = row.get("Clutter Height (m)", 0.0)
            nlcd_color[idx] = row["rgbint"]
            sigma[idx] = row["Conductivity (S/m)"]
            er[idx] = row["Relative Permittivity"]
            rms_slope[idx] = row["RMS Slope"]

        self._nlcd_leg = df
        self._lookup_tables = {
            "clutter_h": clutter_h,
            "colors": nlcd_color,
            "sigma": sigma,
            "er": er,
            "rms_slope": rms_slope,
        }

    @property
    def nlcd_legend(self):
        self._ensure_legend_loaded()
        return self._nlcd_leg

    @property
    def idx_clutter_h(self):
        self._ensure_legend_loaded()
        return self._lookup_tables["clutter_h"]

    @property
    def idx_colors(self):
        self._ensure_legend_loaded()
        return self._lookup_tables["colors"]

    @property
    def idx_er(self):
        self._ensure_legend_loaded()
        return self._lookup_tables["er"]

    @property
    def idx_sigma(self):
        self._ensure_legend_loaded()
        return self._lookup_tables["sigma"]

    @property
    def idx_rms_slope(self):
        self._ensure_legend_loaded()
        return self._lookup_tables["rms_slope"]

    def nlcdcat2clutterh(self, v):
        self._ensure_legend_loaded()
        return self.idx_clutter_h[int(v)]

    def nlcdcat2color(self, v):
        return self.idx_colors[int(v)]

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

    def clutterh(self, nlcd: xr.DataArray) -> xr.DataArray:
        """
        Get clutter height from land cover mapping.

        Parameters
        ----------
        nlcd : xr.DataArray
            Land cover data from self.interp_nlcd

        Returns:
        -------
        Clutter height from the legend file: xr.DataArray
        """
        clutter_h = xr.full_like(nlcd, 1.0)
        clutter_h.data = self.idx_clutter_h[nlcd.values.astype(int)]
        return clutter_h

    @property
    def geotiffs(self):
        """GeoTiff file paths."""
        return self._geotiffs

    @property
    def lcgeotiffs(self):
        """Land cover GeoTiff file paths."""
        return self._lcgeotiffs

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
        self._lcgeotiffs = get_geospatial_data(self.lc_asset, points, local_only=self.local_only)
        if self.geotiffs is not None:
            logger.info(f"Loading geotiffs: {self.geotiffs}")

            self._data = load_geotiffs(self.geotiffs, "dem", "EPSG:4269")

        if self.lcgeotiffs is not None:
            logger.info(f"Loading land cover geotiffs: {self.lcgeotiffs}")
            min_lat = np.rad2deg(self.data.lat.min())
            max_lat = np.rad2deg(self.data.lat.max())
            min_lon = np.rad2deg(self.data.lon.min())
            max_lon = np.rad2deg(self.data.lon.max())
            self._nlcd = load_geotiffs(
                self.lcgeotiffs,
                "lc",
                "EPSG:4326",
                [(min_lon, max_lon), (min_lat, max_lat)],
            )
            # Flip y to be increasing
            self._nlcd = self._nlcd.isel(lat=slice(None, None, -1))

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


# Initialize DEMs
DEM = _DEM()
