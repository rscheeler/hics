"""
Module for using digital elevation model (DEM) data
Automatically downloading terrain data from the national map.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import nest_asyncio
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject
from rasterio.enums import Resampling
from rioxarray import open_rasterio
from xrench.units import ureg
from xrench.xrutils import kw2da

from .config import DEM_SETTINGS
from .downloader import DEM_CATALOG, BoundingBox, GeoAsset, get_geospatial_data
from .setup_assets import generate_rf_csv
from .transforms import XRCRSTransformer

# For linting
if TYPE_CHECKING:
    import xarray as xr
    from rasterio.enums import Resampling


# Utility function to generate a VRT file
def _generate_vrt(
    vrt_path: Path,
    file_paths: list[Path],
    dst_crs: str | None = None,
    resampling: str = "near",
    virtual_warp: bool = False,
    bbox: BoundingBox | None = None,
    pad: float = 0.01,
) -> None:
    if not file_paths:
        raise ValueError("Cannot build VRT with an empty list of files.")

    from osgeo import gdal, osr

    logger.debug(f"If required warp will be virtual: {virtual_warp}")
    # Check whether all inputs already share a common CRS.
    srs_set = set()
    for p in file_paths:
        ds = gdal.Open(str(p))
        srs_set.add(ds.GetProjection())
        ds = None

    if len(srs_set) <= 1:
        # Fast path: all tiles share a CRS, straight mosaic is safe.
        input_files = [str(p) for p in file_paths]
        ds = gdal.BuildVRT(str(vrt_path), input_files)
        if ds is None:
            raise RuntimeError("GDAL BuildVRT failed.")
        ds = None
        logger.debug(f"VRT created (no reprojection needed): {vrt_path}")
        return

    # Slow path: tiles disagree on CRS (e.g. straddling a UTM zone
    # boundary). Instruct GDAL to handle the reprojection dynamically
    # on-the-fly inside virtual memory, restricting the warping calculation
    # exclusively to the chunks you are reading.
    logger.debug(f"Mixed CRS across {len(file_paths)} tiles, warping mismatched tiles")
    target_crs = gdal.Open(str(file_paths[0])).GetProjection()
    if dst_crs:
        srs = osr.SpatialReference()
        srs.SetFromUserInput(dst_crs)
        target_crs = srs.ExportToWkt()

    resampling_map = {
        "near": gdal.GRA_NearestNeighbour,
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
    }
    gdal_resampling = resampling_map.get(resampling.lower(), gdal.GRA_NearestNeighbour)
    if virtual_warp:
        logger.debug("Virtual warp")
        aligned_datasets = []  # Keep a list of open datasets instead of strings
        for p in file_paths:
            ds = gdal.Open(str(p))
            dsproj = ds.GetProjection()
            logger.debug(f"{p.name}: {dsproj}")
            if dsproj == target_crs:
                logger.debug(f"CRS match {dsproj}=={target_crs}")
                aligned_datasets.append(ds)  # Keep it open
                continue
            logger.debug(f"CRS mismatch, warping to {target_crs}")
            # Create the virtual warped view
            vrt_ds = gdal.AutoCreateWarpedVRT(ds, None, target_crs, gdal_resampling)
            aligned_datasets.append(vrt_ds)
            # Note: Do NOT set ds = None here yet, or it will close the underlying vrt_ds dependency
            # Pass the dataset objects directly into BuildVRT
        ds = gdal.BuildVRT(str(vrt_path), aligned_datasets)
    else:
        logger.debug("Warping")
        warp_cache_dir = vrt_path.parent / ".warped"
        warp_cache_dir.mkdir(exist_ok=True)

        warped_paths = []
        for p in file_paths:
            ds = gdal.Open(str(p))
            if ds.GetProjection() == target_crs:
                warped_paths.append(str(p))
                continue
            out_path = warp_cache_dir / f"{p.stem}_warped.tif"
            if not out_path.exists():
                gdal.Warp(
                    str(out_path),
                    ds,
                    dstSRS=target_crs,
                    resampleAlg=gdal_resampling,
                    outputBounds=(
                        bbox.min_lon - pad,
                        bbox.min_lat - pad,
                        bbox.max_lon + pad,
                        bbox.max_lat + pad,
                    ),
                    outputBoundsSRS=dst_crs,
                )
            warped_paths.append(str(out_path))

        ds = gdal.BuildVRT(str(vrt_path), warped_paths)

    if ds is None:
        raise RuntimeError("GDAL BuildVRT failed after warping mismatched tiles.")

    # Clean up and close all handles explicitly now that the file is built
    ds = None
    aligned_datasets = None
    if virtual_warp:
        msg = f"VRT created (with lazy virtual warp): {vrt_path}"
    else:
        msg = f"VRT created (with cached per-tile warp): {vrt_path}"
    logger.debug(msg)


def load_geotiffs(
    geotiffs: list,
    vrt_suffix: str,
    dst_crs: str,
    bbox: BoundingBox,
    resampling: Resampling = Resampling.nearest,
    virtual_warp: bool = False,
) -> xr.DataArray:
    """
    Create VRT and load, reproject to dst_crs, and clip if needed
    Results in a lazy loaded DataArray.

    Parameters
    ----------
    geotiffs : list
        List of geotiff paths
    vrt_suffix : str
        Suffix for VRT file
    dst_crs : str
        Destination coordinate system
    bbox : BoundingBox
        Bounding box for clipping in dst_crs
    resampling : Resampling | None, optional
        Method for resampling, by default nearest


    Returns:
    -------
    data : xr.DataArray
        Lazy loaded, clipped, and re-projected Geotiff data
    """
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
        _generate_vrt(
            vrt_file,
            geotiffs,
            dst_crs=dst_crs,
            resampling=resampling.name,
            virtual_warp=virtual_warp,
            bbox=bbox,
        )
        data = open_rasterio(vrt_file, masked=True, chunks=chunks_dict)
    except ImportError:
        from rioxarray.merge import merge_arrays

        logger.debug("Merging geotiffs...")
        # Open each file
        data_arrays = [open_rasterio(f, chunks=chunks_dict) for f in geotiffs]
        # Merge
        data = merge_arrays(data_arrays)

    _PAD = 0.01
    # Clip and Re-project if needed
    if data.rio.crs != dst_crs:
        logger.debug(f"Reprojecting clipped area to {dst_crs} using {resampling.name}")
        # Get resolution, assume if greater than 0.1 it's in meters and convert to degrees
        res = abs(data.rio.resolution()[0])
        if res > 0.1:
            # Assume in meters and convert to degrees
            res /= 111139.0
        logger.debug(f"{vrt_file} res: {res}")
        dst_geobox = GeoBox.from_bbox(
            (bbox.min_lon - _PAD, bbox.min_lat - _PAD, bbox.max_lon + _PAD, bbox.max_lat + _PAD),
            crs=dst_crs,
            resolution=res,
            tight=False,
        )

        data = xr_reproject(
            data,
            how=dst_geobox,
            resampling=resampling.name,
        )
        # Rename dimensions
        data = data.rename({"longitude": "lon", "latitude": "lat"})

    else:
        data = data.rio.clip_box(
            minx=bbox.min_lon - _PAD,
            miny=bbox.min_lat - _PAD,
            maxx=bbox.max_lon + _PAD,
            maxy=bbox.max_lat + _PAD,
            crs=dst_crs,
        )
        # Rename dimensions
        data = data.rename({"x": "lon", "y": "lat"})
    data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    # Drop dims
    data = data.squeeze("band", drop=True)

    # Convert to base units
    # Change degrees to radians
    data = data.assign_coords(lat=np.deg2rad(data.lat))
    data = data.assign_coords(lon=np.deg2rad(data.lon))

    # Ensure ascending data for slice
    data = data.sortby("lat", ascending=True)
    data = data.sortby("lon", ascending=True)

    return data


def _extract_bbox(lat: xr.DataArray, lon: xr.DataArray) -> list[tuple[float, float]]:
    """Extracts the extreme outer corners of lat/lon DataArrays."""
    # Extract raw numpy arrays
    lapts = lat.data
    lopts = lon.data

    # Strip xrench/pint units if present
    if hasattr(lapts, "magnitude"):
        lapts = lapts.to("radian").magnitude
    if hasattr(lopts, "magnitude"):
        lopts = lopts.to("radian").magnitude

    # Convert radians to degrees instantly via C-speed vector operations
    lat_deg = np.rad2deg(lapts)
    lon_deg = np.rad2deg(lopts)

    # Get the absolute outer boundaries of the current request slice
    min_lat, max_lat = float(lat_deg.min()), float(lat_deg.max())
    min_lon, max_lon = float(lon_deg.min()), float(lon_deg.max())

    # Return the two anchor points that define the entire spatial envelope
    return [(min_lat, min_lon), (max_lat, max_lon)]


class _Terrain:
    """Container class for terrain data.
    Contains both digital elevation model (DEM) and land cover data.
    """

    def __init__(
        self,
        dem_asset: GeoAsset = DEM_CATALOG.USGS30,
        lc_asset: GeoAsset = DEM_CATALOG.LULCv02,
        local_only: bool = False,
        virtual_warp: bool = False,
    ) -> None:
        # Initialize internal data caches to None for lazy loading
        self._dem = None
        self._nlcd = None
        self._dem_asset = None
        self._lc_asset = None
        self.local_only = local_only
        self.dem_asset = dem_asset
        self.lc_asset = lc_asset
        self.virtual_warp = virtual_warp
        # Lazy-loaded legend attributes
        self._nlcd_leg = None
        self._lookup_tables = {}

    def _ensure_legend_loaded(self) -> None:
        """Lazy-loads the NLCD legend and builds lookup tables only when needed."""
        if self._nlcd_leg is not None:
            return

        leg_file = DEM_SETTINGS.NLCDLEG_FOLDER / f"{self.lc_asset.collection}.csv"
        if not leg_file.exists():
            logger.debug(f"Generating legend file: {leg_file}")
            generate_rf_csv(leg_file.stem)

        logger.debug("Loading NLCD Legend and building lookup tables...")
        df = pd.read_csv(leg_file)

        # Process colors
        df["RGBA"] = df["RGBA"].apply(eval)
        df["rgbint"] = df["RGBA"].apply(lambda rgba: tuple(r / 255 for r in rgba))

        # Build Lookup Tables (LUTs) based on max NLCD value
        max_val = df["Value"].max()
        lc_h = np.zeros(max_val + 1)
        nlcd_color = [(0, 0, 0, 0)] * (max_val + 1)
        sigma = np.zeros(max_val + 1)
        er = np.zeros(max_val + 1)
        rms_slope = np.zeros(max_val + 1)
        for _, row in df.iterrows():
            idx = int(row["Value"])
            lc_h[idx] = row.get("Land Cover Height (m)", 0.0)
            nlcd_color[idx] = row["rgbint"]
            sigma[idx] = row["Conductivity (S/m)"]
            er[idx] = row["Relative Permittivity"]
            rms_slope[idx] = row["RMS Slope"]

        self._nlcd_leg = df
        self._lookup_tables = {
            "lc_h": lc_h,
            "colors": nlcd_color,
            "sigma": sigma,
            "er": er,
            "rms_slope": rms_slope,
        }

    @property
    def dem_asset(self) -> GeoAsset:
        return self._dem_asset

    @dem_asset.setter
    def dem_asset(self, value: GeoAsset) -> None:
        self._dem_asset = value
        self._dem = None

    @property
    def lc_asset(self) -> GeoAsset:
        return self._lc_asset

    @lc_asset.setter
    def lc_asset(self, value: GeoAsset) -> None:
        self._lc_asset = value
        self._nlcd = None
        self._nlcd_leg = None
        self._lookup_tables = {}

    @property
    def nlcd_legend(self) -> pd.DataFrame:
        self._ensure_legend_loaded()
        return self._nlcd_leg

    @property
    def idx_lc_h(self) -> list[float]:
        self._ensure_legend_loaded()
        return self._lookup_tables["lc_h"]

    @property
    def idx_colors(self) -> list[tuple[float]]:
        self._ensure_legend_loaded()
        return self._lookup_tables["colors"]

    @property
    def idx_er(self) -> list[float]:
        self._ensure_legend_loaded()
        return self._lookup_tables["er"]

    @property
    def idx_sigma(self) -> list[float]:
        self._ensure_legend_loaded()
        return self._lookup_tables["sigma"]

    @property
    def idx_rms_slope(self) -> list[float]:
        self._ensure_legend_loaded()
        return self._lookup_tables["rms_slope"]

    def nlcdcat2landcoverh(self, v: int) -> float:
        self._ensure_legend_loaded()
        return self.idx_lc_h[int(v)]

    def nlcdcat2color(self, v: int) -> tuple[float]:
        return self.idx_colors[int(v)]

    @property
    def dem(self) -> xr.DataArray:
        """DEM data as xr.DataArray."""
        if self._dem is None:
            raise ValueError(
                "DEM data not loaded. Call load_dem() first, or use an interpolation method."
            )
        return self._dem

    @property
    def nlcd(self) -> xr.DataArray:
        """Land cover data."""
        if self._nlcd is None:
            raise ValueError(
                "NLCD data not loaded. Call load_nlcd() first, or use an interpolation method."
            )
        return self._nlcd

    def landcoverh(self, nlcd: xr.DataArray) -> xr.DataArray:
        """
        Get land cover height from land cover mapping.

        Parameters
        ----------
        nlcd : xr.DataArray
            Land cover data from self.interp_nlcd

        Returns:
        -------
        Land cover height from the legend file: xr.DataArray
        """
        lc_h = xr.full_like(nlcd, 1.0)
        lc_h.data = self.idx_lc_h[nlcd.values.astype(int)]
        return lc_h

    def _loadgeotiff(self, points: list, asset: GeoAsset, vrt_suffix: str) -> None:
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
        # Allow the loop to be nested - needed for compatibility with Jupyter
        nest_asyncio.apply()
        # Get the existing loop
        loop = asyncio.get_event_loop()

        geotiffs = loop.run_until_complete(
            get_geospatial_data(asset, points, local_only=self.local_only)
        )

        bbox = BoundingBox(points)
        # Load geotiff data
        if geotiffs is not None:
            logger.debug(f"Loading {vrt_suffix} geotiffs: {geotiffs}")
            data = load_geotiffs(
                geotiffs, vrt_suffix, "EPSG:4326", bbox, virtual_warp=self.virtual_warp
            )
        else:
            data = None
        return data

    def load_dem(self, points: list) -> None:
        self._dem = self._loadgeotiff(points, self.dem_asset, "dem")

    def load_lc(self, points: list) -> None:
        self._nlcd = self._loadgeotiff(points, self.lc_asset, "lc")

    def load(self, points: list) -> None:
        self.load_dem(points)
        self.load_lc(points)

    def interp(
        self, lat: xr.DataArray | None = None, lon: xr.DataArray | None = None, **kwargs: Any
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
            logger.debug("Trying to interp dem...")
            interp_dat = self.dem.interp(
                lat=lats, lon=lons, **kwargs, kwargs=dict(bounds_error=True)
            ).compute()
        except ValueError:
            logger.debug("Fail to interp loading dem data...")
            # Catch the error and load in data based off of points
            points = _extract_bbox(lats, lons)

            # Load geotiff
            self.load_dem(points)

            # Interpolate
            interp_dat = self.dem.interp(lat=lats, lon=lons, **kwargs)

        return interp_dat

    def interp_nlcd(
        self,
        lat: xr.DataArray | None = None,
        lon: xr.DataArray | None = None,
        **kwargs: Any,
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
            logger.debug("Trying to interp lc...")
            interp_dat = self.nlcd.interp(
                lat=lats,
                lon=lons,
                method="nearest",
                **kwargs,
                kwargs=dict(bounds_error=True, fill_value=0),
            ).compute()

        except ValueError:
            logger.debug("Fail to interp loading lc data...")
            # Catch the error and load in data based off of points
            points = _extract_bbox(lats, lons)

            # Load NLCD
            self.load_lc(points)

            # Interpolate
            interp_dat = self.nlcd.interp(
                lat=lats, lon=lons, method="nearest", **kwargs, kwargs=dict(fill_value=0)
            )

        return interp_dat

    def interp_landcover(
        self,
        lat: xr.DataArray | None = None,
        lon: xr.DataArray | None = None,
        lc_skip_ind: int | None = None,
        **kwargs: Any,
    ) -> tuple[xr.DataArray]:
        nlcd = self.interp_nlcd(lat=lat, lon=lon, **kwargs)
        lch = self.landcoverh(nlcd)
        # Mask points
        if lc_skip_ind is not None:
            lch.loc[dict(distance=lch.distance[:lc_skip_ind])] = 0
            lch.loc[dict(distance=lch.distance[-lc_skip_ind:])] = 0
        return lch, nlcd


# Initialize DEMs
DEM = _Terrain()


def hagl2amsl(*coords: float | xr.DataArray) -> float | xr.DataArray:
    """
    Convert height above ground level (HAGL) to height above mean sea level (AMSL).
    Coords are lat:radian, lon:radian, hagl:meter.
    """
    ll = kw2da(lat=coords[0], lon=coords[1])

    gndlevel = DEM.interp(lat=ll["lat"], lon=ll["lon"])

    if not hasattr(coords[0], "shape"):
        gndlevel = gndlevel.values.squeeze()
    elif coords[0].shape == ():
        gndlevel = gndlevel.values.squeeze()

    return coords[2] + gndlevel


def amsl2hagl(*coords: float | xr.DataArray) -> float | xr.DataArray:
    """Convert height above mean sea level (AMSL) to height above ground level (HAGL).
    Coords are lat:radian, lon:radian, amsl:meter.
    """
    ll = kw2da(lat=coords[0], lon=coords[1])
    gndlevel = DEM.interp(lat=ll["lat"], lon=ll["lon"])

    if not hasattr(coords[0], "shape"):
        gndlevel = gndlevel.values.squeeze()
    elif coords[0].shape == ():
        gndlevel = gndlevel.values.squeeze()

    return coords[2] - gndlevel


# Inject methods for terrain transforms
class XRCRSTransformer_Terrain(XRCRSTransformer):
    def __init__(self, crs_from, crs_to, **kwargs):
        super().__init__(crs_from, crs_to, **kwargs)
        self._amsl2hagl_func = amsl2hagl
        self._hagl2amsl_func = hagl2amsl


llh2geocent = XRCRSTransformer_Terrain("EPSG:4979", "EPSG:4978")
geocent2llh = XRCRSTransformer_Terrain("EPSG:4978", "EPSG:4979")
