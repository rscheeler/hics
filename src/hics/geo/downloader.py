"""Module for downloading terrain data from the national map."""

import asyncio
import json
import re

# import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, Union
from urllib.request import urlretrieve

import aiofiles
import aiohttp
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
import rioxarray
import xarray as xr
from loguru import logger
from pint import Quantity
from rtree import index
from tqdm import tqdm

from .. import ureg
from ..utils import Singleton
from .config import DEM_SETTINGS

_CHUNK_SIZE = 1024 * 1024  # 1MB chunks


@dataclass
class BoundingBox:
    """Bounding box data holder."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def __post_init__(self):
        # Ensure min/max are correct, though constructor handles it if points are ordered
        self.min_lat = min(self.min_lat, self.max_lat)
        self.max_lat = max(self.min_lat, self.max_lat)
        self.min_lon = min(self.min_lon, self.max_lon)
        self.max_lon = max(self.min_lon, self.max_lon)

    def __init__(self, points: list) -> None:
        """
        Initializes a BoundingBox from a list of (lat, lon) points.

        Implements a minimum buffer (TINY_DELTA) to prevent zero-area
        bounding boxes when only a single point is provided, which often fails
        TNM API spatial queries.
        """
        # points expected as [(lat1, lon1), (lat2, lon2), ...]
        self.min_lat = min(p[0] for p in points)
        self.min_lon = min(p[1] for p in points)
        self.max_lat = max(p[0] for p in points)
        self.max_lon = max(p[1] for p in points)

        # Ensure a non-zero area for robust API queries
        TINY_DELTA = 0.0001
        if self.min_lat == self.max_lat:
            # Shift min/max slightly to create a tiny box
            self.max_lat += TINY_DELTA
            self.min_lat -= TINY_DELTA
        if self.min_lon == self.max_lon:
            self.max_lon += TINY_DELTA
            self.min_lon -= TINY_DELTA

    def tolist(self) -> list:
        """Returns bounding box as list."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

    def __str__(self) -> str:
        """Returns the bounding box string in 'west,south,east,north' format (lon, lat)."""
        return f"{self.min_lon},{self.min_lat},{self.max_lon},{self.max_lat}"

    def __repr__(self) -> str:
        return self.__str__()

    def intersects(self, other_bbox: "BoundingBox") -> bool:
        """Checks if this bounding box intersects with another bounding box."""
        # Check if the rectangles overlap
        return not (
            self.max_lon < other_bbox.min_lon
            or self.min_lon > other_bbox.max_lon
            or self.max_lat < other_bbox.min_lat
            or self.min_lat > other_bbox.max_lat
        )


# Define the structure for the metadata we store in the JSON mapping file
class GeoTIFFMetadata(TypedDict):
    """Metadata payload stored for each file in the index map."""

    source: str  # The STAC collection ID (e.g., '3dep-seamless')
    tile_name: str  # Simplified file identifier (e.g., 'n38w106')
    date: int  # Date integer
    path: str  # Full path to the local file
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    gsd: int  # Ground Sample Distance (resolution in meters)


@dataclass
class GeoAsset:
    """Container for planetary catalog collection information."""

    collection: str  # Collection ID
    key: str  # Asset ID
    description: str  # Description
    gsd: int  # Ground sample distance


@dataclass
class _DEM_CATALOG:
    USGS30: GeoAsset
    USGS10: GeoAsset
    COP30: GeoAsset
    COP90: GeoAsset
    NASA30: GeoAsset


DEM_CATALOG = _DEM_CATALOG(
    USGS30=GeoAsset(
        collection="3dep-seamless", key="data", description="USGS 3DEP Seamless DEMs 30m", gsd=30
    ),
    USGS10=GeoAsset(
        collection="3dep-seamless",
        key="data",
        description="USGS 3DEP Seamless DEMs 10m",
        gsd=10,
    ),
    COP30=GeoAsset(
        collection="cop-dem-glo-30",
        key="data",
        description="Copernicus GLO-30 DEM	",
        gsd=30,
    ),
    COP90=GeoAsset(
        collection="cop-dem-glo-90",
        key="data",
        description="Copernicus GLO-90 DEM	",
        gsd=90,
    ),
    NASA30=GeoAsset(
        collection="nasadem", key="elevation", description="NASA DEM 30m data.", gsd=30
    ),
)
GSD_MISSING_COLLECTIONS = {"nasadem"}


class GeoTIFFIndex(Singleton):
    """
    Manages the spatial index (R-tree) of local GeoTIFF files
    for fast spatial querying and coverage checks.
    """

    def __init__(self, cache_dir: Path):
        """
        Initializes the index object, setting up file paths.

        Parameters
        ----------
        folder : Path
            The root directory containing the GeoTIFF files
        """
        # Store folder
        self.cache_dir = cache_dir
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rtree index file paths
        self.index_base = self.cache_dir / "dem_index"

        # JSON file to map R-tree internal IDs to file metadata
        self.metadata_map_path = self.cache_dir / "dem_index_map.json"

        # Set up the Rtree index properties for 2D data
        self.p = index.Property(dimension=2)
        # Use a standard 2D bounding box
        self.p.bounds = "min_lon, min_lat, max_lon, max_lat"

        # Load the metadata map if it exists, otherwise initialize empty
        self.metadata_map: dict[int, GeoTIFFMetadata] = {}
        if self.metadata_map_path.exists():
            try:
                with open(self.metadata_map_path) as f:
                    # Note: JSON keys are strings, so we must convert keys back to integers
                    data = json.load(f)
                    self.metadata_map = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.error(f"Failed to load existing metadata map: {e}")
                self.metadata_map = {}  # Reset map if load fails

    def _save_metadata_map(self):
        """Saves the current state of the metadata map to JSON."""
        try:
            with open(self.metadata_map_path, "w") as f:
                json.dump(self.metadata_map, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata map: {e}")

    def build_index(self):
        """
        Scans the directory for GeoTIFFs, reads their headers ONCE, and builds/updates the
        persistent spatial index. This is the only slow operation.
        """
        logger.info(f"Starting index build/update for {self.cache_dir}...")

        # Start a new index connection, creating the files if they don't exist
        idx = index.Index(str(self.index_base), properties=self.p)

        # Keep track of existing files to remove deleted ones later
        existing_paths = {v["path"]: (k, tuple(v["bbox"])) for k, v in self.metadata_map.items()}
        current_paths = set()

        next_id = max(self.metadata_map.keys()) + 1 if self.metadata_map else 0

        for fpath in self.cache_dir.glob("*.tif"):
            fpath_res = fpath.resolve()
            current_paths.add(str(fpath_res))

            # Check if file is already indexed (by path)
            is_indexed = any(v["path"] == str(fpath_res) for v in self.metadata_map.values())
            if is_indexed:
                continue  # Skip files already in the index

            try:
                with rasterio.open(fpath_res) as src:  # <-- SLOW OPERATION, only happens here
                    bbox = src.bounds

                    # Extract versioning info (tile_name, date) from your existing logic
                    # TODO: this is wrong
                    name_parts = fpath_res.stem.split("-")
                    # Extracting components based on the new convention:
                    tile_name = "-".join(name_parts[1:-2])
                    # tile_name = name_parts[1]
                    gsd = int(name_parts[-2].replace("m", ""))
                    date_int = int(name_parts[-1])  # <-- EASY DATE EXTRACTION

                    # 1. Insert into R-tree
                    bounds = (bbox.left, bbox.bottom, bbox.right, bbox.top)

                    idx.insert(next_id, bounds)
                    # # 2. Determine the source before insertion
                    file_source = infer_source(fpath)

                    # 3. Store metadata in map
                    self.metadata_map[next_id] = GeoTIFFMetadata(
                        source=file_source,
                        tile_name=tile_name,
                        date=date_int,
                        path=str(fpath_res),
                        bbox=bounds,
                        gsd=gsd,
                    )

                    next_id += 1
                    logger.debug(f"Indexed new file: {fpath_res.name}")

            except rasterio.RasterioIOError:
                logger.warning(f"Skipping file {fpath.name}, not a valid GeoTIFF.")
            except Exception as e:
                logger.error(f"Error processing {fpath.name} for indexing: {e}")

        # Remove deleted files from the index and map
        deleted_paths = set(existing_paths.keys()) - current_paths

        if deleted_paths:
            logger.info(f"Removing {len(deleted_paths)} deleted files from index.")

            # Iterate over the paths that no longer exist on disk
            for deleted_path in deleted_paths:
                # Get the internal R-tree ID and the bounds (bbox) for deletion
                id_to_delete, bounds_to_delete = existing_paths[deleted_path]

                try:
                    # 1. Delete from R-tree
                    idx.delete(id_to_delete, bounds_to_delete)
                    # 2. Delete from metadata map
                    del self.metadata_map[id_to_delete]
                    logger.debug(f"Removed ID {id_to_delete} for path: {deleted_path}")
                except Exception as e:
                    # This can happen if the R-tree files were inconsistent, but we continue
                    logger.error(f"Failed to delete ID {id_to_delete} from R-tree: {e}")

        # Commit changes to disk and save the map
        idx.close()
        self._save_metadata_map()
        logger.info("Index build/update complete.")

    def query(self, bounding_box: "BoundingBox") -> list[GeoTIFFMetadata]:
        """
        Performs a fast spatial query against the index.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box to run query with.

        Returns:
        -------
        list[GeoTIFFMetadata]
            A list of metadata for all intersecting files.
        """
        # Ensure the index is ready to be read
        if not self.index_base.with_suffix(".idx").exists():
            logger.warning("Index files not found.")
            self.build_index()
            # If the build still fails, the query will fail.
            if not self.index_base.with_suffix(".idx").exists():
                logger.error("Index build failed. Cannot perform query.")
                return []

        # Open the index for querying
        idx = index.Index(str(self.index_base), properties=self.p)

        # The bounding box for the query: (min_lon, min_lat, max_lon, max_lat)
        query_bbox = bounding_box.tolist()

        # Query the R-tree. This returns a generator of internal IDs.
        intersecting_ids = list(idx.intersection(query_bbox))

        # Retrieve the metadata from the map for the intersecting IDs
        results = [self.metadata_map[id] for id in intersecting_ids if id in self.metadata_map]

        idx.close()

        logger.info(f"R-tree query found {len(results)} tiles intersecting the AOI.")
        return results


def infer_source(fpath: Path) -> str:
    """
    Infers the data source from the file path or name. If no pattern found defaults
    to splitting by '_'.
    """
    for k, v in asdict(DEM_CATALOG).items():
        if fpath.name.startswith(k):
            return v["collection"]
    return "UNKNOWN"


def infer_source_prefix(geo_asset: GeoAsset) -> str:
    """Infers the short prefix for a filename based on GeoAsset properties."""
    for k, v in asdict(DEM_CATALOG).items():
        if v["collection"] == geo_asset.collection and v["gsd"] == geo_asset.gsd:
            return k
    return "UNKNOWN"


def find_local_geotiffs(bounding_box: BoundingBox, source: GeoAsset) -> list[Path]:
    """Finds local GeoTIFF files by querying a pre-built spatial index."""
    # Query the spatial index
    # This returns paths and dates for ALL intersecting tiles

    intersecting_tiles_data = GEOTIFF_INDEX.query(bounding_box)

    filtered_metadata = [
        meta
        for meta in intersecting_tiles_data
        if (meta["source"] == source.collection and meta["gsd"] == source.gsd)
    ]

    intersecting_files_by_tile = {}

    for item in filtered_metadata:
        tile_name = item["tile_name"]

        if (
            tile_name not in intersecting_files_by_tile
            or item["date"] > intersecting_files_by_tile[tile_name]["date"]
        ):
            # Keep only the newest version of the tile
            intersecting_files_by_tile[tile_name] = item

    # Return the list of the latest file paths
    # return [Path(data["path"]) for data in intersecting_files_by_tile.values()]
    return intersecting_files_by_tile


def check_local_coverage(aoi_bbox: BoundingBox, source: GeoAsset) -> bool:
    """Checks if the union of local tiles fully covers the AOI bounding box."""
    # Query the spatial index
    # This returns paths and dates for ALL intersecting tiles
    intersecting_metadata = GEOTIFF_INDEX.query(aoi_bbox)

    filtered_metadata = [
        meta
        for meta in intersecting_metadata
        if (meta["source"] == source.collection and meta["gsd"] == source.gsd)
    ]

    if not filtered_metadata:
        return False  # No local tiles at all

    # 2. Calculate the union (min/max of all tiles)

    # Extract all four bounds from the 'bbox' stored in the metadata
    min_lons = [m["bbox"][0] for m in filtered_metadata]
    min_lats = [m["bbox"][1] for m in filtered_metadata]
    max_lons = [m["bbox"][2] for m in filtered_metadata]
    max_lats = [m["bbox"][3] for m in filtered_metadata]

    # Find the overall extent (union)
    union_bbox_list = [
        min(min_lons),
        min(min_lats),
        max(max_lons),
        max(max_lats),
    ]

    # 3. Check if the AOI is fully contained within the union

    # AOI's bounds
    aoi_min_lon, aoi_min_lat, aoi_max_lon, aoi_max_lat = aoi_bbox.tolist()

    # Union's bounds
    union_min_lon, union_min_lat, union_max_lon, union_max_lat = union_bbox_list

    is_covered = (
        aoi_min_lon >= union_min_lon
        and aoi_min_lat >= union_min_lat
        and aoi_max_lon <= union_max_lon
        and aoi_max_lat <= union_max_lat
    )

    return is_covered


def _delete_old_files(file_paths: list[Path]):
    """Removes a list of files from disk."""
    if not file_paths:
        return

    logger.info(f"Deleting {len(file_paths)} old tiles...")
    for fpath in file_paths:
        try:
            fpath.unlink(missing_ok=True)
            logger.debug(f"Deleted old tile: {fpath.name}")
        except OSError as e:
            logger.error(f"Failed to delete file {fpath.name}: {e}")


async def _download_file(session: aiohttp.ClientSession, url: str, destination: Path) -> None:
    """
    Asynchronously downloads a file from a given URL to a destination path,
    with a progress bar.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Session
    url : str
        url
    destination : Path
        Destination for download

    Returns:
    -------
    Path
        Downloaded file path
    """
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()

            # Get file size for the progress bar
            total_size_in_bytes = int(resp.headers.get("content-length", 0))

            # Use async context managers for non-blocking file I/O
            async with aiofiles.open(destination, "wb") as f:
                with tqdm(
                    desc=destination.name,
                    total=total_size_in_bytes,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                        await f.write(chunk)
                        bar.update(len(chunk))
        logger.info(f"Download of {destination.name} complete.")

    except TimeoutError:
        logger.error(f"The request for {url} timed out.")
        raise
    except aiohttp.ClientError as e:
        logger.error(f"An aiohttp client error occurred for {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {url}: {e}")
        raise


async def _download_all(urls_dests: dict) -> None:
    """
    Downloads all url/destination pairs in dictionary.

    Parameters
    ----------
    urls_dests : dict
        Dictionary of url/destination pairs url(key):desintation(value)

    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks = []
        for url, destination in urls_dests.items():
            logger.debug(f"Downloading to {destination.resolve()}")
            tasks.append(_download_file(session, url, destination))

        # Use asyncio.gather() to run all tasks concurrently and wait for all to complete.
        # This is often simpler than using as_completed when you want all results.
        await asyncio.gather(*tasks)


def get_geospatial_data(
    geo_asset: GeoAsset,
    points: list[tuple],
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    use_cog_only: bool = False,
    local_only: bool = True,
) -> list[Path | str]:
    """Handles retrieval of geospatial data, prioritizing local cache, then
    downloading, or directly using the COG. Utitlizes Microsoft's Planetary computer data.

    Parameters
    ----------
    geo_asset : GeoAsset
        GeoAsset that contains information about the STAC collection.
    points : list[tuple]
        List of lat lon pairs i.e [(lat0, lon0), (lat1, lon1), ..., (latN, lonN)]
    use_cog_only : bool, optional
        Cloud only, by default False

    Returns:
    -------
    list[Path | str]
        List of Paths or str: A local file path or the COG URL string.
    """
    # Convert points to bounding box
    bbox = BoundingBox(points)

    # Get current local tiles that intersect bounding box and match the geo_asset
    GEOTIFF_INDEX.build_index()  # Build the index first
    logger.info(f"Searching bounding box {bbox}")
    local_data = find_local_geotiffs(bbox, geo_asset)
    results = local_data
    logger.info(f"Found {geo_asset.collection} files {local_data}")

    # Paths of local tiles we start with
    initial_local_paths = list(Path(m["path"]) for m in local_data.values())
    results = initial_local_paths.copy()
    # Check local data covers bbox
    is_covered = check_local_coverage(bbox, geo_asset)

    if not is_covered:
        local_only = False

    if local_only:
        logger.info("Local only mode. Returning currently available local tiles.")
        return results

    # Connect to STAC
    try:
        # Access the Planetary Computer STAC catalog
        catalog = pystac_client.Client.open(
            stac_url, modifier=planetary_computer.sign_inplace, timeout=60
        )
        logger.info(f"Connection to {stac_url} established")
    except Exception as e:
        logger.error(f"An error occurred connecting to STAC: {e}. Returning local files.")
        return results

    try:
        # Create search parameters based on collection, bbox, and gsd.
        search_params = {
            "collections": [geo_asset.collection],
            "bbox": bbox.tolist(),
            "limit": 50,
        }

        # Only apply GSD filter if the collection is NOT known to be missing the property.
        if geo_asset.collection not in GSD_MISSING_COLLECTIONS:
            # Use GSD filter to ensure we only get the requested resolution
            search_params["query"] = {"gsd": {"eq": geo_asset.gsd}}
        logger.debug(f"Search params {search_params}")
        # Search for items within the collection and bounding box
        search = catalog.search(**search_params)
        items = search.item_collection()
        logger.info(f"STAC search found {len(items)} items.")
        if not items:
            logger.info(
                f"No items in '{geo_asset.collection}-{geo_asset.gsd}m' in the specified bbox."
            )
            return results

        # Find the Newest STAC Item for each unique tile_name ---
        remote_tile_groups = {}
        for item in items:
            stac_date = item.properties.get("datetime", "0000-00-00T00:00:00Z")
            date_stamp = stac_date.split("T")[0].replace("-", "")
            date_int = int(date_stamp) if date_stamp.isdigit() else 0

            # Use STAC item properties to derive tile_name and GSD
            item_gsd = item.properties.get("gsd")
            actual_gsd = int(round(item_gsd)) if item_gsd is not None else geo_asset.gsd

            # Tile name
            tile_name = item.id

            remote_tile_groups.setdefault(tile_name, []).append(
                {"item": item, "tile_name": tile_name, "date_int": date_int, "gsd": actual_gsd}
            )

        # Select the single newest item for each tile - might not be necessary
        newest_remote_items = {}
        for tile_name, items in remote_tile_groups.items():
            items.sort(key=itemgetter("date_int"), reverse=True)
            newest_remote_items[tile_name] = items[0]

        # Compare and Plan Actions (Download/Delete)

        files_to_download: dict[str, Path] = {}  # {url: destination_path}
        files_to_delete: list[Path] = []
        final_local_paths: list[Path] = []

        # Determine collection name prefix for filename construction (e.g., 'USGS30')
        collection_name_prefix = infer_source_prefix(geo_asset)

        for tile_name, remote_info in newest_remote_items.items():
            item = remote_info["item"]
            remote_date = remote_info["date_int"]
            actual_gsd = remote_info["gsd"]

            # Destination path for the newest remote item
            date_stamp = str(remote_date).zfill(8)
            destination = (
                GEOTIFF_INDEX.cache_dir
                / f"{collection_name_prefix}-{tile_name}-{actual_gsd}m-{date_stamp}.tif"
            )
            # Get the STAC asset URL
            asset_key = geo_asset.key
            if asset_key not in item.assets:
                logger.warning(
                    f"Tile {tile_name} is missing expected asset key: {asset_key}. Skipping."
                )
                continue

            asset_url = item.assets[asset_key].href

            if use_cog_only:
                # If only COG URLs are requested, collect the URL
                final_local_paths.append(asset_url)
                continue

            local_metadata_item = local_data.get(tile_name)

            if local_metadata_item:
                # LOCAL TILE EXISTS: Check if it is the latest
                local_date = local_metadata_item["date"]
                local_path = Path(local_metadata_item["path"])

                if remote_date > local_date:
                    # Remote tile is NEWER: Delete old and download new
                    logger.info(
                        f"Tile {tile_name}: Newer version found ({remote_date} > {local_date})."
                    )
                    files_to_delete.append(local_path)
                    files_to_download[asset_url] = destination
                else:
                    # Local tile is current or newer: Keep local tile
                    final_local_paths.append(local_path)
            else:
                # NO LOCAL TILE: Download the newest remote tile
                files_to_download[asset_url] = destination

        # Handle use_cog_only return here
        if use_cog_only:
            return final_local_paths

        # --- PHASE C: Execute and Rebuild ---

        # 1. DELETE OLD TILES
        if files_to_delete:
            _delete_old_files(files_to_delete)

        # 2. DOWNLOAD NEW TILES
        downloaded_paths = []
        if files_to_download:
            logger.info(f"Downloading {len(files_to_download)} new/updated files...")
            asyncio.run(_download_all(files_to_download))
            downloaded_paths = [v for v in files_to_download.values()]

        # 3. Rebuild index (Essential to reflect deletions and new files)
        if files_to_delete or downloaded_paths:
            GEOTIFF_INDEX.build_index()

        # Final result is the union of files that were already current AND the newly downloaded ones
        logger.debug(f"{final_local_paths}")
        logger.debug(f"{downloaded_paths}")
        results = final_local_paths + downloaded_paths

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during API processing: {e}. Returning current local files."
        )
        return initial_local_paths

    return results


# Initialize the spatial index manager
GEOTIFF_INDEX = GeoTIFFIndex(DEM_SETTINGS.DEM_FOLDER)
