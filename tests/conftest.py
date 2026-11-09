# tests/conftest.py
import matplotlib
import pytest

from hics.geo.config import DEM_SETTINGS
from hics.geo.dem import DEM
from hics.geo.downloader import GEOTIFF_INDEX

matplotlib.use("Agg")


@pytest.fixture(autouse=True, scope="module")
def restore_dem_globals():
    """Restore all mutated DEM globals after every test in the suite."""
    orig_folder = DEM_SETTINGS.DEM_FOLDER
    orig_index = GEOTIFF_INDEX.cache_dir
    orig_dem_asset = DEM.dem_asset
    orig_lc_asset = DEM.lc_asset
    orig_dem_cache = DEM._dem  # restore cached data too
    orig_nlcd_cache = DEM._nlcd
    yield
    DEM_SETTINGS.DEM_FOLDER = orig_folder
    GEOTIFF_INDEX.cache_dir = orig_index
    DEM.dem_asset = orig_dem_asset
    DEM.lc_asset = orig_lc_asset
    DEM._dem = orig_dem_cache
    DEM._nlcd = orig_nlcd_cache
