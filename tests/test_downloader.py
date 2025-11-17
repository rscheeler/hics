"""HICS geo downloader tests."""

import shutil
from pathlib import Path

import xarray as xr

from hics import HCS, ureg
from hics.datatypes import _POSITION_COORD_DICT, _POSITION_DIM
from hics.geo.config import DEM_SETTINGS
from hics.geo.dem import DEM
from hics.geo.downloader import DEM_CATALOG, GEOTIFF_INDEX

DIR = Path(__file__).parent
TEST_DATA = DIR / "_tempdownload"


def delete_dir(del_dir: Path):
    # Check if the directory exists before attempting to delete
    if del_dir.exists() and del_dir.is_dir():
        try:
            shutil.rmtree(del_dir)
            print(f"Directory '{del_dir}' and its contents deleted successfully.")
        except OSError as e:
            print(f"Error deleting directory '{del_dir}': {e}")
    else:
        print(f"Directory '{del_dir}' does not exist or is not a directory.")


def test_download():
    # Remove Directory
    delete_dir(TEST_DATA)
    # Make directory
    TEST_DATA.mkdir(exist_ok=True)

    DEM_SETTINGS.DEM_FOLDER = TEST_DATA
    GEOTIFF_INDEX.cache_dir = TEST_DATA

    truth = xr.DataArray(
        [-1288677.85726808, -4720141.82855096, 4080318.25299769] * ureg.meter,
        dims=[_POSITION_DIM],
        coords=_POSITION_COORD_DICT,
    )
    # Use a course dataset
    DEM.geo_asset = DEM_CATALOG.COP90
    cs_boulder = HCS.from_crs(
        (40.015 * ureg.degree, -105.270556 * ureg.degree, 20 * ureg.m), hagl=True
    )
    xr.testing.assert_allclose(cs_boulder.global_position, truth)
