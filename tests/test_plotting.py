"""hics module plotting tests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation

from hics import GLOBAL_CS, HCS, ureg
from hics.geo.dem import DEM
from hics.geo.downloader import DEM_CATALOG
from hics.geo.scenarios import interp_llpnts2hcs, racetrack_latlon
from hics.plotting import (
    plotnlcd,
    showcs_leafmap,
    view_latlon,
    view_surface_profile,
    viewcs,
    viewcs_pyvista,
)

DIR = Path(__file__).parent
TEST_DATA = DIR / "testdata"
import sys
import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

warnings.simplefilter("always")


@pytest.fixture(scope="function")
def setupcs():
    cs_boulder = HCS.from_crs(
        (40.015 * ureg.degree, -105.270556 * ureg.degree, 20 * ureg.m), hagl=True
    )
    yield cs_boulder


@pytest.fixture(scope="function")
def setupracetrack():
    center = (40.037578, -105.228117) * ureg.degree
    width = 1.5 * ureg.km
    length = 8.48 * ureg.km
    npts = 51
    angle = 90 * ureg.degree
    speed = 100 * ureg.m / ureg.s
    altitude = 10 * ureg.km
    start_time = np.datetime64("2025-11-19 00:00:00")
    racetrack_pnts = racetrack_latlon(*center, length, width, npts, angle)

    racetrack = interp_llpnts2hcs(
        racetrack_pnts, altitude, speed, start_time, bank_turns=True, hagl=False, new_n_pnts=51
    )
    yield racetrack


def test_view_latlon(setupracetrack):
    plot = view_latlon(setupracetrack)


def test_viewcs(setupracetrack):
    plot = viewcs(setupracetrack)


def test_viewcs_ref(setupracetrack, setupcs):
    plot = viewcs(setupracetrack, reference_cs=setupcs)


def test_viewcs_ref_animate(setupracetrack, setupcs):
    plot = viewcs(setupracetrack, reference_cs=setupcs, animate=True, backend="matplotlib")


def test_view_surface_profile(setupracetrack, setupcs):
    plt.figure()
    # DEM.dem_asset = DEM_CATALOG.COP90

    plot = view_surface_profile(setupracetrack, setupcs, time=0)


def test_viewcs_pyvista(setupracetrack):
    plot = viewcs_pyvista(setupracetrack)
    return plot


def test_viewvcs_pyvist_ref(setupracetrack, setupcs):
    plot = viewcs_pyvista(setupracetrack, reference_cs=setupcs)
    return plot


def test_viewcs_vec_units(setupracetrack, setupcs):
    plot = viewcs(setupracetrack, reference_cs=setupcs, vector_length=1000, units="km")


def test_plotnlcd():
    plot = plotnlcd([(40.1, -105.2), (39.941822, -105.315253)])


def test_showcs_leafmap(setupracetrack):
    showcs_leafmap(setupracetrack)


if __name__ == "__main__":
    # test_view_latlon()

    # test_viewcs()

    # test_viewcs_ref()

    # test_viewcs_ref_animate()

    # test_view_surface_profile()

    # plot = test_viewcs_pyvista()
    # plot.show()
    # plot = test_viewvcs_pyvist_ref()
    # plot.show()
    # test_viewcs_vec_units()

    # test_plotnlcd()
    # test_showcs_leafmap()
    # test_plotnlcd()
    plt.show()
    # a = [1, 2, 3]
    # print(a[:2])
