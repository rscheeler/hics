"""hics module plotting tests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation

from hics import GLOBAL_CS, HCS, ureg
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
cs_boulder = HCS.from_crs((40.015 * ureg.degree, -105.270556 * ureg.degree, 20 * ureg.m), hagl=True)


def test_view_latlon():
    plot = view_latlon(racetrack)


def test_viewcs():
    plot = viewcs(racetrack)


def test_viewcs_ref():
    plot = viewcs(racetrack, reference_cs=cs_boulder)


def test_viewcs_ref_animate():
    plot = viewcs(racetrack, reference_cs=cs_boulder, animate=True, backend="matplotlib")


def test_view_surface_profile():
    plt.figure()
    plot = view_surface_profile(racetrack, cs_boulder, time=0)


def test_viewcs_pyvista():
    plot = viewcs_pyvista(racetrack)
    return plot


def test_viewvcs_pyvist_ref():
    plot = viewcs_pyvista(racetrack, reference_cs=cs_boulder)
    return plot


def test_viewcs_vec_units():
    plot = viewcs(racetrack, reference_cs=cs_boulder, vector_length=1000, units="km")


# def test_plotnlcd():
#     plotnlcd(racetrack_pnts)


def test_showcs_leafmap():
    showcs_leafmap(racetrack)


if __name__ == "__main__":
    test_view_latlon()

    test_viewcs()

    test_viewcs_ref()

    test_viewcs_ref_animate()

    test_view_surface_profile()

    # plot = test_viewcs_pyvista()
    # plot.show()
    # plot = test_viewvcs_pyvist_ref()
    # plot.show()
    # test_viewcs_vec_units()

    # test_plotnlcd()
    test_showcs_leafmap()
    plt.show()
    a = [1, 2, 3]
    print(a[:2])
