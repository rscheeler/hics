"""hics module geo scenarios."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation

from hics import GLOBAL_CS, HCS, ureg
from hics.geo.scenarios import interp_llpnts2hcs, racetrack_latlon
from hics.plotting import view_latlon, viewcs_pyvista

DIR = Path(__file__).parent
TEST_DATA = DIR / "testdata"


def test_racetrack_regression():
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
    rctrck_gp = racetrack.global_position
    previous = xr.open_dataarray(TEST_DATA / "racetrack_gp.nc")
    rctrck_gp.data = rctrck_gp.data.magnitude

    # return racetrack
    xr.testing.assert_equal(rctrck_gp, previous)


def test_drive_regression():
    llpnts = [
        (40.000319, -105.259693),
        (39.999103, -105.256796),
        (39.987486, -105.236799),
        (39.985917, -105.236257),
        (39.985675, -105.249975),
        (39.987160, -105.251552),
        (39.989555, -105.256043),
        (39.996602, -105.261070),
        (39.997951, -105.260177),
        (40.000053, -105.260151),
        (40.000319, -105.259693),
    ]

    hagl = 2 * ureg.m
    speed = 30 * ureg.mph
    start_time = np.datetime64("2025-11-19 00:00:00")
    drivecs = interp_llpnts2hcs(
        llpnts, hagl, speed, start_time, bank_turns=False, hagl=True, new_n_pnts=51
    )
    drivecs_gp = drivecs.global_position
    previous = xr.open_dataarray(TEST_DATA / "drive_gp.nc")
    drivecs_gp.data = drivecs_gp.data.magnitude
    # return drivecs
    xr.testing.assert_equal(drivecs_gp, previous)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # rctrk = test_racetrack_regression()
    # rctrck_gp = rctrk.global_position
    # rctrck_path = TEST_DATA / "racetrack_gp"
    # rctrck_gp.to_netcdf(Path(rctrck_path).with_suffix(".nc"))
    drcs = test_drive_regression()
    dr_gp = drcs.global_position
    dr_path = TEST_DATA / "drive_gp"
    dr_gp.to_netcdf(Path(dr_path).with_suffix(".nc"))
    # plot = view_latlon(rctrk)
    # plot = view_latlon(drcs)
    # plt.show()
    # plot = viewcs_pyvista(rctrk, animate=True)
    # plot.show_grid()
    # plot.show()
