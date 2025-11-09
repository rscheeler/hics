"""
test_regression
----------------------------------

Regression Tests for `hics` module.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

from hics import HCS, ureg

DIR = Path(__file__).parent
TEST_DATA = DIR / "testdata"


def test_compound_simple_format_regression():
    cs0 = HCS(
        [(0, 0, i) * ureg.meter for i in np.linspace(0, 10, 10000)],
        rotation=[
            Rotation.from_euler("ZYZ", [d, d, 0], degrees=True) for d in np.linspace(0, 360, 10000)
        ],
    )
    cs1 = HCS(
        (2, 0, 0) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", [0, 40, 35], degrees=True),
        reference=cs0,
    )
    truth = xr.load_dataarray(TEST_DATA / "rotatedhierarch.nc")
    res = cs1.global_position.data.to_base_units().magnitude
    np.testing.assert_almost_equal(res, truth.data, decimal=12)


def test_compound_xr_regression():
    ts = np.linspace(0, 10, 10000)
    pos = [(0, 0, i) for i in ts] * ureg.meter
    pos = xr.DataArray(
        pos, dims=("time", "position"), coords=dict(time=ts, position=["x", "y", "z"])
    )
    rots = [Rotation.from_euler("ZYZ", [d, d, 0], degrees=True) for d in np.linspace(0, 360, 10000)]
    rots = xr.DataArray(rots, dims=("time",), coords=dict(time=ts))

    cs0 = HCS(pos, rots)
    cs1 = HCS(
        (2, 0, 0) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", [0, 40, 35], degrees=True),
        reference=cs0,
    )

    truth = xr.load_dataarray(TEST_DATA / "rotatedhierarch.nc")
    res = cs1.global_position
    res.data = res.data.to_base_units().magnitude
    xr.testing.assert_allclose(res, truth)
