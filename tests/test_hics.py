"""Top-level tests for hics module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation

from hics import GLOBAL_CS, HCS, ureg


def test_tuple():
    tup = (2, 0, 0) * ureg.meter
    cs = HCS(tup)
    np.testing.assert_equal(cs.origin.values, np.array(tup.magnitude))


def test_translation() -> None:
    origins = np.array([[2, 1, 0], [3, -7, 4]])
    cs0 = HCS(origins[0, :] * ureg.meter)
    cs1 = HCS(origins[1, :] * ureg.meter, reference=cs0)
    np.testing.assert_equal(cs1.global_position.data.magnitude, origins.sum(axis=0))


def test_relative_translation():
    origins = np.array([[2, 1, 0], [3, -7, 4]])
    cs0 = HCS(origins[0, :] * ureg.meter)
    cs1 = HCS(origins[1, :] * ureg.meter, reference=cs0)
    np.testing.assert_equal(
        cs1.relative_position(cs0).data.magnitude, origins[0, :] - origins.sum(axis=0)
    )


def test_compound_rotation_global_position():
    roof = HCS((0, 0, 5) * ureg.meter)

    tower = HCS(
        (0, 4, 5) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True),
        reference=roof,
    )

    local_ant = HCS(
        (0, 6, 5) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True),
        reference=tower,
    )

    np.testing.assert_array_almost_equal(
        local_ant.global_position.data.magnitude,
        np.array([-5.0, 10.0, 10.0]),
    )


def test_compound_rotation_mixed_global_position():
    roof = HCS(((0, 0, 5) * ureg.meter).to("inches"))
    tower = HCS(
        ((0, 4, 5) * ureg.meter).to("km"),
        rotation=Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True),
        reference=roof,
    )

    local_ant = HCS(
        (0, 6, 5) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True),
        reference=tower,
    )

    np.testing.assert_array_almost_equal(
        local_ant.global_position.data.magnitude,
        np.array([-5.0, 10.0, 10.0]),
    )


def test_compound_rotation_relative_position():
    roof = HCS((0, 0, 5) * ureg.meter)

    tower = HCS(
        (0, 4, 5) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True),
        reference=roof,
    )

    local_ant = HCS(
        (0, 6, 5) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True),
        reference=tower,
    )

    np.testing.assert_array_almost_equal(
        local_ant.relative_position(tower).data.magnitude,
        np.array([6.0, 0, -5.0]),
    )


def test_bad_tuple_input():
    pytest.raises(ValueError, HCS, [(1, 2, 3, 4)])


def test_bad_array_input():
    pytest.raises(ValueError, HCS, [np.array((1, 2, 3, 4))])


def test_relative_distance():
    roof = HCS((0, 0, 5) * ureg.meter)

    tower = HCS(
        (0, 4, 5) * ureg.meter,
        rotation=Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True),
        reference=roof,
    )

    local_ant = HCS(
        ((0, 6, 5) * ureg.meter).to("ft"),
        rotation=Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True),
        reference=tower,
    )
    reldist = local_ant.relative_distance(tower)
    truthdist = (np.linalg.norm(np.array([6.0, 0, -5.0])) * ureg.meter).to("ft")
    np.testing.assert_almost_equal(reldist.data.magnitude, truthdist.magnitude)


def test_interp():
    ts = pd.date_range("1970-01-01T00:00:00", "1970-01-01T00:00:10", periods=10000)

    dist = np.linspace(0, 10, 10000)
    pos = [(0, 0, i) for i in dist] * ureg.meter
    pos = xr.DataArray(
        pos, dims=("time", "position"), coords=dict(time=ts, position=["x", "y", "z"])
    )
    rots = [Rotation.from_euler("ZYZ", [d, d, 0], degrees=True) for d in np.linspace(0, 360, 10000)]
    rots = xr.DataArray(rots, dims=("time",), coords=dict(time=ts))

    cstruth = HCS(pos, rots)
    tssub = pd.date_range("1970-01-01T00:00:00", "1970-01-01T00:00:10", periods=100)

    dist = np.linspace(0, 10, 100)
    pos = [(0, 0, i) for i in dist] * ureg.meter
    pos = xr.DataArray(
        pos, dims=("time", "position"), coords=dict(time=tssub, position=["x", "y", "z"])
    )
    rots = [Rotation.from_euler("ZYZ", [d, d, 0], degrees=True) for d in np.linspace(0, 360, 100)]
    rots = xr.DataArray(rots, dims=("time",), coords=dict(time=tssub))

    cstest = HCS(pos, rots)

    # print(cstest.origin.interp(time=ts).basemag)
    # # print(cstest.rotation.basemag.time)
    # print(cstest.rotation.interp(time=ts).basemag)
    cstest_interp = cstest.interp(time=ts)
    xr.testing.assert_allclose(cstest_interp.global_position, cstruth.global_position)


def test_common():
    roof = HCS((0, 0, 5) * ureg.meter)
    rot0 = Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True)
    tower = HCS((0, 4, 5) * ureg.meter, rotation=rot0, reference=roof)
    rot1 = Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True)
    local_ant = HCS((0, 6, 5) * ureg.meter, rotation=rot1, reference=tower)

    common = local_ant.find_common_cs(tower)
    assert common == tower


def test_common_rev():
    roof = HCS((0, 0, 5) * ureg.meter)
    rot0 = Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True)
    tower = HCS((0, 4, 5) * ureg.meter, rotation=rot0, reference=roof)
    rot1 = Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True)
    local_ant = HCS((0, 6, 5) * ureg.meter, rotation=rot1, reference=tower)

    common = tower.find_common_cs(local_ant)
    assert common == tower


def test_relative_rotation():
    roof = HCS((0, 0, 5) * ureg.meter)
    rot0 = Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True)
    tower = HCS((0, 4, 5) * ureg.meter, rotation=rot0, reference=roof)
    rot1 = Rotation.from_euler("ZYZ", (90, 0, 0), degrees=True)
    local_ant = HCS((0, 6, 5) * ureg.meter, rotation=rot1, reference=tower)

    xr.testing.assert_equal(
        local_ant.get_relative_rotation(tower).basemag, local_ant.rotation.basemag
    )


def test_relative_mismatch():
    nsamples = 1000
    speed = 100.0
    duration = 1
    h = 1000
    # Default movement to y
    end_pos = speed * duration
    ypos = np.linspace(0, end_pos, nsamples)
    zpos = np.full_like(ypos, h)
    xpos = np.zeros_like(ypos)

    start = pd.Timestamp("2025-11-17 00:00:00")
    end = start + pd.Timedelta(days=0, hours=0, minutes=0, seconds=1)

    ts = np.linspace(start.value, end.value, nsamples)
    isotime = np.asarray(pd.to_datetime(ts))
    pos = np.array([xpos, ypos, zpos]).T * ureg.m
    pos_xr = xr.DataArray(
        pos,
        dims=("time", "position"),
        coords=dict(
            time=isotime,
            position=["x", "y", "z"],
        ),
    )

    # Create coordinate system
    cs = HCS(
        pos_xr,
        reference=GLOBAL_CS,
    )
    target_pos = (800, 0, 5) * ureg.m
    target = HCS(target_pos, reference=GLOBAL_CS)

    rel_pos = cs.relative_position(target)
    truth_rel_pos = target.origin - pos_xr
    truth_rel_pos.data = truth_rel_pos.data * ureg.m
    truth_rel_pos = truth_rel_pos.transpose(*rel_pos.dims)
    xr.testing.assert_equal(truth_rel_pos, rel_pos)


def test_relative_multi_mismatch():
    nsamples = 1000
    speed = 100.0
    duration = 1
    h = 1000
    # Default movement to y
    end_pos = speed * duration
    ypos = np.linspace(0, end_pos, nsamples)
    zpos = np.full_like(ypos, h)
    xpos = np.zeros_like(ypos)

    start = pd.Timestamp("2025-11-17 00:00:00")
    end = start + pd.Timedelta(days=0, hours=0, minutes=0, seconds=1)

    ts = np.linspace(start.value, end.value, nsamples)
    isotime = np.asarray(pd.to_datetime(ts))
    pos = np.array([xpos, ypos, zpos]).T * ureg.m
    pos_xr = xr.DataArray(
        pos,
        dims=("time", "position"),
        coords=dict(
            time=isotime,
            position=["x", "y", "z"],
        ),
    )

    # Create coordinate system
    cs = HCS(
        pos_xr,
        reference=GLOBAL_CS,
    )
    txpos = np.linspace(500, 1300, 21)
    xrpos = np.array([[txp, 0, 0] for txp in txpos]) * ureg.m
    xrpos = xr.DataArray(
        xrpos,
        dims=("target", "position"),
        coords=(dict(target=range(xrpos.shape[0]), position=["x", "y", "z"])),
    )
    targets = HCS(xrpos, reference=GLOBAL_CS)

    rel_pos = cs.relative_position(targets)
    truth_rel_pos = xrpos - pos_xr
    truth_rel_pos.data = truth_rel_pos.data * ureg.m
    truth_rel_pos = truth_rel_pos.transpose(*rel_pos.dims)
    xr.testing.assert_equal(truth_rel_pos, rel_pos)
