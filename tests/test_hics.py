"""
test_hics
----------------------------------

Tests for `hics` module.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hics import HCS, ureg


def test_tuple():
    tup = (2, 0, 0) * ureg.meter
    cs = HCS(tup)
    np.testing.assert_equal(cs.origin, np.array(tup))


def test_translation() -> None:
    origins = np.array([[2, 1, 0], [3, -7, 4]])
    cs0 = HCS(origins[0, :] * ureg.meter)
    cs1 = HCS(origins[1, :] * ureg.meter, reference=cs0)
    np.testing.assert_equal(cs1.global_position, origins.sum(axis=0))


def test_relative_translation():
    origins = np.array([[2, 1, 0], [3, -7, 4]])
    cs0 = HCS(origins[0, :] * ureg.meter)
    cs1 = HCS(origins[1, :] * ureg.meter, reference=cs0)
    np.testing.assert_equal(cs1.relative_position(cs0), origins[0, :] - origins.sum(axis=0))


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
        local_ant.global_position,
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
        local_ant.global_position,
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
        local_ant.relative_position(tower),
        np.array([6.0, 0, -5.0]),
    )


def test_bad_tuple_input():
    pytest.raises(ValueError, HCS, [(1, 2, 3, 4)])


def test_bad_array_input():
    pytest.raises(ValueError, HCS, [np.array((1, 2, 3, 4))])
