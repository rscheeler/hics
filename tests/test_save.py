"""hics saving."""

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hics import HCS, ureg

DIR = Path(__file__).parent
TEST_DATA = DIR / "testdata"


def test_save_read():
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
    svfile = DIR / "testsaveread.hics"
    local_ant.save(svfile)
    with open(svfile, "rb") as file:
        # Serialize the data and write it to the file
        local_ant = pickle.load(file)

    # Delete file
    svfile.unlink()

    np.testing.assert_array_almost_equal(
        local_ant.relative_position(tower).data.magnitude,
        np.array([6.0, 0, -5.0]),
    )
