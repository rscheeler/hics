import numpy as np
import pytest
import xarray as xr

from hics import ureg
from hics.utils import wraps_xr

# --- Mock Functions for Testing ---


@wraps_xr(ureg.m, (ureg.radian,))
def identity_rad_to_m(val):
    """Returns input as-is to inspect what the decorator passed in."""
    return val


@wraps_xr((ureg.m, ureg.s), (ureg.km, ureg.hr))
def multi_unit_func(dist, time):
    """Tests multiple inputs and multiple outputs."""
    return dist, time


@wraps_xr(ureg.m, (ureg.m,))
def list_processor(data_list):
    """Tests recursive list handling."""
    return data_list


# --- Tests ---


def test_input_conversion():
    """Verify that inputs are converted to the target unit magnitude before entering."""
    # Pass 180 degrees, decorator should convert to pi radians
    deg_val = 180 * ureg.degree
    result = identity_rad_to_m(deg_val)

    # Check that the internal function received the magnitude of 180deg in radians
    # result.magnitude is used here because the decorator wraps the return in 'm'
    assert np.isclose(result.magnitude, np.pi)
    assert result.units == ureg.m


def test_dataarray_round_trip():
    """Verify DataArray wrappers and coordinates are preserved."""
    da = xr.DataArray([1.0, 2.0], coords={"time": [0, 1]}, dims="time", name="test") * ureg.km

    # Decorator converts km -> m (1.0 -> 1000.0)
    @wraps_xr(ureg.m, (ureg.m,))
    def simple_add(obj):
        return obj + 10  # 1000 + 10 = 1010

    res = simple_add(da)

    assert isinstance(res, xr.DataArray)
    assert res.data.units == ureg.m
    assert res.data.magnitude[0] == 1010.0
    assert "time" in res.coords
    assert res.name == "test"


def test_list_of_dataarrays():
    """Verify recursive stripping of lists of DataArrays."""
    da1 = xr.DataArray([10]) * ureg.cm
    da2 = xr.DataArray([20]) * ureg.cm

    # Contract: input must be in meters
    @wraps_xr(ureg.m, (ureg.m,))
    def sum_list(items):
        # items should now be [array([0.1]), array([0.2])]
        return items[0] + items[1]

    res = sum_list([da1, da2])
    assert np.isclose(res.data.magnitude, 0.3)
    assert res.data.units == ureg.m


def test_multiple_returns():
    """Verify tuple-to-tuple unit mapping."""
    # Input: 1km, 1hr -> Internal: 1000.0, 3600.0 (if base units)
    # But we specified km and hr in the decorator, so it passes magnitudes as-is
    dist = 1 * ureg.km
    time = 1 * ureg.hr

    d_out, t_out = multi_unit_func(dist, time)

    # Check output units
    assert d_out.units == ureg.m
    assert t_out.units == ureg.s
    # Check magnitudes (1km -> 1m, 1hr -> 1s because internal returns magnitude)
    assert d_out.magnitude == 1.0
    assert t_out.magnitude == 1.0


def test_none_handling():
    """Verify None in arg_units skips conversion for that argument."""

    @wraps_xr(ureg.m, (None, ureg.m))
    def mixed_func(raw, unit_aware):
        return unit_aware  # just return the second one

    # First arg is string, second is km
    res = mixed_func("ignore_me", 1 * ureg.km)
    assert res.magnitude == 1000.0
    assert res.units == ureg.m


@pytest.mark.parametrize("input_type", ["scalar", "list", "numpy", "dataarray"])
def test_input_types(input_type):
    """Verify various container types all strip to magnitudes correctly."""

    @wraps_xr(ureg.m, (ureg.m,))
    def get_mag(val):
        # Use np.ravel() which works on both numpy and xarray
        # Or better yet, use np.asanyarray(val).item() if you expect a scalar
        if isinstance(val, list):
            return val[0]

        # This is the safe way to get the first element regardless of container
        return np.array(val).ravel()[0]

    val_in = 100 * ureg.cm  # = 1.0 meter

    if input_type == "scalar":
        arg = val_in
    elif input_type == "list":
        arg = [val_in]
    elif input_type == "numpy":
        arg = np.array([1.0]) * ureg.m
    elif input_type == "dataarray":
        arg = xr.DataArray([1.0]) * ureg.m

    res = get_mag(arg)
    assert np.isclose(float(res.magnitude), 1.0)
