from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import xarray as xr
from loguru import logger
from xarray import DataArray

from .units import ureg

if TYPE_CHECKING:
    from typing import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")
    # Define types for the units input (can be string, unit object, or sequence)
    UnitLike = Any  # pint.Unit, str, or None
    UnitsIn = Sequence[UnitLike]
    UnitsOut = UnitLike | Sequence[UnitLike]


def kw2da(**kwargs: Any) -> dict[xr.DataArray]:
    """
    Convert kwargs of one-dimensional data into an xarray DataArray object.

    Returns:
    -------
    das : dict
        Dictionary of DataArray
    """
    # Initialize output dict
    out = {}
    # Iterate over kwargs
    for k, v in kwargs.items():
        # Dimensional coordinate data will have quantities removed, convert to base units
        if not isinstance(v, xr.DataArray):
            if isinstance(v, ureg.Quantity):
                v = v.to_base_units()  # noqa: PLW2901
            if not isinstance(v, np.ndarray) or v.shape == ():
                v = [v]  # noqa: PLW2901
            out[k] = DataArray(v, dims=(k,), coords={k: v})
        elif isinstance(v.data, ureg.Quantity):
            v.data = v.data.to_base_units()
            out[k] = v
        else:
            out[k] = v

    return out


class Singleton:
    """
    Singleton class.
    See: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):  # noqa: D102
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


def vector_norm(x: xr.DataArray, dim: str, ord: Any | None = None) -> xr.DataArray:
    """
    Wrapper to perform np.linalg.norm on a xr.DataArray.

    Parameters
    ----------
    x : xr.DataArray
        Array to dermine norm on
    dim : str
        Dimension(s) to calculate norm
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm. Default is None.

    References:
    ----------
    https://www.programcreek.com/python/example/123575/xarray.apply_ufunc
    """
    norm = xr.apply_ufunc(
        np.linalg.norm,
        x,
        input_core_dims=[[dim]],
        kwargs=dict(ord=ord, axis=-1),
    )

    return norm


def compute_if_dask(x):
    """Computes a dask array/quantity, otherwise returns the input."""
    # Check if the object itself is a pint Quantity with dask data
    if hasattr(x, "compute"):
        return x.compute()
    else:
        # It's a numpy array, a pint Quantity with numpy data, etc.
        return x


def basemagxr(*args):
    out = []
    for a in args:
        if not isinstance(a, xr.DataArray):
            if isinstance(a, ureg.Quantity):
                a = a.to_base_units().magnitude  # noqa: PLW2901

        elif isinstance(a.data, ureg.Quantity):
            a.data = a.data.to_base_units().magnitude
        out.append(a)
    return out


def wraps_xr(ret_units: Any, arg_units: Sequence[Any] | None = None) -> Callable:
    """
    Decorator that strips units on entry and adds them back on exit.
    Preserves xarray containers to maintain coordinate/dimension integrity.
    """
    actual_arg_units = arg_units if arg_units is not None else ()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. STRIP: Convert Quantities to Magnitudes inside the containers
            new_args = list(args)
            for i, unit in enumerate(actual_arg_units):
                if i < len(new_args):
                    new_args[i] = _to_mag(new_args[i], unit)

            # 2. EXECUTE: Function works with unit-less DataArrays
            result = func(*new_args, **kwargs)

            # 3. ADD: Re-apply units to the output
            return _to_unit(result, ret_units)

        return wrapper

    return decorator


def _to_mag(obj: Any, unit: Any) -> Any:
    """Strip units while preserving the xarray shell."""
    if unit is None:
        return obj

    # Handle list of items
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "to"):
        return type(obj)(_to_mag(item, unit) for item in obj)

    target_unit = ureg(unit) if isinstance(unit, str) else unit

    if isinstance(obj, xr.DataArray):
        if hasattr(obj.data, "to"):
            # Return a copy of the DataArray but with raw magnitude data
            return obj.copy(data=obj.data.to(target_unit).magnitude)
        return obj

    if hasattr(obj, "to"):
        return obj.to(target_unit).magnitude

    return obj


def _to_unit(obj: Any, unit: Any) -> Any:
    """Apply units back to the data."""
    if isinstance(unit, (tuple, list)):
        if not isinstance(obj, (tuple, list)) and len(unit) == 1:
            return _to_unit(obj, unit[0])
        return type(obj)(_to_unit(o, u) for o, u in zip(obj, unit))

    if unit is None:
        return obj

    q = ureg(unit) if isinstance(unit, str) else unit

    if isinstance(obj, xr.DataArray):
        # Prevent double-wrapping if it already has units
        if hasattr(obj.data, "to"):
            return obj.copy(data=obj.data.to(q))

        # Add units to the raw data (numpy/dask)
        new_obj = obj.copy(deep=False)
        new_obj.data = obj.data * q
        return new_obj

    return obj * q
