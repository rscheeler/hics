import numpy as np
import xarray as xr
from pint import Quantity
from xarray import DataArray


def kw2da(**kwargs) -> dict:
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
            if isinstance(v, Quantity):
                v = v.to_base_units()  # noqa: PLW2901
            if v.shape == ():
                v = [v]  # noqa: PLW2901
            out[k] = DataArray(v, dims=(k,), coords={k: v})
        else:
            v.data = v.data.to_base_units()
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


def vector_norm(x: xr.DataArray, dim: str, ord=None) -> xr.DataArray:
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
