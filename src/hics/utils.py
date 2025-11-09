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
                v = v.to_base_units()
            if v.shape == ():
                v = [v]
            out[k] = DataArray(v, dims=(k,), coords={k: v})
        else:
            v.data = v.data.to_base_units()
            out[k] = v

    return out


class Singleton:
    """
    Singleton class
    See: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
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


def determine_num_samples(
    distance_m, lower_limit: int = 2, upper_limit: int = 600, stretch: float = 1e-5
):
    """
    Guarantee a number of samples between lower and upper limit. Can be used to support the
    Longley-Rice Irregular Terrain Model is limited to only 600 surface points, so this function
    ensures upper limit is not exceeded.

    Parameters
    ----------
    distance_m : int
        Distance between transmitter and receiver in meters.

    Returns:
    -------
    num_samples : int
        Number of samples between lower and upper limit.

    """
    # This is -1/x translated and rescaled:
    # - to hit 2 at x=0
    # - to approach 600 as x->inf

    # Online plot to get a feel for the function:
    # https://www.wolframalpha.com/input/?i=plot+%28-1%2F%280.0001x%2B%281%2F598%29%29%29+%2B+600+from+0+to+100

    # Limits:
    # https://www.wolframalpha.com/input/?i=limit+%28-1%2F%280.0001x%2B%281%2F598%29%29%29+%2B+600

    return round((-1 / ((stretch * distance_m) + (1 / (upper_limit - lower_limit)))) + upper_limit)
