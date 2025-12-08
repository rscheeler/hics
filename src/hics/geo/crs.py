from copy import deepcopy
from typing import Any, Optional

import numpy as np
import xarray as xr
from loguru import logger
from pyproj import Transformer
from scipy.spatial.transform import Rotation

from .. import HCS, ureg
from ..datatypes import _QUATERNION_COORD_DICT, _QUATERNION_COORDS, _QUATERNION_DIM
from ..utils import vector_norm
from .transforms import XRCRSTransformer, hagl2amsl


def from_crs(
    cls,
    coords,
    epsg: str = "EPSG:4979",
    un: str = "h",
    ux: str = "lon",
    hagl: bool = False,
    **kwargs,
) -> HCS:
    """
    Generate coordinate system from lat, lon, h. Rotation defined by unit normal un and unit x.

    References:
    ----------
    https://en.wikipedia.org/wiki/World_Geodetic_System
    https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset

    Parameters
    ----------
    coords : list
        Coordiante in epsg coordinate system
    epsg : str, default="EPSG:4949"
        EPSG Geodetic Parameter Dataset code
    un : str
        Dimension for unit normal, should be a dimension of epsg
    ux : str
        Dimension for unit x vector, should be a dimension of epsg
    hagl : bool
        Whether height is specified as above ground level.
    """
    # Make sure coords is list
    coords = list(coords)

    # Get HAGL to AMSL to ensure unit vectors correctly oriented
    if epsg == "EPSG:4979" and hagl:
        hamsl = hagl2amsl(
            coords[0].to("radian").magnitude,
            coords[1].to("radian").magnitude,
            coords[2].to("m").magnitude,
        )
        coords[-1] = hamsl * ureg.m
        hagl = False
    # Select item if DataArray and no shape
    for i, l in enumerate(coords):
        if isinstance(l, xr.DataArray):
            if l.shape == ():
                coords[i] = l.item()

    # Perform transform
    togeocent = XRCRSTransformer(epsg, "EPSG:4978")
    pnt = togeocent.transform(*coords, to_da=True, hagl=hagl)
    # Form unit vectors by moving a very small distance along lat, lon, or h
    idxmap = togeocent.source_dims
    delta = [d * u for d, u in zip([1e-6, 1e-6, 0.01], togeocent.source_units)]
    coords_un = deepcopy(coords)
    coords_ux = deepcopy(coords)

    # Shift unit vectors
    coords_un[idxmap.index(un.lower())] = (
        coords_un[idxmap.index(un.lower())] + delta[idxmap.index(un.lower())]
    )
    coords_ux[idxmap.index(ux.lower())] = (
        coords_ux[idxmap.index(ux.lower())] + delta[idxmap.index(ux.lower())]
    )

    # Create unit vectors
    un = togeocent.transform(*coords_un, to_da=True, hagl=hagl)
    ux = togeocent.transform(*coords_ux, to_da=True, hagl=hagl)

    # Create normal vector
    un = un - pnt
    ux = ux - pnt
    # Remove units
    un.data = un.data.magnitude
    ux.data = ux.data.magnitude
    un /= vector_norm(un, "position")
    ux /= vector_norm(ux, "position")

    # Transpose so position is first dimension
    un = un.transpose("position", ...)
    ux = ux.transpose("position", ...)

    # Determine unit y vector by taking the cross product of normal (z) and x-vector
    uy = xr.apply_ufunc(np.cross, un, ux, kwargs=dict(axis=0))

    # Get dims and coords and make into a single DataArray
    rd = list(un.dims)
    rd.pop(rd.index("position"))
    rc = dict(un.coords)
    rc.pop("position")

    # Create rotaiton objects
    uxshape = ux.shape
    ux = ux.values.reshape(3, -1)
    uy = uy.values.reshape(3, -1)
    un = un.values.reshape(3, -1)

    matrices = np.stack([ux, uy, un], axis=-1).transpose(1, 2, 0)  # shape: (N, 3, 3)
    logger.debug(matrices.shape)
    rots = Rotation.from_matrix(matrices)
    rots = xr.DataArray(
        rots.as_quat().reshape(list(uxshape[1:]) + [len(_QUATERNION_COORDS)]),
        dims=rd + [_QUATERNION_DIM],
        coords={**rc, **_QUATERNION_COORD_DICT},
    )

    # # Making into array and reshape
    # if uxshape[1:] != ():
    #     rots = np.array(rots).reshape(uxshape[1:])
    #     rot = xr.DataArray(rots, dims=rd, coords=rc)
    # else:
    #     rot = rots[0]

    return cls(pnt, rotation=rots)
