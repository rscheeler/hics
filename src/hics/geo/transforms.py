"""Cartographic transforms using pyproj wrapped to handle xarray DataArrays."""

from copy import deepcopy
from typing import Any

import numpy as np
import xarray as xr
from loguru import logger
from pint import Quantity
from pyproj import Geod, Transformer

from .. import ureg
from ..utils import kw2da
from .dem import DEM

GEOD = Geod(ellps="WGS84")


class XRCRSTransformer:
    """
    Wrapper to handle units and DataArrays.

    Note: Use EPSG:4979 for lat,lon,alt coordinate system as it contains the name and units for
    altitude.

    USE EPSG:4978 for geocentric coordinate system

    References:
    -----------
    https://en.wikipedia.org/wiki/World_Geodetic_System
    https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset
    """

    def __init__(self, crs_from: Any, crs_to: Any, **kwargs) -> None:
        self.transformer = Transformer.from_crs(crs_from, crs_to, **kwargs)

    @property
    def source_dims(self) -> tuple:
        """Return source dimensions."""
        return tuple([axis.abbrev.lower() for axis in self.transformer.source_crs.axis_info])

    @property
    def source_units(self) -> tuple:
        """Units for the corresponding source dimensions."""
        return tuple(
            [ureg.Unit(axis.unit_name.lower()) for axis in self.transformer.source_crs.axis_info]
        )

    @property
    def source_base_units(self):
        """Returns the base unit of source units."""
        return tuple([(1 * u).to_base_units().units for u in self.source_units])

    @property
    def target_dims(self) -> tuple:
        """Return target dimensions."""
        return tuple([axis.abbrev.lower() for axis in self.transformer.target_crs.axis_info])

    @property
    def target_units(self) -> tuple:
        """Units for the corresponding target dimensions."""
        return tuple(
            [ureg.Unit(axis.unit_name.lower()) for axis in self.transformer.target_crs.axis_info]
        )

    @property
    def target_base_units(self):
        """Returns the base unit of source units."""
        return tuple([(1 * u).to_base_units().units for u in self.target_units])

    def sourceunitchecker(self, *args):
        """Check units of arguments and convert to base."""
        # Ensure correct units if no units specified it is assumed that it is in the pint base unit
        basemag = list(deepcopy(args))
        for i, (a, unit, bunit) in enumerate(zip(args, self.source_units, self.source_base_units)):
            if isinstance(a, xr.DataArray):
                if isinstance(a.data, Quantity):
                    adc = xr.DataArray(a.data.to(unit).magnitude, dims=a.dims, coords=a.coords)
                else:
                    adc = xr.DataArray(
                        (a.data * bunit).to(unit).magnitude, dims=a.dims, coords=a.coords
                    )
            elif isinstance(a, Quantity):
                adc = a.to(unit).magnitude
            else:
                adc = (a * bunit).to(unit).magnitude
            basemag[i] = adc
        return basemag

    def transform(self, *args, to_da: bool = False, hagl: bool = False, outunits: bool = False):
        """Perform transform on source data to target data."""
        # Check length
        if len(args) != len(self.source_dims):
            raise ValueError("Arg number doesn't match source dims")

        # Ensure correct units if no units specified it is assumed that it is in the pint base unit
        basemags = self.sourceunitchecker(*args)

        # Add ground height if correct source crs and flag specified
        if self.transformer.source_crs.to_epsg() == 4979 and hagl:
            basemags[2] = hagl2amsl(np.deg2rad(basemags[0]), np.deg2rad(basemags[1]), basemags[2])

        # Perform transform
        transdata = self.transformer.transform(*basemags)
        transdata = list(transdata)

        # Convert to DataArray if input was
        if isinstance(basemags[0], xr.DataArray):
            for i, t in enumerate(transdata):
                transdata[i] = xr.DataArray(t, dims=basemags[0].dims, coords=basemags[0].coords)

        # Remove ground height if correct target crs and flag specified
        if self.transformer.target_crs.to_epsg() == 4979 and hagl:
            transdata[2] = amsl2hagl(*transdata)

        # Apply units on output
        if outunits:
            for i, (t, unit) in enumerate(zip(transdata, self.target_units)):
                if isinstance(t, xr.DataArray):
                    t.data *= unit
                else:
                    t *= unit
                transdata[i] = t

        # Convert to a single DataArray
        if to_da:
            all_same = all(x == self.target_units[0] for x in self.target_units)
            dims = [
                "position",
            ]
            coords = dict(position=list(self.target_dims))

            # Take magnitudes depending on data type
            if isinstance(transdata[0], Quantity):
                tlist = [t.magnitude for t in transdata]

            elif isinstance(transdata[0], xr.DataArray):
                if isinstance(transdata[0].data, Quantity):
                    tlist = [t.data.magnitude for t in transdata]
                else:
                    tlist = [t.data for t in transdata]

                dims = dims + list(transdata[0].dims)
                coords = {**coords, **transdata[0].coords}
            else:
                tlist = transdata

            transdata = xr.DataArray(tlist, dims=dims, coords=coords)

            # Add back in units if all dims the same
            if all_same:
                transdata.data = transdata.data * self.target_units[0]

        return transdata


def hagl2amsl(*coords):
    """
    Convert height above ground level (HAGL) to height above mean sea level (AMSL).
    Coords are lat:radian, lon:radian, hagl:meter
    """
    ll = kw2da(lat=coords[0], lon=coords[1])

    gndlevel = DEM.interp(lat=ll["lat"], lon=ll["lon"])

    if coords[0].shape == ():
        gndlevel = gndlevel.values.squeeze()
    return coords[2] + gndlevel


def amsl2hagl(*coords):
    """Convert height above mean sea level (AMSL) to height above ground level (HAGL).
    Coords are lat:radian, lon:radian, amsl:meter
    """
    ll = kw2da(lat=coords[0], lon=coords[1])
    gndlevel = DEM.interp(lat=ll["lat"], lon=ll["lon"])

    if coords[0].shape == ():
        gndlevel = gndlevel.values.squeeze()

    return coords[2] - gndlevel


llh2geocent = XRCRSTransformer("EPSG:4979", "EPSG:4978")
geocent2llh = XRCRSTransformer("EPSG:4978", "EPSG:4979")
latlong2geocent = XRCRSTransformer("EPSG:4326", "EPSG:4978")
geocent2latlong = XRCRSTransformer("EPSG:4978", "EPSG:4326")


def geo_mid(lon0, lat0, lon1, lat1):
    """
    Return the latitude and longitude points corresponding to the mid point between the coordinates
    provided.

    Returns:
    -------
    lat:float
    lon:float
    """
    invint = GEOD.inv_intermediate(
        lon0, lat0, lon1, lat1, initial_idx=0, terminus_idx=0, npts=3, return_back_azimuth=False
    )
    lat = invint.lats[1]
    lon = invint.lons[1]
    return lat, lon
