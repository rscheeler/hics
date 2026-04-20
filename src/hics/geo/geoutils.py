from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr
from loguru import logger

from .. import ureg
from ..utils import basemagxr, compute_if_dask, wraps_xr
from .dem import DEM
from .transforms import GEOD

if TYPE_CHECKING:
    from ..hics import HCS


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


def get_surface_profile(tx_cs: HCS, rx_cs: HCS, lc_skip_ind: int | None = None) -> xr.Dataset:
    """
    Get the surface profile between the two coordinate systems.

    Parameters
    ----------
    tx_cs : CS
        tx Coordinate system to determine profile between.
    rx_cs : CS
        rx Coordinate system to determine profile between.
    lc_skip_ind : int | None
        Index of the land cover class to skip in the interpolation.
    """
    # Self if assume to be transmitter or starting point of the profile
    tx = list(tx_cs.llh) + [tx_cs.hagl]
    # tx = basemagxr(*tx)
    # Other is then the receiver
    rx = list(rx_cs.llh) + [rx_cs.hagl]
    # rx = basemagxr(*rx)

    # Confirm data array sizes and broadcast accordingly
    if tx[0].size == 1 and rx[0].size != 1:
        full_tx = []
        for txi, rxi in zip(tx, rx):
            full_tx.append(xr.full_like(rxi, txi.item()))
        tx = full_tx
    elif rx[0].size == 1 and tx[0].size != 1:
        full_rx = []
        for txi, rxi in zip(tx, rx):
            full_rx.append(xr.full_like(txi, rxi.item()))
        rx = full_rx

    # Make line list
    line = [tx[1], tx[0], rx[1], rx[0]]

    # Make alts list
    alts = [tx[2], tx[3], rx[2], rx[3]]

    # Call surface_profile_function to get the profile
    surf_prof = surface_profile_xr(line, alts, lc_skip_ind=lc_skip_ind)

    return surf_prof


@wraps_xr(None, (ureg.radian, ureg.m))
def surface_profile_xr(line: list, alts: list, lc_skip_ind: int | None = None) -> xr.DataArray:
    """
    This module takes a set of point coordinates and returns the surface profile.

    Parameters
    ----------
    line : list
        List of DataArrays [lon1,lat1,lon2,lat2]
    alts : list
        List of DataArrays [amsl1,agl1,amsl2,agl2]
    lc_skip_ind : int | None
        Index of the land cover class to skip in the interpolation.

    Returns:
    -------
    surface_profile : xr.DataArray
        Contains the surface profile measurements in meters.

    """
    # Iterate over input DataArray
    surface_profiles = []
    lc_profiles = []
    lc_nlcds = []
    for i, (*line_data, txamsl, txagl, rxamsl, rxagl) in enumerate(
        zip(
            *[np.array([l.data]).ravel() for l in line],
            *[np.array([a.data]).ravel() for a in alts],
        )
    ):
        # Geographic distance
        distance_m = GEOD.inv(*line_data, radians=True)[-1]
        distance_km = distance_m / 1e3
        # Interpolate along line to get sampling points and add one every other time to make sure
        # data doesn't concat
        num_samples = determine_num_samples(distance_m) + int(np.mod(i, 2))
        geod_inv = GEOD.inv_intermediate(
            *line_data,
            npts=num_samples,
            initial_idx=0,
            terminus_idx=0,
            radians=True,
            return_back_azimuth=False,
        )

        # Generate DataArrays for interpolation
        distance = np.linspace(0, distance_m, num_samples)
        dims = "distance"
        coords = dict(
            distance=distance,
            txagl=txagl,
            txamsl=txamsl,
            rxagl=rxagl,
            rxamsl=rxamsl,
        )
        lats = xr.DataArray(geod_inv.lats, dims=dims, coords=coords)
        lons = xr.DataArray(geod_inv.lons, dims=dims, coords=coords)

        # Sample elevation profile by using DataArray interpolate
        surface_profile = DEM.interp(lat=lats, lon=lons)
        lc_profile, lc_nlcd = DEM.interp_landcover(lat=lats, lon=lons, lc_skip_ind=lc_skip_ind)
        surface_profile = compute_if_dask(surface_profile)
        lc_profile = compute_if_dask(lc_profile)
        lc_nlcd = compute_if_dask(lc_nlcd)
        # Add units
        surface_profile.data = surface_profile.data
        lc_profile.data = lc_profile.data + surface_profile.data

        # Add units for distance
        dist = surface_profile.distance
        dist.attrs = dict(units="meter")
        surface_profile.coords["distance"] = dist

        # Add distance attribute
        attrs = dict(
            long_name="Surface Profile",
            description="Terrain Surface Profile",
            distance_km=distance_km,
        )
        surface_profile.attrs = {**surface_profile.attrs, **attrs}
        attrs = dict(
            long_name="Land Cover Profile",
            description="Land Cover Surface Profile",
            distance_km=distance_km,
        )
        lc_profile.attrs = {**lc_profile.attrs, **attrs}
        attrs = dict(
            long_name="Land Cover Class", description="Land Cover Class", distance_km=distance_km
        )
        lc_nlcd.attrs = {**lc_nlcd.attrs, **attrs}
        surface_profiles.append(surface_profile)
        lc_profiles.append(lc_profile)
        lc_nlcds.append(lc_nlcd)

    # Create DataArray if data has shape
    if line[0].shape != ():
        surface_profile = xr.DataArray(
            np.array(surface_profiles, dtype="object").reshape(line[0].shape),
            dims=line[0].dims,
            coords=line[0].coords,
        )
        lc_profile = xr.DataArray(
            np.array(lc_profiles, dtype="object").reshape(line[0].shape),
            dims=line[0].dims,
            coords=line[0].coords,
        )
        lc_nlcd = xr.DataArray(
            np.array(lc_nlcds, dtype="object").reshape(line[0].shape),
            dims=line[0].dims,
            coords=line[0].coords,
        )
    else:
        surface_profile = surface_profiles[-1]
        lc_profile = lc_profiles[-1]
        lc_nlcd = lc_nlcds[-1]

    return xr.Dataset(
        dict(
            surface_profile=surface_profile,
            lc_profile=lc_profile,
            lc_nlcd=lc_nlcd,
        )
    )


def inv_transform(self_cs: HCS, other_cs: HCS):
    """
    Run the inverse transformation from pyproj.GEOD to determine forward and back azimuths, plus distances between
    initial points and terminus points.
    """
    # Self if assume to be transmitter or starting point of the profile
    tx = self_cs.llh
    # Other is then the receiver
    rx = other_cs.llh

    # Confirm data array sizes and broadcast accordingly
    if tx[0].size == 1 and rx[0].size != 1:
        full_tx = []
        for txi, rxi in zip(tx, rx):
            full_tx.append(xr.full_like(rxi, txi.item()))
        tx = full_tx
    elif rx[0].size == 1 and tx[0].size != 1:
        full_rx = []
        for txi, rxi in zip(tx, rx):
            full_rx.append(xr.full_like(txi, rxi.item()))
        rx = full_rx

    # Make line list
    line = [tx[1], tx[0], rx[1], rx[0]]

    # Call surface_profile_function to get the profile
    txazs = []
    rxazs = []
    dists = []
    for line_data in zip(*[l.data.to("radian").magnitude.ravel() for l in line]):
        # Geographic distance and azimuths
        txaz, rxaz, dist = GEOD.inv(*line_data, radians=True)
        txazs.append(txaz)
        rxazs.append(rxaz)
        dists.append(dist)
    # Create DataArrays
    if tx[0].dims == ():
        txazs = txazs[0]
    txazs = xr.DataArray(txazs, dims=tx[0].dims, coords=tx[0].coords)
    txazs.data = (txazs.data * ureg.radian).to("degree")
    if rx[0].dims == ():
        rxazs = rxazs[0]
    rxazs = xr.DataArray(rxazs, dims=rx[0].dims, coords=rx[0].coords)
    rxazs.data = (rxazs.data * ureg.radian).to("degree")
    if tx[0].dims == ():
        dists = dists[0]
    dists = xr.DataArray(dists, dims=tx[0].dims, coords=tx[0].coords)
    dists.data = dists.data * ureg.meter
    return txazs, rxazs, dists


def relative_azimuth(self_cs: HCS, other_cs: HCS):
    """Get forward and reverse azimuth between self and other_cs."""
    # Get inverse transform and only return azimuths
    txazs, rxazs, dists = inv_transform(self_cs, other_cs)

    return txazs, rxazs
