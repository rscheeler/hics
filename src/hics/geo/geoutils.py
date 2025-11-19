import numpy as np
import xarray as xr

from .. import ureg
from ..hics import HCS
from .dem import DEM
from .transforms import GEOD


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


def get_surface_profile(tx_cs: HCS, rx_cs: HCS):
    """
    Get the surface profile between the two coordinate systems.

    Parameters
    ----------
    tx_cs : CS
        tx Coordinate system to determine profile between.
    rx_cs : CS
        rx Coordinate system to determine profile between.
    """
    # Self if assume to be transmitter or starting point of the profile
    tx = list(tx_cs.llh) + [tx_cs.hagl]
    # Other is then the receiver
    rx = list(rx_cs.llh) + [rx_cs.hagl]

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
    surf_prof = surface_profile_xr(line, alts)

    return surf_prof


def surface_profile_xr(line: list, alts: list) -> xr.DataArray:
    """
    This module takes a set of point coordinates and returns the surface profile.

    Parameters
    ----------
    line : list
        List of DataArrays [lon1,lat1,lon2,lat2]
    alts : list
        List of DataArrays [amsl1,agl1,amsl2,agl2]

    Returns:
    -------
    surface_profile : xr.DataArray
        Contains the surface profile measurements in meters.

    """
    # Iterate over input DataArray
    surface_profiles = []
    clutter_profiles = []
    clutter_nlcds = []
    for i, (*line_data, txamsl, txagl, rxamsl, rxagl) in enumerate(
        zip(
            *[l.data.to("radian").magnitude.ravel() for l in line],
            *[a.data.to("m").magnitude.ravel() for a in alts],
        )
    ):
        # Geographic distance
        distance_m = GEOD.inv(*line_data, radians=True)[-1]
        distance_km = distance_m / 1e3

        # Interpolate along line to get sampling points and add one every other time to make sure data doesn't concat
        num_samples = determine_num_samples(distance_m) + int(np.mod(i, 2))
        geod_inv = GEOD.inv_intermediate(
            *line_data, npts=num_samples, initial_idx=0, terminus_idx=0, radians=True
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
        clutter_profile, clutter_nlcd = DEM.interp_clutter(lat=lats, lon=lons)

        # Add units
        surface_profile.data = surface_profile.data * ureg.m
        clutter_profile.data = clutter_profile.data * ureg.m + surface_profile.data

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
            long_name="Clutter Profile",
            description="Clutter Surface Profile",
            distance_km=distance_km,
        )
        clutter_profile.attrs = {**clutter_profile.attrs, **attrs}
        attrs = dict(
            long_name="Clutter Class", description="Clutter Class", distance_km=distance_km
        )
        clutter_nlcd.attrs = {**clutter_nlcd.attrs, **attrs}
        surface_profiles.append(surface_profile)
        clutter_profiles.append(clutter_profile)
        clutter_nlcds.append(clutter_nlcd)

    # Create DataArray if data has shape
    if line[0].shape != ():
        surface_profile = xr.DataArray(
            np.array(surface_profiles, dtype="object").reshape(line[0].shape),
            dims=line[0].dims,
            coords=line[0].coords,
        )
        clutter_profile = xr.DataArray(
            np.array(clutter_profiles, dtype="object").reshape(line[0].shape),
            dims=line[0].dims,
            coords=line[0].coords,
        )
        clutter_nlcd = xr.DataArray(
            np.array(clutter_nlcds, dtype="object").reshape(line[0].shape),
            dims=line[0].dims,
            coords=line[0].coords,
        )
    else:
        surface_profile = surface_profiles[-1]
        clutter_profile = clutter_profiles[-1]
        clutter_nlcd = clutter_nlcds[-1]

    return xr.Dataset(
        dict(
            surface_profile=surface_profile,
            clutter_profile=clutter_profile,
            clutter_nlcd=clutter_nlcd,
        )
    )
