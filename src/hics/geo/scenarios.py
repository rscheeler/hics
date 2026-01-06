from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pint import Quantity
from scipy.spatial.transform import Rotation

from .. import ureg
from ..hics import GLOBAL_CS, HCS
from .dem import DEM
from .transforms import GEOD, llh2geocent


def interp_llpnts2hcs(
    pnts: list,
    h: Quantity,
    speed: Quantity,
    start_time: np.datetime64,
    new_n_pnts: int | None = None,
    loops: int = 1,
    bank_turns: bool = False,
    hagl: bool = False,
    z: int = 1,
    time_precision="s",
) -> HCS:
    """
    Takes in a list of (lat, lon) pairs and a constant height which can either be specified as
    height above ground (alt_above_ground=True) or altitude above mean sea level
    (alt_above_ground=False). The time for each point is determined from the speed and start time.
    The point list can be repeated by increasing the number of loops, and turns can be banked if
    desired. Finally, if new_n_pnts is specified, the HCS object is used to interpolate to a finer
    sampling. The HCS object is returned.

    Parameters
    ----------
    pnts: list
        List of (lat, lon) tuples of points
    h : Quantity
        Constant height - which is either height above ground or mean sea level depending on the
        alt_above_ground flag
    speed : Quantity
        The speed of the points
    start_time : np.datetime64
        Start time
    new_n_pnts : int
        Number of samples, defaults to len(pnts)
    loops : int
        Number of times to iterate the
    bank_turns : bool
        Whether to bank t
        urns
    hagl : bool

    Returns:
    -------
    cs : HCS

    """
    isotime = []
    rots = []
    pos = []
    loop_start = start_time
    # Load DEM if needed
    if hagl:
        DEM.load(pnts)
    for ii in range(loops):
        if ii > 0:
            jj = 1
        else:
            jj = 0

        # Calculate geodetic distance
        dist_m = GEOD.line_lengths([p[1] for p in pnts], [p[0] for p in pnts[jj:]])

        # Add in zero distance
        dist_m = [0] + dist_m

        # Convert to pint quantity
        dist_m *= ureg.m

        # Get time from distance and speed
        elapsed_time = (dist_m / speed).to(time_precision)
        isotime.append(
            loop_start + np.cumsum(elapsed_time.magnitude).astype(f"timedelta64[{time_precision}]")
        )

        # Get the unit vectors
        pnt_ecef = llh2geocent.transform(
            pnts[jj][0] * ureg.degree, pnts[jj][1] * ureg.degree, h, hagl=hagl
        )

        pos += [pnt_ecef]

        bank_rots = dict()
        for kk, pnt in enumerate(pnts[(jj + 1) :]):
            # Convert LLA to ECEF
            pnt_ecef = llh2geocent.transform(
                pnt[0] * ureg.degree, pnt[1] * ureg.degree, h, hagl=hagl
            )

            # Point x- in direction of movement
            ux = np.array(pnt_ecef) - np.array(pos[-1])

            # Point z- up
            un = llh2geocent.transform(
                pnt[0] * ureg.degree, pnt[1] * ureg.degree, h + (z * 0.001) * ureg.m, hagl=hagl
            )
            un = np.array(un) - pnt_ecef
            ux /= np.linalg.norm(np.array(ux))
            un /= np.linalg.norm(np.array(un))
            uy = np.cross(un, ux)

            pos.append(pnt_ecef)
            rots.append(Rotation.from_matrix(np.array([ux, uy, un])))

            # Bank turns if specified
            if bank_turns:
                # Determine angle
                if len(rots) > 1:
                    xang = np.rad2deg(
                        np.arccos(np.dot(rots[-1].as_matrix()[0, :], rots[-2].as_matrix()[0, :]))
                    )
                    # If angle greater than some tolerance then bank
                    if xang > 0.05:
                        # Get rotation sign
                        ytmp = np.rad2deg(
                            np.arccos(
                                np.dot(rots[-1].as_matrix()[0, :], rots[-2].as_matrix()[1, :])
                            )
                        )
                        if ytmp > 90:
                            rot_sgn = -1.0
                        if ytmp < 90:
                            rot_sgn = 1.0

                        # Estimate turn radius
                        # https://en.wikipedia.org/wiki/Standard_rate_turn
                        turn_r = (
                            360
                            / xang
                            * np.linalg.norm(np.array(pos[-1]) - np.array(pos[-2]))
                            / np.pi
                            / 2
                        ) * ureg.m
                        # Calculate bank angle based on speed and turn radius
                        bank_angle = np.rad2deg(
                            np.arctan(
                                speed.to_base_units() ** 2
                                / (turn_r * (9.80665 * ureg.meter / (1 * ureg.second) ** 2))
                            )
                        )
                        # Create rotation object
                        bank_rots[rots[-1]] = Rotation.from_euler(
                            "XZX", [rot_sgn * bank_angle.to("degree").magnitude, 0, 0], degrees=True
                        )

        # Compound the rotation to add in bank
        for k, v in bank_rots.items():
            rots[rots.index(k)] = v * rots[rots.index(k)]

        # Increment start time
        loop_start = isotime[-1][-1] + np.cumsum(elapsed_time.magnitude).astype("timedelta64[s]")[1]
        rots = [rots[0]] + rots

    isotime = np.hstack(isotime).astype("datetime64[ns]")

    # Position DataArray
    pos_xr = xr.DataArray(
        np.array(pos) * ureg.m,
        dims=("time", "position"),
        coords=dict(
            time=isotime,
            position=["x", "y", "z"],
        ),
    )

    # Rotation DataArray
    rot_xr = xr.DataArray(
        rots,
        dims=("time",),
        coords=dict(
            time=isotime,
        ),
    )

    # Create coordinate system
    cs = HCS(
        pos_xr,
        rotation=rot_xr,
        reference=GLOBAL_CS,
    )

    # Interpolate to new number of points if defined
    if new_n_pnts is not None:
        start_time = pd.Timestamp(cs.origin.time[0].item())
        end_time = pd.Timestamp(cs.origin.time[-1].item())
        ts = np.linspace(start_time.value, end_time.value, new_n_pnts)
        ts = np.asarray(pd.to_datetime(ts))

        cs = cs.interp(ts)

    return cs


def racetrack_latlon(
    center_lat: Quantity,
    center_lon: Quantity,
    length: Quantity,
    width: Quantity,
    npts: int,
    angle=0 * ureg.degree,
):
    """
    Makes a racetrack with the specified number of points (or close), length, and width. Point order
    is a counterclockwise track starting in the lower right.

    Parameters
    ----------
    center_lat : Quantity
        Race track center latitude
    center_lon : Quantity
        Race track center longitude
    length : Quantity
        Length of the racetrack
    width : Quantity
        Width (turn diameter)
    npts : int
        Number of points
    angle : Quantity
        Forward azimuth angle
    """
    # Only left turns (Counter Clock-wise)
    # Start in lower left

    # Total Length
    l_tot = length * 2 + np.pi * width

    # Convert to magnitudes
    center_lat_deg = center_lat.to("degree").magnitude
    center_lon_deg = center_lon.to("degree").magnitude
    length_m = length.to("m").magnitude
    width_m = width.to("m").magnitude
    angle_deg = angle.to("degree").magnitude
    angle_rad = angle.to("radians").magnitude

    # Semicircle centers
    start_circle = GEOD.fwd(
        center_lon_deg,
        center_lat_deg,
        np.rad2deg(np.angle(np.exp(1j * (angle_rad - np.pi)))),
        length_m / 2,
    )
    end_circle = GEOD.fwd(
        center_lon_deg,
        center_lat_deg,
        angle_deg,
        length_m / 2,
    )
    # Get corners
    lr = GEOD.fwd(
        start_circle[0],
        start_circle[1],
        np.rad2deg(np.angle(np.exp(1j * (angle_rad - np.pi / 2)))),
        width_m / 2,
    )
    ll = GEOD.fwd(
        start_circle[0],
        start_circle[1],
        np.rad2deg(np.angle(np.exp(1j * (angle_rad + np.pi / 2)))),
        width_m / 2,
    )

    ur = GEOD.fwd(
        end_circle[0],
        end_circle[1],
        np.rad2deg(np.angle(np.exp(1j * (angle_rad - np.pi / 2)))),
        width.to("m").magnitude / 2,
    )
    ul = GEOD.fwd(
        end_circle[0],
        end_circle[1],
        np.rad2deg(np.angle(np.exp(1j * (angle_rad + np.pi / 2)))),
        width.to("m").magnitude / 2,
    )

    # Start points
    pnts = [(ll[1], ll[0])]
    # First Leg
    leg_pnts = int(np.floor((npts - 1) * length / l_tot))
    first_leg = GEOD.inv_intermediate(*ll[:2], *ul[:2], npts=leg_pnts, return_back_azimuth=False)
    pnts += [
        (
            lat,
            lon,
        )
        for lon, lat in zip(first_leg.lons, first_leg.lats, strict=False)
    ]

    # Go around first semi-circle
    circ_pnts = int(np.floor(((npts - 1) - 2 * leg_pnts) / 2))
    da = 180 * ureg.degree / (circ_pnts + 1)
    for i in range(1, circ_pnts + 1):
        ang = angle.to("degree") + 90 * ureg.degree - da * i
        ang = np.rad2deg(np.angle(np.exp(1j * ang.to("radians").magnitude)))
        pnt = GEOD.fwd(
            end_circle[0],
            end_circle[1],
            ang,
            width_m / 2,
        )
        pnts.append((pnt[1], pnt[0]))

    # Return Leg
    return_leg = GEOD.inv_intermediate(*ur[:2], *lr[:2], npts=leg_pnts, return_back_azimuth=False)
    pnts += [
        (
            lat,
            lon,
        )
        for lon, lat in zip(return_leg.lons, return_leg.lats, strict=False)
    ]

    # End semi-circle
    for i in range(1, circ_pnts + 1):
        ang = angle.to("degree") - 90 * ureg.degree - da * i
        ang = np.rad2deg(np.angle(np.exp(1j * ang.to("radians").magnitude)))
        pnt = GEOD.fwd(
            start_circle[0],
            start_circle[1],
            ang,
            width_m / 2,
        )
        pnts.append((pnt[1], pnt[0]))

    # Close the loop
    pnts += [pnts[0]]

    # TODO: figure out how to ensure the length is correct
    return pnts


def heading_latlon(
    starting_lat: Quantity,
    starting_lon: Quantity,
    speed: Quantity,
    duration: Quantity,
    sample_rate: Quantity,
    azimuth: Quantity = 0 * ureg.degree,
) -> list[tuple]:
    """Generate lat,lon points for specific azimuth heading.

    Parameters
    ----------
    starting_lat : Quantity
        Starting latitude
    starting_lon : Quantity
        Starting longitude
    speed : Quantity
        Speed
    duration : Quantity
        Time of flight
    sample_rate : Quantity
        How fast to sample position
    azimuth : Quantity, optional
        Azimuth of heading, by default 0*ureg.degree

    Returns:
    -------
    list[tuple] : List of lat,lon pairs.
    """
    # Get sample distance and total points
    delta = (speed / sample_rate).to("m").magnitude
    npoints = int(duration * sample_rate) + 1

    # Convert to degrees
    az_deg = azimuth.to("degree").magnitude
    latdeg = starting_lat.to("degree").magnitude
    londeg = starting_lon.to("degree").magnitude

    # Call fwd and include initial and terminal points
    res = GEOD.fwd_intermediate(
        londeg,
        latdeg,
        az_deg,
        npoints,
        delta,
        initial_idx=0,
        terminus_idx=0,
    )
    points = [(la, lo) for la, lo in zip(res.lats, res.lons)]
    return points
