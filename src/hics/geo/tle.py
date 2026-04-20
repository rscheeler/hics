import hashlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from loguru import logger
from matplotlib import animation
from scipy.spatial.transform import Rotation
from skyfield.api import EarthSatellite, Topos, load, wgs84
from skyfield.framelib import itrs

from .. import GLOBAL_CS, HCS, ureg
from ..plotting import viewcs
from .config import DEM_SETTINGS


def get_celestrak_tle(group: str) -> str | None:
    """Fetches TLE data from CelesTrak of the specified group.

    Parameters
    ----------
    group : str
        The satellite group to fetch TLE data for (e.g., "starlink").

    Returns:
    -------
    str or None
        The TLE data as a string, or None if an error occurs.
    """
    try:
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
        response = requests.get(url)
        response.raise_for_status()
        # Check if the response is the "Data has not updated" message
        if "GP data has not updated" in response.text:
            logger.debug(f"CelesTrak: {group} data hasn't changed. Using cached file.")
            return None
        return response.text
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching TLE data: {e}")
        return None


def calculate_hash(data: str | None) -> str | None:
    """Calculates the SHA-256 hash of the given data.

    Parameters
    ----------
    data : str or None
        The data to hash.

    Returns:
    -------
    str or None
        The SHA-256 hash of the data, or None if data is None.
    """
    if data:
        return hashlib.sha256(data.encode()).hexdigest()
    else:
        return None


def check_celestrak_tle(group: str, dir: Path = DEM_SETTINGS.TLE_FOLDER) -> bool:
    """Checks if the latest TLE data is different from the saved file.

    Parameters
    ----------
    group : str
        The satellite group to fetch TLE data for (e.g., "starlink").
    dir : Path
        The directory to save the TLE file.

    Returns:
    -------
    bool
        True if the TLE data has been updated, False otherwise.
    """
    filename = dir / f"{group}_celstrak_tle.txt"
    new_tle_data = get_celestrak_tle(group=group)

    if new_tle_data is None:
        return False

    new_hash = calculate_hash(new_tle_data)

    try:
        with open(filename, newline="") as f:
            old_tle_data = f.read()
        old_hash = calculate_hash(old_tle_data)

        if new_hash != old_hash:
            logger.info("TLE data has been updated, updating file.")
            with open(filename, "w", newline="") as f:
                f.write(new_tle_data)  # write the new data to the file.
            return True
        else:
            logger.debug("TLE data is up-to-date.")
            return False

    except FileNotFoundError:
        logger.debug("Existing TLE file not found. Creating new file.")
        with open(filename, "w", newline="") as f:
            f.write(new_tle_data)
        return True  # Because a file was just created, it is by definition a change.
    except OSError as e:
        logger.error(f"Error reading/writing file: {e}")
        return False


def get_closest_from_tle(
    latitude: float,
    longitude: float,
    group: str,
    t0: datetime | None = None,
    update_tle: bool = False,
) -> tuple[EarthSatellite | None, list[EarthSatellite]]:
    """Finds the closest satellite to a given location at a specified time.

    Parameters
    ----------
    latitude : float
        The latitude of the observer's location (degrees).
    longitude : float
        The longitude of the observer's location (degrees).
    group : str
        The satellite group to fetch TLE data for (e.g., "starlink").
    t0 : datetime, optional
        The time to calculate the satellite positions (defaults to now).
    update_tle : bool
        Whether to update the TLE file.

    Returns:
    -------
    tuple of (EarthSatellite or None, list of EarthSatellite)
        A tuple containing the closest EarthSatellite object and a list of all EarthSatellite objects.
    """
    # Load time scales. timescale allows for time conversions.
    tscale = load.timescale()

    # Define observer point, Topos creates a location on the surface of the earth.
    observer = Topos(latitude, longitude)

    # Define time
    if t0 is None:
        t0 = datetime.now(UTC)
    time = tscale.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)

    # Load TLE data
    tle_file = DEM_SETTINGS.TLE_FOLDER / f"{group}_celstrak_tle.txt"
    if update_tle or not tle_file.exists():
        check_celestrak_tle(group, DEM_SETTINGS.TLE_FOLDER)
    lines = open(tle_file).readlines()

    # Parse TLE data, EarthSatellite creates a satellite object from the TLE data.
    satellites = []
    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        satellites.append(EarthSatellite(line1, line2, name))

    # Find closest satellite
    min_distance = float("inf")
    closest_satellite = None

    for satellite in satellites:
        # Get satellite position in ICRS, .xyz.m returns the position vector in meters.
        geocentric = satellite.at(time)
        satellite_pos = geocentric.xyz.m

        # Calculate observer position in ICRS
        observer_geocentric = observer.at(time)
        observer_pos = observer_geocentric.xyz.m

        # Calculate distance
        distance = np.linalg.norm(np.array(satellite_pos) - np.array(observer_pos)) / 1000.0  # km

        if distance < min_distance:
            min_distance = distance
            closest_satellite = satellite

    # Log results
    if closest_satellite:
        logger.success(f"Closest satellite: {closest_satellite.name}")
        logger.debug(f"Distance: {min_distance:.2f} km")
        geocentric = satellite.at(time)
        lat, lon = wgs84.latlon_of(geocentric)
        # Convert satellite position to ECEF using ITRS frame.
        logger.debug(f"Satellite ECEF: {geocentric.frame_xyz(itrs).m}")
        logger.debug(
            f"Satellite Latitude, Longitude: {lat.degrees:.4f} degrees,{lon.degrees:.4f} degrees"
        )
    else:
        logger.warning("No satellites found.")
    return closest_satellite, satellites


def moving_cloud(
    satellites,
    sat_highlight,
    t0=None,
    tdelta=timedelta(seconds=20),
    npts=7,
    msize=2,
    mhsize=5,
    animate=True,
):
    # Get positions
    pos_xr = satellite_position_time(satellites, t0=t0, tdelta=tdelta, npts=npts)
    posh_xr = satellite_position_time([sat_highlight], t0=t0, tdelta=tdelta, npts=npts)

    # Convert to km
    pos_xr.data = pos_xr.data.to("km")
    posh_xr.data = posh_xr.data.to("km")

    # Create figure with Global HCS
    ax = viewcs(GLOBAL_CS, GLOBAL_CS, units="km", vector_length=1000)
    ax.set_xlim(-7500, 7500)
    ax.set_ylim(-7500, 7500)
    ax.set_zlim(-7500, 7500)

    # Get middle as this should be t0
    nplt = int(npts / 2)
    # Setup plot
    pos_h_t0 = posh_xr.isel(time=nplt)
    (scath,) = ax.plot(
        pos_h_t0.sel(position="x"),
        pos_h_t0.sel(position="y"),
        pos_h_t0.sel(position="z"),
        c="C1",
        marker="o",
        alpha=1,
        markersize=mhsize,
        zorder=10,
        ls="",
    )
    pos_t0 = pos_xr.isel(time=nplt)
    (scat,) = ax.plot(
        pos_t0.sel(position="x"),
        pos_t0.sel(position="y"),
        pos_t0.sel(position="z"),
        c="C0",
        marker="o",
        alpha=0.2,
        markersize=msize,
        zorder=0,
        ls="",
    )

    # Turn off the axes
    ax.set_axis_off()
    plt.tight_layout()

    if animate:

        def moving_cs(i):
            # Get data
            pos_ti = pos_xr.isel(time=i)
            pos_h_ti = posh_xr.isel(time=i)

            scath._verts3d = (
                pos_h_ti.sel(position="x"),
                pos_h_ti.sel(position="y"),
                pos_h_ti.sel(position="z"),
            )
            scat._verts3d = (
                pos_ti.sel(position="x"),
                pos_ti.sel(position="y"),
                pos_ti.sel(position="z"),
            )

            return (scath,)

        return animation.FuncAnimation(
            plt.gcf(), moving_cs, save_count=pos_xr.time.size, interval=10, blit=True
        )
    else:
        return ax


def satellite_position_time(
    satellites: list[EarthSatellite],
    t0: datetime | None = None,
    tdelta: timedelta = timedelta(seconds=20),
    npts: int = 7,
) -> xr.DataArray:
    """Computes the ECEF position of satellites over a time range.

    Parameters
    ----------
    satellites : list of EarthSatellite
        A list of Skyfield EarthSatellite objects.
    t0 : datetime, optional
        The center time of the time range (defaults to now).
    tdelta : timedelta, optional
        The time delta between each point in the time range (defaults to 20 seconds).
    npts : int, optional
        The number of points in the time range (defaults to 7). If even, it's incremented to be odd.

    Returns:
    -------
    xr.DataArray
        An xarray DataArray containing the ECEF positions of the satellites over time.
        Dimensions: (satellite, time, position).
        Coordinates: satellite names, time stamps, and position components (x, y, z).
    """
    # Load time scales. timescale allows for time conversions.
    tscale = load.timescale()

    # Define the center time if not provided.
    if t0 is None:
        t0 = datetime.now(UTC)

    # Ensure npts is odd for symmetrical time range.
    if npts % 2 == 0:
        npts += 1

    # Calculate start and end times for the time range.
    start_time = pd.Timestamp(t0 - tdelta * int((npts - 1) / 2))
    end_time = pd.Timestamp(t0 + tdelta * int((npts - 1) / 2))

    # Create an array of time stamps within the range.
    ts = np.linspace(start_time.value, end_time.value, npts)
    ts = pd.to_datetime(ts)

    # Convert pandas timestamps to Skyfield Time objects.
    times = [tscale.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) for dt in ts]

    # Initialize a list to store satellite positions.
    pos = []

    # Iterate over each satellite.
    for sat in satellites:
        pos_t = []
        # Iterate over each time in the time range.
        for ti in times:
            # Get ECEF position using the ITRS frame. frame_xyz(itrs).m returns the position vector in meters.
            satellite_ecef = sat.at(ti).frame_xyz(itrs).m
            pos_t.append(satellite_ecef)
        pos.append(pos_t)

    # Extract satellite names for coordinates.
    sat_names = [sat.name for sat in satellites]

    # Create an xarray DataArray to store the results with units.
    pos_xr = xr.DataArray(
        np.array(pos) * ureg.m,
        dims=("satellite", "time", "position"),
        coords=dict(
            satellite=sat_names,
            time=np.asarray(ts),
            position=["x", "y", "z"],
        ),
    )

    return pos_xr


def satellite_to_cs(
    satellite: EarthSatellite,
    t0: datetime | None = None,
    tdelta: timedelta = timedelta(seconds=20),
    npts: int = 7,
) -> HCS:
    """Creates a coordinate system aligned with the satellite's velocity and nadir pointing.

    This function computes the satellite's position and orientation over a time range.
    The x-axis of the coordinate system is aligned with the satellite's velocity vector,
    and the z-axis points towards the nadir (the point on the Earth's surface directly below the satellite).

    Parameters
    ----------
    satellite : EarthSatellite
        The Skyfield EarthSatellite object.
    t0 : datetime, optional
        The center time of the time range (defaults to now).
    tdelta : timedelta, optional
        The time delta between each point in the time range (defaults to 20 seconds).
    npts : int, optional
        The number of points in the time range (defaults to 7). If even, it's incremented to be odd.

    Returns:
    -------
    HCS
        A HCS object representing the satellite's coordinate system.
    """
    # Load time scales. timescale allows for time conversions.
    tscale = load.timescale()

    # Define the center time if not provided.
    if t0 is None:
        t0 = datetime.now(UTC)

    # Ensure npts is odd for symmetrical time range.
    if npts % 2 == 0:
        npts += 1

    # Calculate start and end times for the time range.
    start_time = pd.Timestamp(t0 - tdelta * int((npts - 1) / 2))
    end_time = pd.Timestamp(t0 + tdelta * int((npts - 1) / 2))

    # Create an array of time stamps within the range.
    ts = np.linspace(start_time.value, end_time.value, npts)
    ts = pd.to_datetime(ts)

    # Convert pandas timestamps to Skyfield Time objects.
    times = [tscale.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) for dt in ts]

    # Initialize lists to store position and rotation data.
    pos = []
    rots = []

    # Iterate over each time in the time range.
    for ti in times:
        # Get ECEF position and velocity using the ITRS frame.
        geocentric = satellite.at(ti)
        satellite_ecef, satellite_velocity_ecef = geocentric.frame_xyz_and_velocity(itrs)

        # Calculate position and velocity vectors.
        position_vector = np.array(satellite_ecef.m)
        velocity_vector = np.array(satellite_velocity_ecef.m_per_s)

        # Calculate subpoint (nadir point).
        target_lat, target_lon = wgs84.latlon_of(geocentric)

        # Create Topos object for the nadir point.
        target_topos = Topos(target_lat.degrees, target_lon.degrees)
        target_ecef = np.array(target_topos.at(ti).frame_xyz(itrs).m)
        logger.debug(f"TARGET lat,lon: {target_lat.degrees}, {target_lon.degrees}")
        logger.debug(f"Target ECEF: {target_ecef}")

        # Calculate the z-vector (satellite to nadir).
        z_vector = target_ecef - position_vector

        # Normalize the z-vector.
        z_axis = z_vector / np.linalg.norm(z_vector)

        # Calculate the x-axis (orthogonalized velocity).
        x_proj = velocity_vector - np.dot(velocity_vector, z_axis) * z_axis
        x_axis = x_proj / np.linalg.norm(x_proj)

        # Calculate the y-axis (cross product of z and x).
        y_axis = np.cross(z_axis, x_axis)

        # Create the rotation matrix.
        rotation_matrix = np.array((x_axis, y_axis, z_axis))
        rotation = Rotation.from_matrix(rotation_matrix)

        # Append position and rotation data.
        pos.append(position_vector)
        rots.append(rotation)

    # Create xarray DataArray for position data with units.
    pos_xr = xr.DataArray(
        np.array(pos) * ureg.m,
        dims=("time", "position"),
        coords=dict(
            time=np.asarray(ts),
            position=["x", "y", "z"],
        ),
    )

    # Create xarray DataArray for rotation data.
    rot_xr = xr.DataArray(
        rots,
        dims=("time"),
        coords=dict(
            time=np.asarray(ts),
        ),
    )

    # Create and return the coordinate system object.
    tlehcs = HCS(pos_xr, rotation=rot_xr, reference=GLOBAL_CS, name=satellite.name)

    return tlehcs
