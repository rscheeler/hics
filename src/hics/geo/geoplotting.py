import time
from copy import deepcopy
from typing import Optional, Union

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import leafmap.foliumap as leafmap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from leafmap import maplibregl as leaflibre
from loguru import logger
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import PathPatch
from shapely.geometry import LineString

from .. import ureg
from ..hics import HCS
from .dem import DEM, NLCDLEG, nlcdcat2color
from .geoutils import get_surface_profile
from .transforms import GEOD

__all__ = ("view_surface_profile", "view_latlon", "showcs_leafmap", "plotnlcd")

# For animations
plt.rcParams["animation.html"] = "jshtml"


def nlcdcolor(nlcd):
    return list(map(nlcdcat2color, nlcd))


def view_surface_profile(
    tx_cs: HCS,
    rx_cs: HCS,
    aspect: float = 2,
    fill_delta: float = 0.05,
    ax: plt.Axes | None = None,
    **kwargs,
):
    """
    View the surface profile between tx_cs (TX) and rx_cs (RX).

    Parameters
    ----------
    tx_cs : CS
        TX coordinate system to determine profile between.
    rx_cs : CS
        RX coordinate system to determine profile between.
    aspect : float
        Aspect ratio of the plot.
    fill_delta : float
        Amount relative to profile delta to fill below

    **kwargs
    --------
    Additional kwargs to down-select surface profiles with temporal dependence.
    """
    # Get the surface profile
    surface_profile = get_surface_profile(tx_cs, rx_cs)

    # Down-select if kwargs specified
    if len(kwargs) > 0:
        surface_profile = surface_profile.isel(**kwargs)
        for s in ["surface_profile", "clutter_profile", "clutter_nlcd"]:
            surface_profile[s] = surface_profile[s].item()

    # Plot the surface line
    surface_profile.surface_profile.plot(color="k", label="Surface", ax=ax)
    surface_profile.clutter_profile.plot(color="k", lw=1, ls=":", label="Clutter", ax=ax)
    if ax is None:
        ax = plt.gca()

    # Fill below to fill_delta of the delta in altitudes
    below_fill = xr.full_like(
        surface_profile.distance,
        surface_profile.surface_profile.min()
        - (surface_profile.surface_profile.max() - surface_profile.surface_profile.min())
        * fill_delta,
    )
    ax.fill_between(
        surface_profile.distance,
        surface_profile.surface_profile.data,
        below_fill,
        color="k",
        alpha=0.3,
    )

    # Fill with land cover color between earth and the clutter
    colors = nlcdcolor(surface_profile.clutter_nlcd.data)
    for i, c in zip(range(surface_profile.distance.size - 1), colors):
        ax.fill_between(
            surface_profile.distance[i : i + 2],
            surface_profile.clutter_profile.data[i : i + 2],
            surface_profile.surface_profile.data[i : i + 2],
            step="mid",
            color=c,
            alpha=0.3,
        )

    # Plot markers
    ax.plot(
        surface_profile.distance[0],
        surface_profile.txamsl,
        "o",
        mec="C0",
        mfc="w",
        markersize=12,
        label="TX",
    )
    ax.plot(
        surface_profile.distance[-1],
        surface_profile.rxamsl,
        "o",
        mec="C0",
        mfc="w",
        markersize=12,
        label="RX",
    )
    ax.plot(surface_profile.distance[0], surface_profile.txamsl, ".", mec="C0", mfc="C0")
    ax.plot(surface_profile.distance[-1], surface_profile.rxamsl, ".", mec="C0", mfc="C0")
    # Set plot aspect
    aspect = (
        (surface_profile.distance.max() * ureg.m) / surface_profile.clutter_profile.max()
    ) / aspect
    ax.set_aspect(aspect)

    return ax


def view_latlon(
    cs,
    animate=False,
    ax: plt.Axes | None = None,
    extra_extents_deg: list = [0.05, 0.05],
    marker: str = "o",
    markersize=5,
    extent=None,
    fmt_ax=True,
    osm_img=cimgt.GoogleTiles(style="street"),
    **kwargs,
) -> plt.Axes:
    # Make sure marker is a Marker object
    marker = plt.matplotlib.markers.MarkerStyle(marker=marker)

    # Get position in lat/lon
    lat, lon, alt = cs.llh

    if ax is None:
        fig = plt.figure()  # open matplotlib figure
        ax = plt.axes(
            projection=osm_img.crs
        )  # project using coordinate reference system (CRS) of street map
    if extent is None:
        extent = np.array(
            [
                lon.data.min().item() - extra_extents_deg[0],
                lon.data.max().item() + extra_extents_deg[0],
                lat.data.min().item() - extra_extents_deg[1],
                lat.data.max().item() + extra_extents_deg[1],
            ]
        )

    ax.set_extent(extent)  # set extents
    ax.set_xticks(
        np.linspace(extent[0], extent[1], 5), crs=ccrs.PlateCarree()
    )  # set longitude indicators
    ax.set_yticks(
        np.linspace(extent[2], extent[3], 7)[1:], crs=ccrs.PlateCarree()
    )  # set latitude indicators
    lon_formatter = LongitudeFormatter(
        number_format="0.2f", dateline_direction_label=True
    )  # format lons
    lat_formatter = LatitudeFormatter(number_format="0.2f")  # format lats
    ax.xaxis.set_major_formatter(lon_formatter)  # set lons
    ax.yaxis.set_major_formatter(lat_formatter)  # set lats
    ax.set_xlabel("")
    ax.set_ylabel("")
    scale = np.ceil(
        -np.sqrt(2) * np.log(np.divide((extent[1] - extent[0]) / 4.0, 350.0))
    )  # empirical solve for scale based on zoom
    scale = (scale < 20) and scale or 19  # scale cannot be larger than 19
    ax.add_image(osm_img, int(scale))  # add OSM with zoom specification

    # Plot data
    (l,) = ax.plot(
        lon, lat, marker=marker, markersize=markersize, transform=ccrs.PlateCarree(), **kwargs
    )
    rmstart = list(ax.get_lines()).index(l)
    if animate:

        def moving_cs(i, kw=kwargs, remove_idx=rmstart):
            for l in plt.gca().get_lines()[remove_idx:]:
                l.remove()
            if "color" not in kw.keys():
                kw = {**dict(color="C0"), **kw}
            # Plot Line
            plt.plot(lon[: i + 1], lat[: i + 1], transform=ccrs.PlateCarree(), **kw)

            # Rotate marker
            mrk = deepcopy(marker)
            txaz, rxaz, dist = GEOD.inv(lon[i - 1], lat[i - 1], lon[i], lat[i], radians=False)
            mrk._transform = mrk.get_transform().rotate_deg(-txaz)

            # Plot Marker
            line = plt.plot(
                lon[i],
                lat[i],
                markersize=markersize,
                marker=mrk,
                color="w",
                transform=ccrs.PlateCarree(),
            )
            plt.gca().set_title(lon[i].time.data)

            return line

    if animate:
        return animation.FuncAnimation(
            plt.gcf(), moving_cs, save_count=lon.size, interval=10, blit=True
        )
    else:
        return ax


def airplane_marker():
    # Raw points
    ap = np.array(
        [
            [0.02, 0],
            [0, 0.5],
            [0.07, 0.1],
            [0.2, 0.1],
            [0.18, 1.3],
            [0.3, 0.1],
            [0.45, 0.1],
            [0.48, 0],
            [0.45, -0.1],
            [0.3, -0.1],
            [0.18, -1.3],
            [0.2, -0.1],
            [0.07, 0.1],
            [0.07, -0.1],
            [0, -0.5],
            [0.02, 0],
        ]
    )
    # Scaling
    ap[:, 0] -= (ap[:, 0].max() - ap[:, 0].min()) / 2
    ap[:, 0] *= 5
    # Create marker
    ap = plt.matplotlib.markers.MarkerStyle(marker=ap)
    # Rotate to point north
    ap._transform = ap.get_transform().rotate_deg(90)

    return ap


def plotnlcd(pts: list | None = None):
    """
    Plot NLCD data with correct colors.
    """
    # colormap determination and setting bounds
    indx_list = np.unique(DEM.nlcd.data)
    r_cmap = NLCDLEG.loc[NLCDLEG["Value"].isin(indx_list)]["rgbint"].values
    raster_cmap = ListedColormap(r_cmap)  # defining the NLCD specific color map
    # Have to add an item to boundary norm boundaries to get correct colorbar
    norm = matplotlib.colors.BoundaryNorm(
        list(indx_list) + [255], raster_cmap.N
    )  # specifying colors based on num. unique points

    # Plot with colormap
    fig, ax = plt.subplots()
    if pts:
        nlcd = DEM.nlcd.sel(
            lon=slice(np.deg2rad(pts[0][1]), np.deg2rad(pts[1][1])),
            lat=slice(np.deg2rad(pts[0][0]), np.deg2rad(pts[1][0])),
        ).copy()
    else:
        nlcd = DEM.nlcd.copy()
    # Convert quantities
    lon = (nlcd.lon.data * ureg.radian).to("degree").magnitude
    lat = (nlcd.lat.data * ureg.radian).to("degree").magnitude
    nlcd = nlcd.assign_coords(lon=lon, lat=lat)
    # Plot
    nlcd.plot(cmap=raster_cmap, norm=norm, ax=ax)
    ax.set_aspect("equal")

    # Format axis labels
    lon_formatter = LongitudeFormatter(
        number_format="0.2f", dateline_direction_label=True
    )  # format lons
    lat_formatter = LatitudeFormatter(number_format="0.2f")  # format lats
    ax.xaxis.set_major_formatter(lon_formatter)  # set lons
    ax.yaxis.set_major_formatter(lat_formatter)  # set lats

    # Label colors
    # Assume colorbar was plotted last one plotted last
    cb = fig.axes[-1]

    labels = NLCDLEG.loc[NLCDLEG["Value"].isin(indx_list)]["Name"].values
    cb.set_yticks(indx_list)
    cb.set_yticklabels(labels, verticalalignment="bottom")
    cb.set_ylabel("")

    return ax


def cs2geodf(cs):
    hamsl = cs.llh[2]
    if hamsl.shape == ():
        hamsl = xr.DataArray(
            [hamsl.data.magnitude] * hamsl.data.units, dims=("time",), coords=dict(time=[0])
        )
        lat = xr.DataArray(
            [cs.llh[0].data.magnitude] * cs.llh[0].data.units, dims=("time",), coords=dict(time=[0])
        )
        lon = xr.DataArray(
            [cs.llh[1].data.magnitude] * cs.llh[1].data.units, dims=("time",), coords=dict(time=[0])
        )

        data = xr.Dataset(dict(hamsl=hamsl))
        data = data.assign_coords(dict(lat=lat, lon=lon, name=cs.name))
    else:
        data = xr.Dataset(dict(hamsl=hamsl))
        data = data.assign_coords(dict(lat=cs.llh[0], lon=cs.llh[1], name=cs.name))

    points_gdf = gpd.GeoDataFrame(
        data.to_dataframe().reset_index(),
        geometry=gpd.points_from_xy(data.lon, data.lat),
        crs="EPSG:4326",
    )
    # Create a GeoDataFrame with the polygon geometry
    if data.lon.shape == (1,):
        line_gdf = None
    else:
        line_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(geom.x, geom.y) for geom in points_gdf.geometry])],
            crs=points_gdf.crs,
        )
    return points_gdf, line_gdf


def point_json(x, y):
    return {
        "type": "Point",
        "coordinates": [x, y],
    }


def showcs_leafmap(
    cs,
    animate: bool = False,
    zoom: int = 7,
    tiles: str = "cartodb positron",
    line_style: dict | None = None,
    style: str = "positron",
    pitch: float = 30,
    paint: dict | None = None,
    m: leafmap.Map | leaflibre.Map | None = None,
    **kwargs,
):
    gdf_pts, gdf_line = cs2geodf(cs)
    center = (gdf_pts.geometry.y.mean().item(), gdf_pts.geometry.x.mean().item())

    if m is None:
        if animate:
            m = leaflibre.Map(center=center[::-1], zoom=zoom, style=style, pitch=pitch, **kwargs)
        else:
            m = leafmap.Map(center=center, zoom=zoom, tiles=tiles, **kwargs)

    if animate:
        source = {
            "type": "geojson",
            "data": point_json(gdf_pts.iloc[0].geometry.x, gdf_pts.iloc[0].geometry.y),
        }
        m.add_source("point", source)
        layer = {
            "id": "point",
            "source": "point",
            "type": "circle",
            "paint": {"circle-radius": 8, "circle-color": "#007cbf"},
        }
        if paint is None:
            paint = {"line-color": "#ff7f0e", "line-width": 5, "line-opacity": 0.8}
        if gdf_line is not None:
            m.add_gdf(gdf_line, layer_type="line", paint=paint)
        m.add_layer(layer)

        def leafmap_animator(run_times: int = 2, speed: float = 0.05):
            for i in range(run_times):
                for row in gdf_pts.itertuples():
                    time.sleep(speed)
                    m.set_data("point", point_json(row.geometry.x, row.geometry.y))

        return m, leafmap_animator

    else:
        m.add_gdf(gdf_pts, zoom_to_layer=False)
        if line_style is None:
            line_style = {"fillColor": "none", "color": "#ff7f0e", "weight": 5, "opacity": 0.8}
        if gdf_line is not None:
            m.add_gdf(gdf_line, style=line_style, zoom_to_layer=False)
        return m
