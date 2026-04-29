# :map: hics: Hierarchical Coordinate Systems
```hics``` is a Python package for handling hierarchical coordinate systems (HCS). It allows you to define relative transformations (translation and rotation) between frames and automatically resolves them to global positions (ECEF) or relative positions between any two frames in the tree.

## Core Concepts
### 1. Defining a Coordinate System
You can define a coordinate system using a position tuple and an optional reference frame. If no reference is provided, it defaults to the ```GLOBAL_CS```.

```python
from hics import HCS, ureg
import numpy as np

# A simple coordinate system 5 meters above the global origin
roof = HCS((0, 0, 5) * ureg.meter)

# A coordinate system relative to 'roof'
antenna = HCS((0, 0, 2) * ureg.meter, reference=roof)

# Resolves to [0, 0, 7]
print(antenna.global_position)
```
### 2. Rotations and Compound Transformations
```hics``` integrates with ```scipy.spatial.transform.Rotation```. Transformations are chained automatically down the hierarchy.

```python
from scipy.spatial.transform import Rotation

# Define a tower rotated 90 degrees on the roof
tower = HCS(
    (0, 4, 5) * ureg.meter,
    rotation=Rotation.from_euler("ZYZ", (0, 90, 0), degrees=True),
    reference=roof,
)

# Define an antenna on that rotated tower
local_ant = HCS(
    (0, 6, 5) * ureg.meter,
    reference=tower,
)
```
### 3. Relative Calculations
You can calculate the position or distance of one frame as seen from another, even if they are in different branches of the hierarchy.

```python
# How far is the antenna from the tower?
dist = local_ant.relative_distance(tower)

# What is the roof's position in the antenna's coordinate frame?
# This returns the roof's coordinates relative to local_ant's origin and rotation.
rel_pos = local_ant.relative_position(roof)
```
## Geospatial Integration
```hics``` includes powerful utilities for mapping coordinate systems to real-world paths using Digital Elevation Models (DEM) and Land Cover data.
### 1. Creating a CS from Lat/Lon/Alt
You can initialize a coordinate system directly from geographic coordinates. If ```hagl=True```, the altitude is treated as "Height Above Ground Level" and added to the terrain elevation at that point.

```python
# Initialize an HCS at a specific location in Boulder, CO
# 20 meters above the ground (HAGL)
cs_boulder = HCS.from_crs(
    (40.015 * ureg.degree, -105.270556 * ureg.degree, 20 * ureg.m), 
    hagl=True
)

# This resolves to a global ECEF position
print(cs_boulder.global_position)
```
### 2. Racetrack and Path Interpolation
You can generate a coordinate system that follows a trajectory (like a vehicle or aircraft) defined by latitude/longitude points.

```python
from hics.geo.scenarios import interp_llpnts2hcs, racetrack_latlon

# Generate a 10km altitude racetrack path
center = (40.037578, -105.228117) * ureg.degree
racetrack_pnts = racetrack_latlon(center[0], center[1], length=8.48*ureg.km, width=1.5*ureg.km)

racetrack = interp_llpnts2hcs(
    racetrack_pnts, 
    altitude=10*ureg.km, 
    speed=100*ureg.m/ureg.s, 
    bank_turns=True
)
```

### 3. Height Above Ground Level (HAGL)
If ```hagl=True``` is specified, hics uses the underlying DEM data to "pin" the coordinate system to the terrain height.

```python
# A car driving 2 meters above the ground following a path
drivecs = interp_llpnts2hcs(
    ll_path_points, 
    hagl=2*ureg.m, 
    speed=30*ureg.mph, 
    hagl=True
)
```

## Advanced Usage: Xarray and Lazy Loading
```hics``` leverages ```xarray``` and ```dask``` for performance.

* Lazy Merging: Large terrain datasets are merged lazily via VRT (Virtual Raster) files to save memory.
* Time-Series Support: Coordinate systems can be defined over time-indexed DataArrays, allowing for easy interpolation of moving objects.

## References

- [Cookiecutter Python Project](https://github.com/wyattferguson/pattern) - A modern cookiecutter template for your next Python project.
