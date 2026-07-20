"""
hics: Hierarchical Coordinate System Module containing HCS class which stores position and
orientation.
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from functools import reduce
from operator import mul
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation, Slerp
from xrench.units import ureg
from xrench.xrutils import vector_norm, wraps_xr

from .datatypes import (
    _POSITION_COORD_DICT,
    _POSITION_COORDS,
    _POSITION_DIM,
    _QUATERNION_COORD_DICT,
    _QUATERNION_COORDS,
    _QUATERNION_DIM,
)
from .geo import HAS_GEO_DEPS

if HAS_GEO_DEPS:
    from .geo.crs import from_crs
    from .geo.dem import amsl2hagl, geocent2llh
if TYPE_CHECKING:
    from pint import Quantity

_wamsl2hagl = wraps_xr(ureg.m, (ureg.radian, ureg.radian, ureg.m))(amsl2hagl)


class HCSOrigin:
    """
    A wrapper class for Hierarchical Coordinate System origin data, enforcing a standard
    internal representation: a unit-stripped xarray.DataArray in base units.

    Supports positional and rotational data.
    """

    def __init__(self, data: list | np.ndarray | Quantity | xr.DataArray) -> None:
        if isinstance(data, (list, np.ndarray, ureg.Quantity)):
            if isinstance(data, ureg.Quantity):
                self.original_units = data.units
                basedata = data.to_base_units()
                self.base_units = basedata.units
                data = basedata.magnitude
            elif isinstance(data[0], ureg.Quantity):
                self.original_units = data[0].units
                self.base_units = data[0].to_base_units().units
                basedata = [d.to_base_units().magnitude for d in data]
                data = np.asarray(basedata)
            else:
                data = np.asarray(data)
                self.original_units = ureg.dimensionless
                self.base_units = ureg.dimensionless

            if len(_POSITION_COORDS) in data.shape:
                # Make dims
                dims = [f"dim{i}" for i in range(len(data.shape))]
                dims[data.shape.index(len(_POSITION_COORDS))] = _POSITION_DIM
                data = xr.DataArray(data, dims=dims, coords=_POSITION_COORD_DICT)
            else:
                raise ValueError(f"Data shape not compatible: {data.shape}")

        elif isinstance(data, xr.DataArray):
            if hasattr(data.data, "units"):
                self.original_units = data.data.units
                basedata = data.data.to_base_units()
                self.base_units = basedata.units
                data.data = basedata.magnitude
            else:
                self.original_units = ureg.dimensionless
                self.base_units = ureg.dimensionless

        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        self._basemag = data

    @property
    def basemag(self) -> xr.DataArray:
        """Returns the internal, unit-stripped xarray.DataArray in base units."""
        return self._basemag

    @property
    def data(self) -> xr.DataArray:
        """Returns the internal data with the original unit reapplied."""
        return self.apply_units(self._basemag.copy())

    def interp(self, **kwargs) -> HCSOrigin:
        """Interpolate basedata."""
        newdata = self.apply_units(self._basemag.interp(**kwargs))
        return HCSOrigin(newdata)

    def apply_units(self, magdata: xr.DataArray) -> xr.DataArray:
        """Applys original_units to magnitude data assuming same base_units.

        Parameters
        ----------
        magdata : xr.DataArray
            Magnitude data

        Returns:
        -------
        xr.DataArray
            Unit full data
        """
        magdata.data = (magdata.data * self.base_units).to(self.original_units)
        return magdata

    def testcoords(self, other: HCSOrigin | HCSRotation) -> bool:
        """Test non-positional coordinates with another HCSOrigin."""
        if _POSITION_DIM in self.basemag.dims:
            selfcoords = self.basemag.drop_vars(_POSITION_DIM).coords
        else:
            selfcoords = self.basemag.coords

        if _POSITION_DIM in other.basemag.dims:
            othercoords = other.basemag.drop_vars(_POSITION_DIM).coords
        elif _QUATERNION_DIM in other.basemag.dims:
            othercoords = other.basemag.drop_vars(_QUATERNION_DIM).coords
        else:
            othercoords = other.basemag.coords

        return selfcoords.equals(othercoords)

    def __repr__(self) -> str:
        if self._basemag is None:
            return "HCSOrigin(None)"
        return f"HCSOrigin \n{self.data!r}"

    def __getattr__(self, name: str) -> Any:
        if self._basemag is not None and hasattr(self._basemag, name):
            return getattr(self._basemag, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __deepcopy__(self, memo) -> HCSOrigin:
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = deepcopy(v, memo)
        return new


class HCSRotation:
    """
    A wrapper class for Hierarchical Coordinate System rotation data, enforcing a standard
    internal representation: quaternions.

    Supports rotational data, including Rotation objects and quaternions.
    """

    def __init__(
        self,
        data: list | np.ndarray | Quantity | xr.DataArray | Rotation | list[Rotation],
    ) -> None:
        # Handle Rotation object or list of them
        if isinstance(data, Rotation):
            quat = data.as_quat()
            dims = [f"dim{i}" for i in range(len(quat.shape))]
            dims[quat.shape.index(len(_QUATERNION_COORDS))] = _QUATERNION_DIM
            data = xr.DataArray(quat, dims=dims, coords=_QUATERNION_COORD_DICT)

        elif isinstance(data, (list, np.ndarray)) and isinstance(data[0], Rotation):
            quats = np.array([r.as_quat() for r in np.array(data).ravel()]).reshape(
                list(np.array(data).shape) + [len(_QUATERNION_COORDS)]
            )
            dims = [f"dim{i}" for i in range(len(quats.shape))]
            dims[quats.shape.index(len(_QUATERNION_COORDS))] = _QUATERNION_DIM
            data = xr.DataArray(quats, dims=dims, coords=_QUATERNION_COORD_DICT)
        elif isinstance(data, xr.DataArray) and _QUATERNION_DIM in data.dims:
            pass
        elif isinstance(data, xr.DataArray) and isinstance(data.data.ravel()[0], Rotation):
            datar = data.data.ravel()
            quats = np.array([r.as_quat() for r in datar]).reshape(
                list(data.shape) + [len(_QUATERNION_COORDS)]
            )
            data = xr.DataArray(
                quats,
                dims=list(data.dims) + [_QUATERNION_DIM],
                coords={**data.coords, **_QUATERNION_COORD_DICT},
            )
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        # Make sure last dim is the quaternion
        data = data.transpose(*[d for d in data.dims if d != _QUATERNION_DIM], _QUATERNION_DIM)
        self._quat = data
        self._slerp = None

    @property
    def basemag(self) -> xr.DataArray:
        """Return quaternion data in xr.DataArray format."""
        return self._quat

    @property
    def data(self) -> xr.DataArray:
        """Return data as Rotation objects."""
        raise NotImplementedError

    def apply(self, origindata: HCSOrigin | xr.DataArray, inverse: bool = False) -> xr.DataArray:
        """
        Apply rotation to origin data which can be and HCSOrigin or xr.DataArray.

        Parameters
        ----------
        origindata : HCSOrigin | xr.DataArray
            Data to apply rotation to
        inverse : bool, optional
            Wheter to invert the rotation

        Returns:
        -------
        xr.DataArray
            Rotated data.
        """
        # Align the data arrays on shared dimensions
        if isinstance(origindata, HCSOrigin):
            origindata = origindata.basemag

        # Align on shared dimensions
        quat_da, vector_da = xr.align(self.basemag, origindata, join="inner")

        # Check if rotation metadata dims are a subset of position dims
        rotation_dims = set(quat_da.dims) - {_QUATERNION_DIM}
        position_dims = set(vector_da.dims) - {_POSITION_DIM}
        if not rotation_dims.issubset(position_dims):
            # Broadcast position to match rotation metadata dims
            vector_da = vector_da.broadcast_like(quat_da.isel(**{_QUATERNION_DIM: 0}).squeeze())
        if not position_dims.issubset(rotation_dims):
            # Broadcast position to match rotation metadata dims
            quat_da = quat_da.broadcast_like(vector_da.isel(**{_POSITION_DIM: 0}).squeeze())

        # Align dimensions and flatten
        q = quat_da.transpose(*[d for d in quat_da.dims if d != _QUATERNION_DIM], _QUATERNION_DIM)
        v = vector_da.transpose(*[d for d in vector_da.dims if d != _POSITION_DIM], _POSITION_DIM)

        flat_q = np.array(q.data).reshape(-1, len(_QUATERNION_COORDS))
        flat_v = np.array(v.data).reshape(-1, len(_POSITION_COORDS))

        rot = Rotation.from_quat(flat_q)
        rotated = rot.apply(flat_v, inverse=inverse)

        # Reshape back
        rotated = rotated.reshape(v.shape)
        return xr.DataArray(rotated, dims=v.dims, coords=v.coords)

    @property
    def slerp(self) -> Slerp:
        """Slerp works for interpolating rotatations for only a single dimension."""
        if self._slerp is None:
            if self.basemag.dims != ("time", _QUATERNION_DIM):
                msg = "Slerp only support for 'time' dim."
                raise NotImplementedError(msg)
            # Create single Rotation object with the multiple rotations from the time dimension
            multi_rotation = self.as_multirotation()
            # Converting date time into second timestamp
            ts = (self.basemag.time - self.basemag.time[0]) / np.timedelta64(1, "ns")
            # Generate slerp and convert time to float (ns)
            self._slerp = Slerp(ts, multi_rotation)

        return self._slerp

    def interp(self, time) -> HCSRotation:
        """Interpolate self using slerp, only works for time."""
        # Interpolate
        tdelta = (time - time[0]) / np.timedelta64(1, "ns")
        new_rot_quats = self.slerp(tdelta).as_quat()

        return HCSRotation(
            xr.DataArray(
                new_rot_quats,
                dims=("time", _QUATERNION_DIM),
                coords={**dict(time=time), **_QUATERNION_COORD_DICT},
            )
        )

    def inverse(self) -> HCSRotation:
        """Return inverse rotation as HCSRotation object."""
        rot = self.as_multirotation().inv()
        inv_quats = rot.as_quat().reshape(self.basemag.shape)
        return HCSRotation(
            xr.DataArray(inv_quats, dims=self.basemag.dims, coords=self.basemag.coords)
        )

    def as_multirotation(self) -> Rotation:
        """
        Returns self as a flattened Rotation object.

        Returns:
        -------
        Rotation
            Flattened rotation object.
        """
        return Rotation.from_quat(np.array(self.basemag.data).reshape(-1, len(_QUATERNION_COORDS)))

    def __mul__(self, other: HCSRotation) -> HCSRotation:
        # Align and flatten
        q1, q2 = xr.broadcast(self.basemag, other.basemag)

        q1 = q1.transpose(*[d for d in q1.dims if d != _QUATERNION_DIM], _QUATERNION_DIM)
        q2 = q2.transpose(*[d for d in q2.dims if d != _QUATERNION_DIM], _QUATERNION_DIM)
        r1 = Rotation.from_quat(q1.data.reshape(-1, len(_QUATERNION_COORDS)))
        r2 = Rotation.from_quat(q2.data.reshape(-1, len(_QUATERNION_COORDS)))

        composed = r1 * r2
        composed_quats = composed.as_quat().reshape(q1.shape)

        return HCSRotation(xr.DataArray(composed_quats, dims=q1.dims, coords=q1.coords))

    def __repr__(self) -> str:
        if self._quat is None:
            return "HCSRotation(None)"
        return f"HCSRotation with coordinates\n{self.basemag!r}"

    def __getattr__(self, name: str) -> Any:
        if self._quat is not None and hasattr(self._quat, name):
            return getattr(self._quat, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __deepcopy__(self, memo) -> HCSRotation:
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = deepcopy(v, memo)
        return new


class HCS:
    """
    Class for storing hierarchical origin and rotation. Origin and rotation are relative to the
    reference HCS.

    Note on scipy rotation chains:
    >>> from scipy.spatial.transform import Rotation
    >>> # Define a rotated coordinate system that is rotated 90 degrees in z so the local coordinates x-axis now points in global y-axis
    >>> rotz90 = Rotation.from_euler("Z", 90, degrees=True)
    >>> # returns [0,1,0] Coordinate frame transform: Moves x-axis to y-axis describes local x in global coordinates
    >>> print(rotz90.apply([1, 0, 0]))
    >>> # returns [0,-1,0] Vector transform - describes global x-axis in local coordinates
    >>> print(rotz90.apply([1, 0, 0], inverse=True))
    >>> # Chained, now a y-axis rotation is added, when chained z-axis now points in global y
    >>> # Chains are not commutative so order matters
    >>> # Chain is built from global to self (left to right)
    >>> roty90 = Rotation.from_euler("Y", 90, degrees=True)
    >>> rot = rotz90 * roty90
    >>> # returns [0,1,0] Coordinate frame transform: Moves z-axis to y-axis
    >>> print(rot.apply([0, 0, 1]))
    >>> # returns [-1,0,0] Vector transform - describes global z-axis in frame coordinates
    >>> print(rot.apply([0, 0, 1], inverse=True))

    Parameters
    ----------
    origin : xr.DataArray, Quantity
        Origin data can be specified as a Quantity (x,y,z), or a DataArray with dim "position" and
        coords "x","y","z"
    rotation : xr.DataArray, scipy.spatial.transform.Rotation
        Rotation object
    reference : HCS
        Reference coordinate system for origin and rotation.
    """

    def __init__(
        self,
        origin: xr.DataArray | Quantity,
        rotation: xr.DataArray | Rotation | None = None,
        reference: HCS | None = None,
        name: str | None = "Untitled",
    ) -> None:
        self.name = name
        # Grab reference
        if reference is None:
            reference = GLOBAL_CS
        self._reference = reference

        # Store origin and rotation in data objects which will validate inputs
        self.origin = origin

        # Validate DataArray
        if (
            _POSITION_DIM not in self.origin.basemag.dims
            and self.origin.basemag[_POSITION_DIM].values != _POSITION_COORDS
        ):
            msg = "Improperly formatted origin data"
            raise ValueError(msg)

        # Default rotation
        if rotation is None:
            tmp = self.origin.basemag.sel(position="x")
            tmp = tmp.drop_vars("position")
            if tmp.dims != ():
                # Can't use full like because it copies the object
                rotation = xr.DataArray(
                    np.array([Rotation.identity() for i in range(tmp.size)]).reshape(tmp.shape),
                    dims=tmp.dims,
                    coords=tmp.coords,
                )
            else:
                rotation = Rotation.identity()

        self.rotation = rotation
        # Validate rotation match
        if not self.origin.testcoords(self.rotation):
            msg = "Origin and rotation must have identical coordinates. Coordinate mismatch found."
            raise ValueError(
                msg,
            )
        self.clear_cache()

    @classmethod
    def from_crs(cls, *args, **kwargs) -> HCS:
        """Alternate constructor using geospatial Coordinate Reference System."""
        raise ImportError

    def clear_cache(self) -> None:
        """Clear cache by initializing private variables."""
        self.__global_position = None
        self._llh = None
        self._hagl = None

    @property
    def origin(self) -> HCSOrigin:
        """Origin of the coordinate system."""
        return self._origin

    @origin.setter
    def origin(self, new_origin: Any | HCSOrigin) -> None:
        """Origin setter."""
        # Always wrap input in HCSOrigin using the expected unit (meter)
        if isinstance(new_origin, HCSOrigin):
            self._origin = new_origin
        else:
            self._origin = HCSOrigin(new_origin)
        self.clear_cache()

    @property
    def rotation(self) -> HCSRotation:
        """Orientation of the coordinate system relative to the reference."""
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation: Any | HCSRotation) -> None:
        """Rotation setter."""
        # Always wrap input in HCSRotation using the expected unit (dimensionless)
        if isinstance(new_rotation, HCSRotation):
            self._rotation = new_rotation
        else:
            self._rotation = HCSRotation(new_rotation)
        self.clear_cache()

    @property
    def reference(self) -> HCS:
        """Reference HCS of the coordinate system."""
        return self._reference

    @reference.setter
    def reference(self, value: HCS | None = None) -> None:
        """Reference HCS setter."""
        if isinstance(value, HCS) or value is None:
            self._reference = value
            # Reinitialize
            self.clear_cache()
        else:
            msg = "Reference must be a HCS object or None."
            raise ValueError(msg)

    @property
    def _global_position(self) -> HCSOrigin:
        """Determine position in global coordinates as HCSOrigin object."""
        if self.__global_position is None:
            # Get origins and rotations
            origins = self.origin_tree
            rots = self.rotation_tree

            # Starting position
            pos = origins[-1]

            # If no hierarchy need to return data properly
            if len(origins[-2::-1]) == 0:
                if isinstance(pos, HCSOrigin):
                    pos = pos.basemag
            else:
                # Loop through by applying the rotation and adding to the origin
                for o, r in zip(origins[-2::-1], rots[-2::-1], strict=False):
                    pos = r.apply(pos) + o
            pos = self.origin.apply_units(pos)

            self.__global_position = HCSOrigin(pos)

        return self.__global_position

    @property
    def global_position(self) -> xr.DataArray:
        """Determine position in global coordinates."""
        return self._global_position.data

    @property
    def reference_tree(self) -> list:
        """Get hierarchical list of references."""
        if self.reference is not None and isinstance(self.reference, HCS):
            return [self] + self.reference.reference_tree

        return [self]

    @property
    def origin_tree(self) -> list:
        """Get hierarchical list of origins."""
        if self.reference is not None and isinstance(self.reference, HCS):
            return self.reference.origin_tree + [self.origin]
        return [self.origin]

    @property
    def rotation_tree(self) -> list:
        """Get hierarchical list of Rotations."""
        if self.reference is not None and isinstance(self.reference, HCS):
            return self.reference.rotation_tree + [self.rotation]
        return [self.rotation]

    @property
    @wraps_xr((ureg.degree, ureg.degree, ureg.m), None)
    def llh(self):
        """Return position in latitude, longitude, and height."""
        if self._llh is None and HAS_GEO_DEPS:
            gp = self.global_position
            llh = geocent2llh.transform(
                gp.sel(position="x"), gp.sel(position="y"), gp.sel(position="z")
            )
            # Make sure to drop position dimension
            _llh = []
            for l in llh:
                _llh.append(l.drop_vars("position"))
            self._llh = _llh
        return deepcopy(self._llh)

    @property
    def hagl(self):
        """Return height above ground level (HAGL)."""
        if self._hagl is None and HAS_GEO_DEPS:
            # Get llh
            lat, lon, h = self.llh
            h_orig = deepcopy(h)

            # Get hagl
            hagl = _wamsl2hagl(lat, lon, h)

            # Convert to DataArray if needed
            if isinstance(h_orig, xr.DataArray):
                hagl = xr.DataArray(hagl, dims=h_orig.dims, coords=h_orig.coords, name="HAGL")

            self._hagl = hagl
        return self._hagl

    def interp(self, time: xr.DataArray) -> HCS:
        """
        Interpolates and returns new HCS object. Only works for time dimension.

        Parameters
        ----------
        times : xr.DataArray
        """
        # Update positions
        neworigin = self.origin.interp(time=time)

        # Reset self.rotation
        newrotation = self.rotation.interp(time=time)

        return HCS(neworigin, newrotation)

    def isel(self, missing_dims="ignore", **kwargs) -> HCS:
        """Index-select along a dimension, returning a new HCS."""
        return self._xrfunc("isel", missing_dims=missing_dims, **kwargs)

    def sel(self, missing_dims="ignore", **kwargs) -> HCS:
        """Select along a dimension, returning a new HCS."""
        return self._xrfunc("sel", missing_dims=missing_dims, **kwargs)

    def _xrfunc(self, func, **kwargs) -> HCS:
        """Apply function to HCSOrigin and HCSRotation basemag data."""
        o_func = getattr(self.origin.basemag, func)
        r_func = getattr(self.rotation.basemag, func)
        new_origin = HCSOrigin(o_func(**kwargs))
        new_origin.original_units = self.origin.original_units
        new_origin.base_units = self.origin.base_units
        new_rotation = HCSRotation(r_func(**kwargs))
        return HCS(new_origin, new_rotation, reference=self.reference, name=self.name)

    def relative_position_basemag(self, other_hcs: HCS | xr.DataArray) -> xr.DataArray:
        """Determine position of other_hcs in self HCS in base magnitude."""
        if isinstance(other_hcs, HCS):
            oc = other_hcs._global_position.basemag
        else:
            oc = other_hcs

        # Relative position in global coordinates
        r_pos = oc - self._global_position.basemag

        # Each rotation in rotation_tree is parent to local frame transform
        # Composing left-to-right gives you global to self frame transform
        rot = reduce(mul, self.rotation_tree)
        # Position in global to self is the inverse of the rotation
        pos = rot.apply(r_pos, inverse=True)
        return pos

    def relative_position(self, other_hcs: HCS | xr.DataArray):
        """Determine position of other_hcs in self HCS."""
        # Basemag relative position
        pos = self.relative_position_basemag(other_hcs)
        # Add units back
        pos = self.origin.apply_units(pos)

        return pos

    def get_relative_rotation(self, request_cs: HCS) -> HCSRotation:
        """
        Get the rotations from request_cs back to the global and then from global back to self.

        To transform a point defined in request_cs into self's frame, apply the inverse of this rotation:
            r = self.get_relative_rotation(request_cs)
            v_in_self = r.apply(v_in_request, inverse=True)

        Parameters
        ----------
        request_cs : CS
            Requested coordinate system
        """
        if request_cs == self:
            return HCSRotation(Rotation.identity())

        # Get rotations from global to self
        rot_g2s = reduce(mul, self.rotation_tree)

        # Get rotations from request_cs to global, inverting each step
        rot_r2g = reduce(mul, [r.inverse() for r in request_cs.rotation_tree[::-1]])

        return rot_r2g * rot_g2s

    def find_common_cs(self, other_hcs: HCS) -> HCS:
        """
        Finds a common coordinate system between the two coordinate systems. Returns None if no
        common coordinate system is found.

        Parameters
        ----------
        other_hcs : HCS
            Other coordinate system for determining common coordinate system
        """
        self_refs = self.reference_tree
        other_refs = other_hcs.reference_tree

        # Loop through self and other references to find a common reference
        common = None
        for sref in self_refs:
            if common is not None:
                break
            for oref in other_refs:
                if sref == oref:
                    common = sref
                    break
        return common

    def relative_distance(self, other_hcs: HCS | xr.DataArray):
        """Determine distance of other_hcs in self HCS."""
        rel_pos = self.relative_position(other_hcs)
        rel_dist = vector_norm(rel_pos, dim="position")
        rel_dist.name = "Relative Distance"

        return rel_dist

    def save(self, filename: Path):
        with open(Path(filename).with_suffix(".hics"), "wb") as file:
            # Serialize the data and write it to the file
            pickle.dump(self, file)


# Create global coorinate system, this is the default
GLOBAL_CS = HCS((0, 0, 0) * ureg.m, reference="GLOBAL", name="Global HCS")
GLOBAL_CS.reference = None
# Patch from_crs if geo dependencies exist
if HAS_GEO_DEPS:
    HCS.from_crs = classmethod(from_crs)
