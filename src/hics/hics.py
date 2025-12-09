"""
hics: Hierarchical Coordinate System Module containing HCS class which stores position and
orientation.
"""

from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
import xarray as xr
from loguru import logger
from pint import Quantity
from scipy.spatial.transform import Rotation, Slerp

from . import ureg
from .datatypes import (
    _POSITION_COORD_DICT,
    _POSITION_COORDS,
    _POSITION_DIM,
    _QUATERNION_COORD_DICT,
    _QUATERNION_COORDS,
    _QUATERNION_DIM,
)
from .geo import HAS_GEO_DEPS
from .geo.transforms import amsl2hagl, geocent2llh
from .utils import vector_norm


class HCSOrigin:
    """
    A wrapper class for Hierarchical Coordinate System origin data, enforcing a standard
    internal representation: a unit-stripped xarray.DataArray in base units.

    Supports positional and rotational data.
    """

    def __init__(self, data: list | np.ndarray | Quantity | xr.DataArray) -> None:
        if isinstance(data, (list, np.ndarray, Quantity)):
            if isinstance(data, Quantity):
                self.original_units = data.units
                basedata = data.to_base_units()
                self.base_units = basedata.units
                data = basedata.magnitude
            elif isinstance(data[0], Quantity):
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
                logger.info(f"Assigning default dims: {dims}")
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

    def interp(self, **kwargs) -> "HCSOrigin":
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

    def testcoords(self, other: Union["HCSOrigin", "HCSRotation"]) -> bool:
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
        pass

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
            logger.info("Broadcasting position")
            vector_da = vector_da.broadcast_like(quat_da.isel(**{_QUATERNION_DIM: 0}).squeeze())
        if not position_dims.issubset(rotation_dims):
            # Broadcast position to match rotation metadata dims
            logger.info("Broadcasting rotation")
            quat_da = quat_da.broadcast_like(vector_da.isel(**{_POSITION_DIM: 0}).squeeze())

        # Align dimensions and flatten
        q = quat_da.transpose(*[d for d in quat_da.dims if d != _QUATERNION_DIM], _QUATERNION_DIM)
        v = vector_da.transpose(*[d for d in vector_da.dims if d != _POSITION_DIM], _POSITION_DIM)

        flat_q = q.data.reshape(-1, len(_QUATERNION_COORDS))
        flat_v = v.data.reshape(-1, len(_POSITION_COORDS))

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

    def interp(self, time) -> "HCSRotation":
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

    def inverse(self) -> "HCSRotation":
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
        return Rotation.from_quat(self.basemag.data.reshape(-1, len(_QUATERNION_COORDS)))

    def __mul__(self, other: "HCSRotation") -> "HCSRotation":
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


class HCS:
    """
    Class for storing hierarchical origin and rotation. Origin and rotation are relative to the
    reference HCS.

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
        reference: Optional["HCS"] = None,
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
    def from_crs(cls, *args, **kwargs) -> "HCS":
        """Alternate constructor using geospatial Coordinate Reference System."""
        raise ImportError

    def clear_cache(self) -> None:
        """Clear cache by initializing private variables."""
        self.__global_position = None
        self._reference_tree = None
        self._origin_tree = None
        self._rotation_tree = None
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
    def reference(self) -> "HCS":
        """Reference HCS of the coordinate system."""
        return self._reference

    @reference.setter
    def reference(self, value: Optional["HCS"] = None) -> None:
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
            if len(origins[:-1][::-1]) == 0:
                if isinstance(pos, HCSOrigin):
                    pos = pos.basemag
            else:
                # Loop through by inverting the rotation and applying and add to the origin
                for o, r in zip(origins[:-1][::-1], rots[:-1][::-1], strict=False):
                    pos = r.apply(pos, inverse=True) + o
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
        if self._reference_tree is None:
            if self.reference is not None and isinstance(self.reference, HCS):
                refs = [self] + self.reference.reference_tree
                self._reference_tree = refs
            else:
                self._reference_tree = [self]
        return self._reference_tree

    @property
    def origin_tree(self) -> list:
        """Get hierarchical list of origins."""
        if self._origin_tree is None:
            if self.reference is not None and isinstance(self.reference, HCS):
                origins = self.reference.origin_tree + [self.origin]
                self._origin_tree = origins
            else:
                self._origin_tree = [self.origin]
        return self._origin_tree

    @property
    def rotation_tree(self) -> list:
        """Get hierarchical list of Rotations."""
        if self._rotation_tree is None:
            if self.reference is not None and isinstance(self.reference, HCS):
                rots = self.reference.rotation_tree + [self.rotation]
                self._rotation_tree = rots
            else:
                self._rotation_tree = [self.rotation]
        return self._rotation_tree

    @property
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
            hagl = amsl2hagl(np.deg2rad(lat), np.deg2rad(lon), h)

            # # Apply units on output
            # if isinstance(hagl, xr.DataArray):
            #     hagl.data *= ureg.meter
            # else:
            #     hagl *= ureg.meter

            # Convert to DataArray if needed
            if isinstance(h_orig, xr.DataArray):
                hagl = xr.DataArray(hagl, dims=h_orig.dims, coords=h_orig.coords, name="HAGL")

            self._hagl = hagl

        return self._hagl

    def interp(self, time: xr.DataArray) -> "HCS":
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

    def relative_position(self, other_hcs: Union["HCS", xr.DataArray]):
        """Determine position of other_hcs in self HCS."""
        if isinstance(other_hcs, HCS):
            oc = other_hcs._global_position.basemag
        else:
            oc = other_hcs

        # Relative position
        r_pos = oc - self._global_position.basemag
        # Note: Need to reverse the order of the rotations
        rots = self.rotation_tree[::-1]
        # Create rotation product
        rot_prod = rots[0]
        for rot in rots[1:]:
            rot_prod *= rot
        # Apply rotation to the relative position
        pos = rot_prod.apply(r_pos)
        # Add units back
        pos = self.origin.apply_units(pos)

        return pos

    def get_relative_rotation(self, request_cs: "HCS") -> xr.DataArray | Rotation:
        """
        Get the rotations from request_cs back to the global and then from global back to self.

        Parameters
        ----------
        request_cs : CS
            Requested coordinate system
        """
        if request_cs == self:
            return Rotation.identity()

        # Get rotations from request_cs to global (reverse and invert)
        req_to_global = [r.inverse() for r in request_cs.rotation_tree[::-1]]

        # Get rotations from global to self
        global_to_self = self.rotation_tree

        # Compose all rotations
        rot_chain = req_to_global + global_to_self
        # Reverse chain order
        rot_chain = rot_chain[::-1]
        composed = rot_chain[0]
        for r in rot_chain[1:]:
            composed = composed * r

        return composed

    def find_common_cs(self, other_hcs: "HCS") -> "HCS":
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

    def relative_distance(self, other_hcs: Union["HCS", xr.DataArray]):
        """Determine distance of other_hcs in self HCS."""
        rel_pos = self.relative_position(other_hcs)
        rel_dist = vector_norm(rel_pos, dim="position")
        rel_dist.name = "Relative Distance"

        return rel_dist


# Create global coorinate system, this is the default
GLOBAL_CS = HCS((0, 0, 0) * ureg.m, reference="GLOBAL", name="Global HCS")
GLOBAL_CS.reference = None
