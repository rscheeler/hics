"""
hics: Hierarchical Coordinate System Module containing HCS class which stores position and
orientation.
"""

from typing import Any, Optional, Union

import numpy as np
import xarray as xr
from loguru import logger
from pint import Quantity, Unit
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
                self._original_unit = data.units
                basedata = data.to_base_units()
                self._base_units = basedata.units
                data = basedata.magnitude
            else:
                data = np.asarray(data)
                self._original_unit = ureg.dimensionless
                self._base_units = ureg.dimensionless

            logger.info(f"Checking origin shape {data.shape}.")

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
                self._original_unit = data.data.units
                basedata = data.data.to_base_units()
                self._base_units = basedata.units
                data.data = basedata.magnitude
            else:
                self._original_unit = ureg.dimensionless
                self._base_units = ureg.dimensionless

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
        _view = self._basemag.copy()
        _view.data = (_view.data * self._base_units).to(self._original_unit)
        return _view

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
            data = xr.DataArray(quat, dims=[_QUATERNION_DIM], coords=_QUATERNION_COORD_DICT)

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
            logger.info("Converting Rotations to quaternions")
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

        self._quat = data

    @property
    def basemag(self) -> xr.DataArray:
        """Return quaternion data in xr.DataArray format."""
        return self._quat

    @property
    def data(self) -> xr.DataArray:
        """Return data as Rotation objects."""
        pass

    def apply(self, origindata: HCSOrigin | xr.DataArray, inverse: bool = False) -> xr.DataArray:
        # Align the data arrays on shared dimensions
        if isinstance(origindata, HCSOrigin):
            origindata = origindata.basemag

        # # Align on shared dimensions
        quat_da, vector_da = xr.align(self.basemag, origindata, join="inner")

        # Check if rotation metadata dims are a subset of position dims
        rotation_dims = set(quat_da.dims) - {_QUATERNION_DIM}
        position_dims = set(vector_da.dims) - {_POSITION_DIM}
        if not rotation_dims.issubset(position_dims):
            # Broadcast position to match rotation metadata dims
            logger.info("Broadcasting position")
            vector_da = vector_da.broadcast_like(quat_da.isel(**{_QUATERNION_DIM: 0}).squeeze())

        # Align dimensions and flatten
        q = quat_da.transpose(*[d for d in quat_da.dims if d != _QUATERNION_DIM], _QUATERNION_DIM)
        v = vector_da.transpose(*[d for d in vector_da.dims if d != _POSITION_DIM], _POSITION_DIM)
        logger.debug(f"Rot q: {q}")
        logger.debug(f"Rot v: {v}")
        flat_q = q.data.reshape(-1, len(_QUATERNION_COORDS))
        flat_v = v.data.reshape(-1, len(_POSITION_COORDS))

        rot = Rotation.from_quat(flat_q)
        rotated = rot.apply(flat_v, inverse=inverse)

        # Reshape back
        rotated = rotated.reshape(v.shape)
        return xr.DataArray(rotated, dims=vector_da.dims, coords=vector_da.coords)

    # def inverse(self) -> "HCSRotation":
    #     rot = self.as_rotation().inv()
    #     inv_quats = rot.as_quat().reshape(self.quat_da.shape)
    #     return HCSRotation(
    #         xr.DataArray(inv_quats, dims=self.quat_da.dims, coords=self.quat_da.coords)
    #     )

    def __mul__(self, other: "HCSRotation") -> "HCSRotation":
        # Align and flatten
        q1, q2 = xr.align(self.basemag, other.basemag, join="outer")

        r1 = Rotation.from_quat(q1.data.reshape(-1, len(_QUATERNION_COORDS)))
        r2 = Rotation.from_quat(q2.data.reshape(-1, len(_QUATERNION_COORDS)))

        composed = r1 * r2
        composed_quats = composed.as_quat().reshape(q1.shape)

        return HCSRotation(xr.DataArray(composed_quats, dims=q1.dims, coords=q1.coords))

    def __repr__(self) -> str:
        if self._data is None:
            return "HCSRotation(None)"
        return f"HCSRotation with coordinates {self._original_unit}:\n{self.unitdata!r}"

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
        logger.debug(f"original units: {self.origin._original_unit}")
        logger.debug(f"orgin data: {self.origin.data}")
        logger.debug(f"orgin base data: {self.origin.basemag}")
        # Validate DataArray
        if (
            _POSITION_DIM not in self.origin.basemag.dims
            and self.origin.basemag[_POSITION_DIM].values != _POSITION_COORDS
        ):
            msg = "Improperly formatted origin data"
            raise ValueError(msg)

        # Default rotation
        if rotation is None:
            logger.debug(self.origin.basemag.sel(position="x"))
            tmp = self.origin.basemag.sel(position="x")
            tmp = tmp.drop_vars("position")
            logger.debug(tmp)
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
        logger.debug(f"orgin data: {self.rotation.data}")
        logger.debug(f"orgin base data: {self.rotation.basemag}")
        # Validate rotation match
        if not self.origin.testcoords(self.rotation):
            msg = "Origin and rotation must have identical coordinates. Coordinate mismatch found."
            raise ValueError(
                msg,
            )
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear cache by initializing private variables."""
        self._slerp = None
        self.__global_position = None
        # self._llh = None
        # self._hagl = None

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
            origins = self.get_origins()
            rots = self.get_rotations()

            # Starting position
            pos = origins[-1]

            # Loop through by inverting the rotation and applying and add to the origin
            for o, r in zip(origins[:-1][::-1], rots[:-1][::-1], strict=False):
                pos = r.apply(pos, inverse=True) + o
            pos.data = pos.data * self.origin._base_units
            self.__global_position = HCSOrigin(pos)

        return self.__global_position

    @property
    def global_position(self) -> xr.DataArray:
        """Determine position in global coordinates."""
        return self._global_position.data

    @property
    def slerp(self) -> Slerp:
        """Slerp works for interpolating rotatations for only a single dimension."""
        if self._slerp is None:
            if isinstance(self.rotation, xr.DataArray):
                if self.rotation.dims != ("time",):
                    msg = "Slerp only support for 'time' dim."
                    raise NotImplementedError(msg)
                # Create single Rotation object with the multiple rotations from the time dimesion
                multi_rotation = Rotation.concatenate(list(self.rotation.values))
                # Converting date time into second timestamp
                ts = (self.rotation.time - self.rotation.time[0]) / np.timedelta64(1, "ns")
                # Generate slerp and convert time to float (ns)
                self._slerp = Slerp(ts, multi_rotation)
            else:
                # Return empty function
                self._slerp = lambda x: self.rotation
        return self._slerp

    def get_references(self):
        """Get hierarchical list of references."""
        if self.reference is not None and isinstance(self.reference, HCS):
            refs = [self] + self.reference.get_references()
            return refs
        refs = [self]
        return refs

    def get_origins(self):
        """Get hierarchical list of origins."""
        if self.reference is not None and isinstance(self.reference, HCS):
            origins = self.reference.get_origins() + [self.origin]
            return origins
        origins = [self.origin]
        return origins

    def get_rotations(self):
        """Get hierarchical list of Rotations."""
        if self.reference is not None and isinstance(self.reference, HCS):
            rots = self.reference.get_rotations() + [self.rotation]
            return rots
        rots = [self.rotation]
        return rots

    def relative_position(self, other_hcs: Union["HCS", xr.DataArray]):
        """Determine position of other_hcs in self HCS."""
        if isinstance(other_hcs, HCS):
            oc = other_hcs._global_position.basemag
        else:
            oc = other_hcs

        # Relative position
        r_pos = oc - self._global_position.basemag

        # Note: Need to reverse the order of the rotations and convert to DataArrays if needed
        rots = self.get_rotations()[::-1]

        rot_prod = rots[0]
        for rot in rots[1:]:
            rot_prod *= rot

        return rot_prod.apply(r_pos)

    def find_common_cs(self, other_hcs: "HCS") -> "HCS":
        """
        Finds a common coordinate system between the two coordinate systems. Returns None if no
        common coordinate system is found.

        Parameters
        ----------
        other_hcs : HCS
            Other coordinate system for determining common coordinate system
        """
        self_refs = self.get_references()
        other_refs = other_hcs.get_references()

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
