"""hics: Hierarchical Coordinate Systems."""

__version__ = "0.1.0"
# --- CORE IMPORTS ---
from loguru import logger
from pint import get_application_registry

# --- Pint unit registry ---
# Get global unit registry
ureg = get_application_registry()

# Settings
ureg.autoconvert_offset_to_baseunit = True
ureg.force_ndarray_like = True

# Import after ureg to avoid circular import issue
from .geo import HAS_GEO_DEPS
from .hics import GLOBAL_CS, HCS

# --- CONDITIONAL GEO PATCHING ---
if HAS_GEO_DEPS:
    # If successful, import the geo-specific functions
    from .geo.crs import from_crs

    # Dynamically patch the HCS class with the classmethod
    HCS.from_crs = classmethod(from_crs)

# --- EXPORTS ---
__all__ = [
    "GLOBAL_CS",
    "HAS_GEO_DEPS",
    "HCS",
    # ... all other core exports ...
]
