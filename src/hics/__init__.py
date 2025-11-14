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
from .hics import GLOBAL_CS, HCS

# --- CONDITIONAL GEO PATCHING ---
try:
    # 1. Check for required geo dependencies (pyproj, rioxarray, etc.)
    import pyproj
    import rioxarray

    # 2. If successful, import the geo-specific functions
    from .geo.crs import from_crs

    # 3. Dynamically patch the HCS class with the classmethod
    # Remember to rename CS to HCS in coordinate_system.py first!
    HCS.from_crs = classmethod(from_crs)
    # Define a flag for internal package use
    HAS_GEO_DEPS = True
    logger.info("Geo dependencies installed.")

except ValueError:
    # If any geo dependency is missing, do nothing, and the method isn't attached.
    HAS_GEO_DEPS = False
    logger.info("Geo dependencies not installed.")

# --- EXPORTS ---
__all__ = [
    "GLOBAL_CS",
    "HAS_GEO_DEPS",
    "HCS",
    # ... all other core exports ...
]
