"""hics: Hierarchical Coordinate Systems."""

__version__ = "0.1.0"
from pint import get_application_registry

# --- Pint unit registry ---
# Get global unit registry
ureg = get_application_registry()

# Settings
ureg.autoconvert_offset_to_baseunit = True
ureg.force_ndarray_like = True

# Lazy-loading the core class to avoid circular triggers
from .hics import GLOBAL_CS, HCS

__all__ = ["HCS", "GLOBAL_CS", "ureg"]
