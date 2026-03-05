"""hics: Hierarchical Coordinate Systems."""

__version__ = "0.1.0"


# Lazy-loading the core class to avoid circular triggers
from .hics import GLOBAL_CS, HCS
from .units import ureg

__all__ = ["HCS", "GLOBAL_CS", "ureg"]
