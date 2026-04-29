"""hics: Hierarchical Coordinate Systems."""

__version__ = "0.1.0"


# Lazy-loading the core class to avoid circular triggers
from xrench.units import ureg

from .config import HICSLogger
from .hics import GLOBAL_CS, HCS

__all__ = ["HCS", "GLOBAL_CS", "ureg", "HICSLogger"]
