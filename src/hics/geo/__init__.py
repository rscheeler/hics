import importlib.util

# --- CONDITIONAL GEO PATCHING ---
# Check for required geo dependencies (pyproj, rioxarray, etc.)
HAS_GEO_DEPS = importlib.util.find_spec("pyproj") is not None
