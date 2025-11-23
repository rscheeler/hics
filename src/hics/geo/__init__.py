from loguru import logger

# --- CONDITIONAL GEO PATCHING ---
try:
    # 1. Check for required geo dependencies (pyproj, rioxarray, etc.)
    import pyproj
    import rioxarray

    # Define a flag for internal package use
    HAS_GEO_DEPS = True
    logger.info("Geo dependencies installed.")

except ValueError:
    # If any geo dependency is missing, do nothing, and the method isn't attached.
    HAS_GEO_DEPS = False
