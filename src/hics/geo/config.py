import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

# Load .env
load_dotenv(find_dotenv())


class _DEMSettings:
    DEM_FOLDER: Path = Path(os.getenv("DEM_FOLDER"))
    NLCD_FILE: Path = Path(os.getenv("NLCD_FILE"))
    NLCDLEG_FILE: Path = Path(os.getenv("NLCDLEG_FILE"))


DEM_SETTINGS = _DEMSettings()
# Create folder if needed
try:
    DEM_SETTINGS.DEM_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory {DEM_SETTINGS.DEM_FOLDER} created succesfully.")
except OSError as error:
    logger.exception(f"Directory {DEM_SETTINGS.DEM_FOLDER} cannot be created: {error}")
