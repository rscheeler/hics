import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from platformdirs import user_data_dir

# Optional: Still support .env if they HAVE one, but don't require it
load_dotenv(find_dotenv(), override=False)

APP_NAME = "hics"


class _DEMSettings:
    def __init__(self):
        self._base_data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
        self._dem_folder = None
        self._nlcdleg_folder = None

    def _get_path(self, env_key: str, folder_name: str) -> Path:
        # 1. Check for Environment Variable first (Priority for Shared Drives)
        env_val = os.getenv(env_key)
        if env_val:
            return Path(env_val)

        # 2. Otherwise, use the standard OS data directory
        return self._base_data_dir / folder_name

    @property
    def DEM_FOLDER(self) -> Path:
        if self._dem_folder is None:
            self._dem_folder = self._get_path("HICS_DEM_FOLDER", "terrain")
        return self._dem_folder

    @DEM_FOLDER.setter
    def DEM_FOLDER(self, path: str | Path) -> None:
        self._dem_folder = Path(path)

    @property
    def NLCDLEG_FOLDER(self) -> Path:
        if self._nlcdleg_folder is None:
            self._nlcdleg_folder = self._get_path("HICS_NLCDLEG_FOLDER", "legends")
        return self._nlcdleg_folder

    @NLCDLEG_FOLDER.setter
    def NLCDLEG_FOLDER(self, path: str | Path) -> None:
        self._nlcdleg_folder = Path(path)


DEM_SETTINGS = _DEMSettings()


# Lazy directory creation: only create when we actually try to load data
def initialize_folders() -> None:
    """Initialize folders if needed."""
    DEM_SETTINGS.DEM_FOLDER.mkdir(parents=True, exist_ok=True)
    DEM_SETTINGS.NLCDLEG_FOLDER.mkdir(parents=True, exist_ok=True)
