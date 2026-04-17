from __future__ import annotations

import pandas as pd
import planetary_computer
import pystac_client
import rasterio

from .config import DEM_SETTINGS


def get_lc_class(collection_str: str) -> pd.DataFrame:
    """Get land cover class data.

    Parameters
    ----------
    collection_str : str
        Collection string from Planetary Computer

    Returns:
    -------
    Class data : pd.DataFrame
    """
    # 1. Setup Connection
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 2. Get Metadata from Collection
    collection = catalog.get_collection(collection_str)
    item_assets = collection.extra_fields.get("item_assets", {})
    data_def = item_assets.get("data", {})
    class_list = data_def.get("file:values", [])

    # 3. Get one Item to extract the actual Colormap
    # We need to open one real file to read the embedded color table
    items = collection.get_all_items()
    sample_item = next(items)
    data_url = sample_item.assets["data"].href

    with rasterio.open(data_url) as src:
        embedded_colormap = src.colormap(1)

    # 4. Build the list of data for the DataFrame
    rows = []
    for item in class_list:
        class_name = item["summary"]
        val = item["values"][0]

        # Get the RGBA tuple from the raster's colormap
        # We use .get() in case a value in metadata isn't in the raster palette
        rgba_tuple = embedded_colormap.get(val, (0, 0, 0, 0))

        rows.append(
            {
                "Value": val,
                "Name": class_name,
                "RGBA": rgba_tuple,
            }
        )

    # 5. Create DataFrame
    df = pd.DataFrame(rows)
    df = df.sort_values("Value").reset_index(drop=True)

    return df


# The Translator: Maps dataset-specific names to our Master Categories
dataset_mappings = {
    "io-lulc-annual-v02": {
        "Water": "WATER",
        "Trees": "FOREST",
        "Flooded vegetation": "WETLAND",
        "Crops": "OPEN_LOW",
        "Built area": "URBAN",
        "Bare ground": "BARREN",
        "Snow/ice": "BARREN",
        "Clouds": "OPEN_LOW",
        "Rangeland": "OPEN_LOW",
    },
    "esa-worldcover": {
        "Tree cover": "FOREST",
        "Shrubland": "OPEN_LOW",
        "Grassland": "OPEN_LOW",
        "Cropland": "OPEN_LOW",
        "Built-up": "URBAN",
        "Bare / sparse vegetation": "BARREN",
        "Snow and ice": "BARREN",
        "Permanent water bodies": "WATER",
        "Herbaceous wetland": "WETLAND",
    },
}
# Define the physical properties for our Master Categories
# Land cover (clutter) height and ITM Ground type reference: https://www.pathloss.com/webhelp/terrain_data/terdat_clutter_clutdef.html
MAIN_PROPS = {
    "WATER": {
        "Land Cover Height (m)": 0.1,
        "ITM Ground Type": "fresh_water",
        "Relative Permittivity": 81,
        "Conductivity (S/m)": 0.01,
        "RMS Slope": 0.07,
    },
    "FOREST": {
        "Land Cover Height (m)": 15.0,
        "ITM Ground Type": "average",
        "Relative Permittivity": 15,
        "Conductivity (S/m)": 0.005,
        "RMS Slope": 0.35,
    },
    "URBAN": {
        "Land Cover Height (m)": 10.0,
        "ITM Ground Type": "average",
        "Relative Permittivity": 15,
        "Conductivity (S/m)": 0.005,
        "RMS Slope": 0.35,
    },
    "WETLAND": {
        "Land Cover Height (m)": 5.0,
        "ITM Ground Type": "good",
        "Relative Permittivity": 25,
        "Conductivity (S/m)": 0.02,
        "RMS Slope": 0.19,
    },
    "OPEN_LOW": {
        "Land Cover Height (m)": 0.5,
        "ITM Ground Type": "average",
        "Relative Permittivity": 15,
        "Conductivity (S/m)": 0.005,
        "RMS Slope": 0.21,
    },
    "BARREN": {
        "Land Cover Height (m)": 0.1,
        "ITM Ground Type": "poor",
        "Relative Permittivity": 4,
        "Conductivity (S/m)": 0.001,
        "RMS Slope": 0.20,
    },
}


def get_surface_props(row: pd.Series, mapping: dict) -> dict:
    """Find electrical and surface properties for the row.

    Parameters
    ----------
    row : pd.Series
        Row to append to.
    mapping : dict
        Dictionary mapping to MAIN_PROPS

    Returns:
    -------
    props : dict
        Properties for the surface.
    """
    cat = mapping.get(row["Name"], "BARREN")
    props = MAIN_PROPS[cat]
    return props


def generate_rf_csv(dataset_id: str) -> pd.DataFrame:
    """Generate legend csv.

    Parameters
    ----------
    dataset_id : str
        Dataset from planetary computer.

    Returns:
    -------
    legend : pd.DataFrame
    """
    # Get data
    df = get_lc_class(dataset_id)

    # Add electrical properties
    mapping = dataset_mappings.get(dataset_id)
    new_cols = df.apply(get_surface_props, args=[mapping], axis=1, result_type="expand")
    df = pd.concat([df, new_cols], axis=1)
    # Save
    fname = DEM_SETTINGS.NLCDLEG_FOLDER / f"{dataset_id}.csv"
    df.to_csv(fname, index=False)

    return df
