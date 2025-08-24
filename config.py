# config.py

"""
Central configuration file for the U.S. County-Level Drought Analysis dashboard.
This file stores constants and settings to make the application more maintainable.
"""

from typing import Dict, Final

# URLs for the climate data sources from data.cdc.gov (using JSON endpoints)
DATA_URLS: Final[Dict[str, str]] = {
    "SPEI": "https://data.cdc.gov/resource/6nbv-ifib.json",
    "SPI": "https://data.cdc.gov/resource/xbk2-5i4e.json",
    "PDSI": "https://data.cdc.gov/resource/en5r-5ds4.json",
}

# File paths for local data assets
GEOJSON_PATH: Final[str] = "counties.geojson"
FIPS_PATH: Final[str] = "fips_to_county.csv"
