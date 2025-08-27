# config.py

"""
Central configuration file for the U.S. County-Level Drought Analysis dashboard.
This file stores constants and settings to make the application more maintainable.
"""

from typing import Dict, Final

# URLs for the climate data sources from data.cdc.gov
DATA_URLS: Final[Dict[str, str]] = {
    "SPEI": "https://data.cdc.gov/resource/6nbv-ifib.csv",
    "SPI": "https://data.cdc.gov/resource/xbk2-5i4e.csv",
    "PDSI": "https://data.cdc.gov/resource/en5r-5ds4.csv",
}

# File paths for local data assets
GEOJSON_PATH: Final[str] = "counties.geojson"
# The final, cleaned list of counties that have data for all indices
FIPS_PATH = "valid_counties.csv"

# The original, raw FIPS file used as input for the build script
RAW_FIPS_PATH = "fips_to_county.csv"
