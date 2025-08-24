import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import requests
import hashlib
import time
import urllib.parse
from config import DATA_URLS, GEOJSON_PATH, FIPS_PATH
from schemas import climate_data_schema

# --- Constants ---
CACHE_DIR = "cache"
CACHE_EXPIRATION_SECONDS = 24 * 60 * 60  # 24 hours

# --- Helper Functions ---

def initialize_cache():
    """Ensures the cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)

initialize_cache()

@st.cache_data(ttl=CACHE_EXPIRATION_SECONDS)
def get_map_data(index_type: str) -> pd.DataFrame:
    """
    Fetches the latest month's data for a given index type for all counties.
    This is used to populate the main interactive map.
    """
    base_url = DATA_URLS.get(index_type)
    if not base_url:
        st.error(f"No data URL configured for index type: {index_type}")
        return pd.DataFrame()

    query = "$order=year DESC, month DESC&$limit=1"
    try:
        response = requests.get(f"{base_url}?{query}")
        response.raise_for_status()
        latest_entry = response.json()
        if not latest_entry:
            st.warning(f"No data available for index {index_type}")
            return pd.DataFrame()
        
        latest_year = latest_entry[0]['year']
        latest_month = latest_entry[0]['month']

        # FIX: Manually build and encode the URL to ensure quotes are preserved.
        where_clause = f"year='{latest_year}' AND month='{latest_month}'"
        encoded_where = urllib.parse.quote(where_clause)
        full_url = f"{base_url}?$limit=10000&$where={encoded_where}"
        
        response = requests.get(full_url)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        
        # --- Data Standardization ---
        df["month"] = df["month"].map("{:02}".format)
        df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str)
        
        if "fips" in df.columns and "countyfips" not in df.columns:
            df.rename(columns={"fips": "countyfips"}, inplace=True)
        df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)

        value_col_mapping = {"SPEI": "spei", "SPI": "spi", "PDSI": "pdsi"}
        df.rename(columns={value_col_mapping[index_type]: "Value"}, inplace=True)
        df["index_type"] = index_type
        
        fips_df = get_county_options(return_df=True)
        merged_df = pd.merge(df, fips_df, on="countyfips", how="left")
        merged_df.dropna(subset=["display_name"], inplace=True)
        
        return merged_df[["date", "countyfips", "Value", "index_type", "display_name"]]

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch map data for {index_type}: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while processing map data for {index_type}: {e}")
        return pd.DataFrame()


@st.cache_data()
def get_county_options(return_df: bool = False) -> pd.Series | pd.DataFrame:
    """
    Loads FIPS data to populate the county selection dropdown.
    """
    if not os.path.exists(FIPS_PATH):
        st.error(f"Fatal Error: The FIPS data file was not found at {FIPS_PATH}")
        return pd.Series(dtype='str') if not return_df else pd.DataFrame()

    fips_df = pd.read_csv(FIPS_PATH, dtype=str)
    fips_df["state_fips"] = fips_df["state_fips"].str.zfill(2)
    fips_df["county_fips"] = fips_df["county_fips"].str.zfill(3)
    fips_df["countyfips"] = fips_df["state_fips"] + fips_df["county_fips"]
    fips_df.rename(columns={"county": "county_name"}, inplace=True)
    fips_df["display_name"] = fips_df["county_name"] + ", " + fips_df["state"]

    if return_df:
        return fips_df[["countyfips", "display_name", "state"]]
        
    return fips_df.sort_values("display_name").set_index("countyfips")["display_name"]


def get_live_data_for_counties(county_fips_list: list[str]) -> pd.DataFrame:
    """
    Fetches and processes live climate data for a specific list of counties.
    """
    if not county_fips_list:
        return pd.DataFrame()

    county_fips_list.sort()
    cache_key = hashlib.md5("".join(county_fips_list).encode()).hexdigest()
    cache_file_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")

    if os.path.exists(cache_file_path):
        file_mod_time = os.path.getmtime(cache_file_path)
        if (time.time() - file_mod_time) < CACHE_EXPIRATION_SECONDS:
            try:
                st.info("Loading data from local cache...")
                return pd.read_parquet(cache_file_path)
            except Exception as e:
                st.warning(f"Could not read cache file. Refetching data. Error: {e}")

    with st.spinner(f"Fetching live data for {len(county_fips_list)} selected counties..."):
        all_data = []
        fips_col_mapping = {"SPEI": "fips", "SPI": "countyfips", "PDSI": "countyfips"}

        for index_type, base_url in DATA_URLS.items():
            fips_col = fips_col_mapping[index_type]
            where_clause = " OR ".join([f"{fips_col}='{fips}'" for fips in county_fips_list])
            params = {"$limit": 10000000, "$where": where_clause}

            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                df = pd.DataFrame(response.json())

                df["month"] = df["month"].map("{:02}".format)
                df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str)
                if "fips" in df.columns and "countyfips" not in df.columns:
                    df.rename(columns={"fips": "countyfips"}, inplace=True)
                df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)

                value_col_mapping = {"SPEI": "spei", "SPI": "spi", "PDSI": "pdsi"}
                df.rename(columns={value_col_mapping[index_type]: "Value"}, inplace=True)
                df["index_type"] = index_type
                all_data.append(df)

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {index_type} data: {e}")
                continue
        
        if not all_data:
            st.warning("Could not load any climate data for the selected counties.")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        
        # FIX: Add robust date parsing to handle potential malformed date strings.
        combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y-%m", errors='coerce')
        combined_df.dropna(subset=["date"], inplace=True)

        fips_df = get_county_options(return_df=True)
        final_df = pd.merge(combined_df, fips_df, on="countyfips", how="left")
        final_df.dropna(subset=["display_name"], inplace=True)

        try:
            validated_df = climate_data_schema.validate(final_df)
            validated_df.to_parquet(cache_file_path)
            return validated_df
        except Exception as e:
            st.error(f"Final data validation failed: {e}")
            return pd.DataFrame()


@st.cache_data()
def get_geojson() -> gpd.GeoDataFrame | None:
    """
    Loads the GeoJSON file for county boundaries.
    """
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Fatal Error: The GeoJSON file was not found at {GEOJSON_PATH}")
        return None
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf.rename(columns={"id": "countyfips"}, inplace=True)
    return gdf
