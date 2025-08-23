import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import requests
from io import StringIO
import hashlib
import time
from config import DATA_URLS, GEOJSON_PATH, FIPS_PATH
from schemas import climate_data_schema




@st.cache_data()
def get_county_options() -> pd.Series:
    """
    Loads FIPS data to populate the county selection dropdown.
    This is cached as it rarely changes.
    """
    if not os.path.exists(FIPS_PATH):
        st.error(f"Fatal Error: The FIPS data file was not found at {FIPS_PATH}")
        return pd.Series(dtype='str')

    fips_df = pd.read_csv(FIPS_PATH, dtype=str)
    fips_df["state_fips"] = fips_df["state_fips"].str.zfill(2)
    fips_df["county_fips"] = fips_df["county_fips"].str.zfill(3)
    fips_df["countyfips"] = fips_df["state_fips"] + fips_df["county_fips"]
    fips_df.rename(columns={"county": "county_name"}, inplace=True)
    fips_df["display_name"] = fips_df["county_name"] + ", " + fips_df["state"]

    return (
        fips_df.sort_values("display_name")
        .set_index("countyfips")["display_name"]
    )


# @st.cache_data(ttl="1h") # We are replacing Streamlit's cache with a custom file-based cache
def get_live_data_for_counties(county_fips_list: list[str]) -> pd.DataFrame:
    """
    Fetches and processes live climate data for a specific list of counties
    using the Socrata API. Results are cached to a local Parquet file for 24 hours.
    """
    if not county_fips_list:
        return pd.DataFrame()

    # --- Caching Logic ---
    CACHE_DIR = "cache"
    CACHE_EXPIRATION_SECONDS = 24 * 60 * 60  # 24 hours

    # Create a unique hash for the list of counties to use as a filename
    county_fips_list.sort()
    cache_key = hashlib.md5("".join(county_fips_list).encode()).hexdigest()
    cache_file_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")

    # Check if a valid cache file exists
    if os.path.exists(cache_file_path):
        file_mod_time = os.path.getmtime(cache_file_path)
        if (time.time() - file_mod_time) < CACHE_EXPIRATION_SECONDS:
            try:
                st.info("Loading data from local cache...")
                return pd.read_parquet(cache_file_path)
            except Exception as e:
                st.warning(f"Could not read cache file. Refetching data. Error: {e}")

    # --- API Fetching Logic ---
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
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)

                df["month"] = df["month"].map("{:02}".format)
                df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str)
                if "fips" in df.columns and "countyfips" not in df.columns:
                    df.rename(columns={"fips": "countyfips"}, inplace=True)
                df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)

                value_col_mapping = {"SPEI": "spei", "SPI": "spi", "PDSI": "pdsi"}
                df.rename(columns={value_col_mapping[index_type]: "Value"}, inplace=True)
                df["index_type"] = index_type

                cols_to_keep = ["date", "countyfips", "Value", "index_type"]
                if all(col in df.columns for col in cols_to_keep):
                    all_data.append(df[cols_to_keep])

            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to download {index_type} data: Server error ({e.response.status_code}).")
                continue
            except Exception as e:
                st.error(f"Failed to process data for {index_type}. Error: {e}")
                continue

        if not all_data:
            st.warning("Could not load any climate data for the selected counties.")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df["date"] = pd.to_datetime(combined_df["date"])

        # --- Merge and Finalize ---
        fips_df = pd.read_csv(FIPS_PATH, dtype=str)
        fips_df["state_fips"] = fips_df["state_fips"].str.zfill(2)
        fips_df["county_fips"] = fips_df["county_fips"].str.zfill(3)
        fips_df["countyfips"] = fips_df["state_fips"] + fips_df["county_fips"]
        fips_df.rename(columns={"county": "county_name"}, inplace=True)

        df = pd.merge(
            combined_df,
            fips_df[["countyfips", "county_name", "state"]],
            on="countyfips",
            how="left",
        )
        df.dropna(subset=["county_name", "state"], inplace=True)
        df["display_name"] = df["county_name"] + ", " + df["state"]

        try:
            validated_df = climate_data_schema.validate(df)
            # Save to cache
            validated_df.to_parquet(cache_file_path)
            return validated_df
        except Exception as e:
            st.error(f"Final data validation failed: {e}")
            return pd.DataFrame()



@st.cache_data()
def get_geojson() -> gpd.GeoDataFrame | None:
    """
    Loads the GeoJSON file for county boundaries.
    Cached indefinitely as it's a static file.
    """
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Fatal Error: The GeoJSON file was not found at {GEOJSON_PATH}")
        return None
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf.rename(columns={"id": "countyfips"}, inplace=True)
    return gdf
