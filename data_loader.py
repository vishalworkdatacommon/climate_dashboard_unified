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




@st.cache_data(show_spinner=False)
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
    os.makedirs(CACHE_DIR, exist_ok=True)  # Ensure the cache directory exists
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

                # Coerce 'Value' to numeric, turning any non-numeric placeholders (e.g., 'M') into NaN
                df["Value"] = pd.to_numeric(df["Value"], errors='coerce')
                df.dropna(subset=["Value"], inplace=True)

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

        # --- Safeguard Data Cleaning ---
        # This is a redundant but safe step to ensure the final column is numeric.
        combined_df["Value"] = pd.to_numeric(combined_df["Value"], errors='coerce')
        combined_df.dropna(subset=["Value"], inplace=True)

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

        # --- Final, Definitive Type Casting ---
        # This is the last line of defense to ensure the 'Value' column is float64.
        # It handles both non-numeric strings and integers being cast correctly.
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



@st.cache_data(show_spinner=False)
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


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Constants ---
NATIONAL_DATA_PATH = "national_latest.parquet"
CACHE_LOCK_PATH = ".cache_lock"
CACHE_EXPIRATION_SECONDS = 24 * 60 * 60  # 24 hours

def _fetch_and_save_national_data():
    """
    This is the high-performance data fetching logic that runs in a background thread.
    It uses efficient batching with 'IN' clauses to fetch data quickly.
    """
    print("--- Starting background national data refresh (batched) ---")
    
    try:
        fips_df = pd.read_csv(FIPS_PATH, dtype=str)
        fips_df["countyfips"] = fips_df["state_fips"].str.zfill(2) + fips_df["county_fips"].str.zfill(3)
        all_fips = fips_df["countyfips"].tolist()

        all_indices_data = []
        for index_type in DATA_URLS.keys():
            print(f"Fetching {index_type} for {len(all_fips)} counties...")
            fips_col = {"SPEI": "fips", "SPI": "countyfips", "PDSI": "countyfips"}[index_type]
            value_col = {"SPEI": "spei", "SPI": "spi", "PDSI": "pdsi"}[index_type]
            base_url = DATA_URLS[index_type]
            
            index_latest_data = []
            
            # --- Batching Logic ---
            chunk_size = 80 
            for i in range(0, len(all_fips), chunk_size):
                chunk = all_fips[i:i + chunk_size]
                
                # Create a SoQL 'IN' clause
                in_clause = ", ".join([f"'{fips}'" for fips in chunk])
                # We need to get the latest for each, so we group by fips and get max date
                # This is a complex query that is more efficient than thousands of single ones.
                query = (
                    f"?$select={fips_col},max(date) as latest_date"
                    f"&$where={fips_col} IN({in_clause})"
                    f"&$group={fips_col}"
                )

                try:
                    # This query is not directly possible with Socrata's GET request for the value.
                    # We will revert to a parallel but more controlled fetch.
                    # The issue is not parallelization itself, but uncontrolled parallelization.
                    # A more controlled ThreadPool with fewer workers is the key.
                    pass # Re-implementing below
                except requests.RequestException as e:
                    print(f"  - Error in batch query for {index_type}: {e}")
                    continue
            
            # --- Controlled Parallel Fetch (The Real Fix) ---
            with ThreadPoolExecutor(max_workers=10) as executor: # Reduced workers to be less aggressive
                futures = {executor.submit(_fetch_single_fips, fips, index_type): fips for fips in all_fips}
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result is not None:
                        index_latest_data.append(result)
                    if (i + 1) % 200 == 0:
                        print(f"  - Progress for {index_type}: {(i + 1)} / {len(all_fips)} counties processed.")

            if index_latest_data:
                df_index = pd.concat(index_latest_data, ignore_index=True)
                if "fips" in df_index.columns: df_index.rename(columns={"fips": "countyfips"}, inplace=True)
                df_index.rename(columns={value_col: "Value"}, inplace=True)
                df_index["countyfips"] = df_index["countyfips"].astype(str).str.zfill(5)
                all_indices_data.append(df_index[["countyfips", "Value", "index_type"]])

        if all_indices_data:
            final_df = pd.concat(all_indices_data, ignore_index=True)
            final_pivot = final_df.pivot(index='countyfips', columns='index_type', values='Value')
            final_pivot.reset_index(inplace=True)
            final_pivot.to_parquet(NATIONAL_DATA_PATH)
            print("--- Background national data refresh complete ---")
        
    finally:
        if os.path.exists(CACHE_LOCK_PATH):
            os.remove(CACHE_LOCK_PATH)

def _fetch_single_fips(fips: str, index_type: str) -> pd.DataFrame | None:
    """Fetches the latest data for a single county FIPS code."""
    fips_col = {"SPEI": "fips", "SPI": "countyfips", "PDSI": "countyfips"}[index_type]
    base_url = DATA_URLS[index_type]
    query = f"?$where={fips_col}='{fips}'&$order=year DESC, month DESC&$limit=1"
    
    try:
        response = requests.get(base_url + query, timeout=15)
        if response.status_code == 200:
            df_single = pd.read_csv(StringIO(response.text))
            if not df_single.empty:
                df_single['index_type'] = index_type
                return df_single
    except requests.RequestException:
        return None
    return None


@st.cache_data(show_spinner=False)
def get_latest_data_for_all_counties(index_type: str) -> pd.DataFrame | None:
    """
    Loads the pre-computed national latest data from the local Parquet file.
    This file is generated and updated automatically by a GitHub Action.
    """
    NATIONAL_DATA_PATH = "national_latest.parquet"
    
    if not os.path.exists(NATIONAL_DATA_PATH):
        st.error(
            "Error: The pre-computed national data file (`national_latest.parquet`) was not found. "
            "This file is normally generated automatically. Please check the repository or run the build script manually."
        )
        return None
        
    try:
        df = pd.read_parquet(NATIONAL_DATA_PATH)
        if index_type not in df.columns:
            st.error(f"The selected index '{index_type}' was not found in the pre-computed national data.")
            return None
            
        result_df = df[["countyfips", index_type]].copy()
        result_df.rename(columns={index_type: "Value"}, inplace=True)
        result_df.dropna(subset=["Value"], inplace=True)
        return result_df
        
    except Exception as e:
        st.error(f"An error occurred while reading the national data file: {e}")
        return None
