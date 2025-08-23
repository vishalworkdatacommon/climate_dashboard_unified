import streamlit as st
import pandas as pd
import geopandas as gpd
import os
from datetime import datetime
from config import GEOJSON_PATH, FIPS_PATH
from schemas import climate_data_schema, fips_data_schema
from typing import Tuple, Optional

# Define the path to the Parquet file
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "climate_indices.parquet")


@st.cache_data()
def get_prebuilt_data() -> Tuple[
    pd.DataFrame, pd.Series, Optional[gpd.GeoDataFrame], Optional[datetime]
]:
    """
    Loads pre-built climate data from a local Parquet file and merges it
    with geospatial data for mapping.
    """
    with st.spinner("Loading pre-built climate and geo data..."):
        # --- Load Climate Data from Parquet ---
        if not os.path.exists(PARQUET_PATH):
            st.error(
                "Fatal Error: The climate data file (climate_indices.parquet) was not found. Please run the build_data.py script first."
            )
            return pd.DataFrame(), pd.Series(), None, None

        try:
            combined_df = pd.read_parquet(PARQUET_PATH)
            last_updated = datetime.fromtimestamp(os.path.getmtime(PARQUET_PATH))
        except Exception as e:
            st.error(f"Failed to load or process the Parquet data file. Error: {e}")
            return pd.DataFrame(), pd.Series(), None, None

        # --- Load Geospatial Data ---
        if not os.path.exists(GEOJSON_PATH):
            st.error(
                "Fatal Error: The GeoJSON file for county boundaries was not found."
            )
            return pd.DataFrame(), pd.Series(), None, None
        gdf = gpd.read_file(GEOJSON_PATH)
        gdf.rename(columns={"id": "countyfips"}, inplace=True)

        # --- Load FIPS Data ---
        if not os.path.exists(FIPS_PATH):
            st.error("Fatal Error: The FIPS data file was not found.")
            return pd.DataFrame(), pd.Series(), None, None

        fips_df = pd.read_csv(FIPS_PATH)
        fips_df["state_fips"] = fips_df["state_fips"].astype(str).str.zfill(2)
        fips_df["county_fips"] = fips_df["county_fips"].astype(str).str.zfill(3)
        fips_df["countyfips"] = fips_df["state_fips"] + fips_df["county_fips"]
        fips_df.rename(columns={"county": "county_name"}, inplace=True)

        # --- Validate FIPS Data ---
        try:
            fips_df = fips_data_schema.validate(fips_df)
        except Exception as e:
            st.error(f"FIPS data validation failed: {e}")
            return pd.DataFrame(), pd.Series(), None, None

        # --- Merge and Finalize ---
        df = pd.merge(
            combined_df,
            fips_df[["countyfips", "county_name", "state"]],
            on="countyfips",
            how="left",
        )
        df.dropna(subset=["county_name", "state"], inplace=True)
        df["display_name"] = df["county_name"] + ", " + df["state"]

        # --- Validate Final Data ---
        try:
            df = climate_data_schema.validate(df)
        except Exception as e:
            st.error(f"Final data validation failed after merge: {e}")
            return pd.DataFrame(), pd.Series(), None, None

        fips_options = (
            df[["countyfips", "display_name"]]
            .drop_duplicates()
            .sort_values("display_name")
            .set_index("countyfips")
        )

        return df, fips_options, gdf, last_updated