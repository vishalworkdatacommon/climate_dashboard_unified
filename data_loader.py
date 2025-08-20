import streamlit as st
import pandas as pd
import geopandas as gpd
import os
from datetime import datetime
import traceback
from config import DATA_URLS, GEOJSON_PATH, FIPS_PATH
from schemas import climate_data_schema, fips_data_schema

from typing import Tuple, Optional


@st.cache_data(ttl="1d")
def get_live_data() -> Tuple[
    pd.DataFrame, pd.Series, Optional[gpd.GeoDataFrame], Optional[datetime]
]:
    """
    Downloads, parses, and combines live climate data from data.cdc.gov,
    and merges it with geospatial data for mapping.
    The result is cached for 24 hours.
    """
    with st.spinner(
        "Fetching and processing live climate and geo data... This may take several minutes..."
    ):
        # --- Load Geospatial Data ---
        if not os.path.exists(GEOJSON_PATH):
            st.error(
                "Fatal Error: The GeoJSON file for county boundaries was not found."
            )
            return pd.DataFrame(), pd.Series(), None, None
        gdf = gpd.read_file(GEOJSON_PATH)
        gdf.rename(columns={"id": "countyfips"}, inplace=True)

        # --- Load Climate Data ---
        all_data = []
        for index_type, url in DATA_URLS.items():
            try:
                full_url = f"{url}?$limit=10000000"
                df = pd.read_csv(full_url)
                df["month"] = df["month"].map("{:02}".format)
                df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str)
                if "fips" in df.columns:
                    df.rename(columns={"fips": "countyfips"}, inplace=True)
                df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)
                if index_type == "SPEI":
                    df.rename(columns={"spei": "Value"}, inplace=True)
                elif index_type == "SPI":
                    df.rename(columns={"spi": "Value"}, inplace=True)
                elif index_type == "PDSI":
                    df.rename(columns={"pdsi": "Value"}, inplace=True)
                df["index_type"] = index_type
                cols_to_keep = ["date", "countyfips", "Value", "index_type"]
                if all(col in df.columns for col in cols_to_keep):
                    all_data.append(df[cols_to_keep])
            except Exception:
                st.error(f"Failed to load or process data for {index_type}.")
                st.code(traceback.format_exc())
                continue

        if not all_data:
            st.error("Could not load any climate data. The application cannot proceed.")
            return pd.DataFrame(), pd.Series(), None, None

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df["date"] = pd.to_datetime(combined_df["date"])

        # --- Load FIPS Data ---
        if not os.path.exists(FIPS_PATH):
            st.error("Fatal Error: The FIPS data file was not found.")
            return pd.DataFrame(), pd.Series(), None, None

        fips_df = pd.read_csv(FIPS_PATH)
        fips_df["state_fips"] = fips_df["state_fips"].astype(str).str.zfill(2)
        fips_df["county_fips"] = fips_df["county_fips"].astype(str).str.zfill(3)
        fips_df["countyfips"] = fips_df["state_fips"] + fips_df["county_fips"]
        fips_df.rename(columns={"county": "county_name"}, inplace=True)

        # --- Validate Data ---
        try:
            fips_df = fips_data_schema.validate(fips_df)
        except Exception as e:
            st.error("FIPS data validation failed.")
            st.code(str(e))
            return pd.DataFrame(), pd.Series(), None, None

        # Merge
        df = pd.merge(
            combined_df,
            fips_df[["countyfips", "county_name", "state"]],
            on="countyfips",
            how="left",
        )
        df.dropna(subset=["county_name", "state"], inplace=True)
        df["display_name"] = df["county_name"] + ", " + df["state"]

        try:
            df = climate_data_schema.validate(df)
        except Exception as e:
            st.error("Climate data validation failed after merge.")
            st.code(str(e))
            return pd.DataFrame(), pd.Series(), None, None

        fips_options = (
            df[["countyfips", "display_name"]]
            .drop_duplicates()
            .sort_values("display_name")
            .set_index("countyfips")
        )

        return df, fips_options, gdf, datetime.now()
