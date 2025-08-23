import streamlit as st
import pandas as pd
import geopandas as gpd
import os
from urllib.error import HTTPError
from urllib.parse import quote
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


@st.cache_data(ttl="1h")
def get_live_data_for_counties(county_fips_list: list[str]) -> pd.DataFrame:
    """
    Fetches and processes live climate data for a specific list of counties
    using the Socrata API. The result is cached for 1 hour.
    """
    if not county_fips_list:
        return pd.DataFrame()

    with st.spinner(f"Fetching live data for {len(county_fips_list)} selected counties..."):
        all_data = []
        where_clause = " OR ".join([f"countyfips='{fips}'" for fips in county_fips_list])

        for index_type, base_url in DATA_URLS.items():
            try:
                # Use quote on the entire where clause for proper encoding
                soql_query = f"?$limit=10000000&$where={quote(where_clause)}"
                full_url = base_url + soql_query
                df = pd.read_csv(full_url)

                df["month"] = df["month"].map("{:02}".format)
                df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str)
                if "fips" in df.columns:
                    df.rename(columns={"fips": "countyfips"}, inplace=True)
                df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)

                value_col_mapping = {"SPEI": "spei", "SPI": "spi", "PDSI": "pdsi"}
                df.rename(columns={value_col_mapping[index_type]: "Value"}, inplace=True)
                df["index_type"] = index_type

                cols_to_keep = ["date", "countyfips", "Value", "index_type"]
                if all(col in df.columns for col in cols_to_keep):
                    all_data.append(df[cols_to_keep])

            except HTTPError as e:
                st.error(f"Failed to download {index_type} data due to a server error ({e.code}). The data source may be unavailable.")
                continue
            except Exception as e:
                st.error(f"Failed to load or process data for {index_type}. Error: {e}")
                continue

        if not all_data:
            st.warning("Could not load any climate data for the selected counties.")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df["date"] = pd.to_datetime(combined_df["date"])

        # Merge with FIPS data to get display names
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
            return climate_data_schema.validate(df)
        except Exception as e:
            st.error(f"Final data validation failed after merge: {e}")
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
