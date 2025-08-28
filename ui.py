# -*- coding: utf-8 -*-
"""
This module contains the UI components for the Streamlit application.
"""
import streamlit as st
from datetime import datetime

def setup_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(
        page_title="U.S. County-Level Drought Analysis",
        page_icon="🌦️",
        layout="wide",
    )

def display_header_and_about():
    """Displays the main title and the 'About' expander."""
    st.title("U.S. County-Level Drought Analysis")
    st.markdown(
        "Explore, compare, and forecast key drought indices for any county in the United States. "
        "Data is fetched live from NOAA's National Centers for Environmental Information (NCEI)."
    )
    with st.expander("About the Climate Indices"):
        st.markdown(
            """
            - **PDSI (Palmer Drought Severity Index):** Measures long-term drought based on temperature and precipitation data.
            - **SPI (Standardized Precipitation Index):** Shows how precipitation compares to the long-term average for a given period.
            - **SPEI (Standardized Precipitation-Evapotranspiration Index):** Similar to SPI, but also includes the effect of temperature on water demand.
            """
        )

def display_sidebar(fips_options):
    """
    Renders the sidebar controls.

    Args:
        fips_options (pd.Series): A series of county FIPS codes and names.

    Returns:
        tuple: A tuple containing the selected index choice and a list of FIPS codes.
    """
    with st.sidebar:
        st.header("Dashboard Controls")
        st.info(f"Data is fetched live. Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        if fips_options.empty:
            st.error("County list could not be loaded. The dashboard cannot be displayed.")
            st.stop()

        index_choice = st.selectbox(
            "1. Select Climate Index:",
            options=["PDSI", "SPI", "SPEI"],
            key="index_selectbox",
        )

        fips_code_inputs = st.multiselect(
            "2. Analyze a County (or Several):",
            options=fips_options.index.tolist(),
            format_func=lambda x: fips_options.get(x, "Unknown County"),
            key="fips_multiselect",
            help="You can type to search for a county and select multiple counties for comparison.",
        )

        if not fips_code_inputs:
            st.success("Start Here! 👆 Select a county to begin analysis.")
        
        return index_choice, fips_code_inputs

def display_download_button(full_data, index_choice, fips_code_inputs):
    """
    Renders the download button in the sidebar.
    
    Args:
        full_data (pd.DataFrame): The dataframe to be downloaded.
        index_choice (str): The selected climate index.
        fips_code_inputs (list): The list of selected FIPS codes.
    """
    if not full_data.empty:
        st.sidebar.download_button(
            label="Download Selected Data (CSV)",
            data=full_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{index_choice}_data_{'_'.join(fips_code_inputs)}.csv",
            mime="text/csv",
            key="download_button",
        )
