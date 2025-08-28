# -*- coding: utf-8 -*-
import streamlit as st
import warnings
import plotly.graph_objects as go

# --- Custom Modules ---
from data_loader import get_county_options, get_live_data_for_counties, get_geojson, get_latest_data_for_all_counties
from plotting import (
    plot_trend_analysis,
    plot_anomaly_detection,
    plot_seasonal_decomposition,
    plot_autocorrelation,
    plot_comparison_mode,
    display_historical_insights,
    plot_national_map,
)
from ml_models import handle_forecasting_tab
from ui import setup_page_config, display_header_and_about, display_sidebar, display_download_button

# Suppress warnings for a cleaner app
warnings.filterwarnings("ignore")

def main() -> None:
    """Main function to run the Streamlit application."""
    setup_page_config()
    display_header_and_about()

    fips_options = get_county_options()
    index_choice, fips_code_inputs = display_sidebar(fips_options)

    if not fips_code_inputs:
        st.header("National Map View")
        st.markdown("**Dive deeper: Search for a county in the sidebar to access detailed trends and forecasts.**")
        latest_data = get_latest_data_for_all_counties(index_choice)
        
        if latest_data is not None:
            gdf = get_geojson()
            if gdf is not None:
                fig = plot_national_map(latest_data, gdf, index_choice, fips_options)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not load geospatial data for the map.")
        st.stop()

    full_data = get_live_data_for_counties(fips_code_inputs)
    if full_data.empty:
        st.warning("No data could be fetched for the selected counties. Please try other selections or check back later.")
        st.stop()

    selected_county_names = [fips_options.get(fips, "Unknown") for fips in fips_code_inputs]

    if len(fips_code_inputs) == 1:
        handle_single_county_view(full_data, fips_code_inputs[0], index_choice, selected_county_names[0])
    else:
        handle_multi_county_view(full_data, fips_code_inputs, index_choice, selected_county_names, fips_options)

    display_download_button(full_data, index_choice, fips_code_inputs)
    st.markdown("---")
    st.markdown("Data Source: [NOAA NCEI](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/time-series)")

def handle_single_county_view(full_data, fips_code, index_choice, county_name):
    """Handles the UI and logic for the single-county analysis view."""
    st.header(county_name)
    st.caption(f"Climate Index: {index_choice}")

    tabs = st.tabs(["Trend Analysis", "Anomaly Detection", "Seasonal Decomposition", "Autocorrelation", "Forecasting"])
    
    filtered_df = full_data[(full_data["countyfips"] == fips_code) & (full_data["index_type"] == index_choice)]
    if not filtered_df.empty:
        time_series = filtered_df.set_index("date")["Value"].asfreq("MS")

        with tabs[0]:
            fig = plot_trend_analysis(time_series, index_choice)
            st.plotly_chart(fig, use_container_width=True)
            display_historical_insights(time_series)
        with tabs[1]:
            fig = plot_anomaly_detection(time_series, index_choice)
            st.plotly_chart(fig, use_container_width=True)
            display_historical_insights(time_series)
        with tabs[2]:
            fig = plot_seasonal_decomposition(time_series, index_choice)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            fig = plot_autocorrelation(time_series, index_choice)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[4]:
            handle_forecasting_tab(time_series, index_choice, county_name)

def handle_multi_county_view(full_data, fips_codes, index_choice, county_names, fips_options):
    """Handles the UI and logic for the multi-county comparison view."""
    analysis_choice = st.selectbox("Select Analysis:", ["Trend Analysis", "Anomaly Detection", "Comparison Mode"], key="analysis_selectbox")
    
    st.header(f"{analysis_choice}: {index_choice}")
    st.markdown(f"**Displaying data for:** `{', '.join(county_names)}`")

    if analysis_choice == "Comparison Mode":
        plot_comparison_mode(full_data, fips_codes, index_choice)
    else:
        fig = go.Figure()
        for fips_code in fips_codes:
            filtered_df = full_data[(full_data["countyfips"] == fips_code) & (full_data["index_type"] == index_choice)]
            if not filtered_df.empty:
                time_series = filtered_df.set_index("date")["Value"].asfreq("MS")
                county_name = fips_options.get(fips_code, "Unknown")
                if analysis_choice == "Trend Analysis":
                    fig.add_trace(go.Scatter(x=time_series.index, y=time_series, mode="lines", name=county_name))
                elif analysis_choice == "Anomaly Detection":
                    rolling_mean = time_series.rolling(window=12).mean()
                    rolling_std = time_series.rolling(window=12).std()
                    anomalies = time_series[(time_series > rolling_mean + (2 * rolling_std)) | (time_series < rolling_mean - (2 * rolling_std))]
                    fig.add_trace(go.Scatter(x=time_series.index, y=time_series, mode="lines", name=county_name))
                    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode="markers", name=f"{county_name} Anomaly", marker=dict(symbol="x")))
        
        fig.update_layout(
            title=f"{index_choice} {analysis_choice}",
            xaxis_title="Year", yaxis_title=f"{index_choice} Value",
            legend_title="Counties", template="plotly_white",
            font=dict(size=14), height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
