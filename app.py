# -*- coding: utf-8 -*-
import streamlit as st
import warnings
import plotly.graph_objects as go
from datetime import datetime
import os
import subprocess
import toml
import time




st.set_page_config(
    page_title="U.S. County-Level Drought Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Modules ---
from data_loader import get_county_options, get_live_data_for_counties, get_geojson, get_prebuilt_data_for_map
from plotting import (
    plot_trend_analysis,
    plot_anomaly_detection,
    plot_seasonal_decomposition,
    plot_autocorrelation,
    plot_comparison_mode,
    display_historical_insights,
)
from ml_models import (
    plot_forecasting_arima,
    plot_forecasting_prophet,
    plot_forecasting_both,
)
from map_view import create_interactive_map

# --- Initial Setup: Smartly handle pre-built data ---
PARQUET_PATH = "climate_indices.parquet"
if not os.path.exists(PARQUET_PATH):
    if 'build_process_running' not in st.session_state:
        st.session_state.build_process_running = True
        try:
            # Start the build process in the background
            subprocess.Popen(["python3", "build_data.py"])
        except Exception as e:
            st.error(f"Failed to start the data build process: {e}")
            st.session_state.build_process_running = False
    
    # Show a placeholder and auto-refresh using a meta tag
    st.info("Map data is being prepared in the background. The dashboard will automatically refresh when it's ready.")
    st.markdown('<meta http-equiv="refresh" content="15">', unsafe_allow_html=True)
    st.stop()
else:
    # If the file exists and the flag is still in session_state, it means the build just finished.
    if 'build_process_running' in st.session_state:
        del st.session_state['build_process_running']

# --- Session State Initialization ---
if 'selected_fips' not in st.session_state:
    st.session_state.selected_fips = []

# Suppress warnings for a cleaner app
warnings.filterwarnings("ignore")

def main() -> None:
    st.title("U.S. County-Level Drought Analysis")
    st.markdown("Explore and compare key drought indices for any county in the United States. Data is fetched live from NOAA.")

    fips_options = get_county_options()
    gdf = get_geojson()

    with st.sidebar:
        st.header("Dashboard Controls")
        st.info(f"Data is fetched live. Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        if fips_options.empty:
            st.error("County list could not be loaded.")
            st.stop()

        index_choice = st.selectbox("1. Select Climate Index:", ["PDSI", "SPI", "SPEI"], key="index_selectbox")

        fips_code_inputs = st.multiselect(
            "2. Search and Select Counties:",
            options=fips_options.index.tolist(),
            format_func=lambda x: fips_options.get(x, "Unknown County"),
            key="fips_multiselect",
            help="You can type to search, select multiple counties, or click on the map.",
            default=st.session_state.selected_fips
        )
        st.session_state.selected_fips = fips_code_inputs

        st.divider()
        
        theme_options = ["Light", "Dark"]
        current_theme = st.query_params.get("theme", "Light")
        
        def on_theme_change():
            new_theme = st.session_state.theme_selectbox
            st.query_params["theme"] = new_theme
            # No need to call apply_theme, Streamlit handles it
            st.rerun()

        st.selectbox(
            "Select Theme:",
            theme_options,
            index=theme_options.index(current_theme) if current_theme in theme_options else 0,
            key="theme_selectbox",
            on_change=on_theme_change,
        )
        
        analysis_choice = None
        if len(fips_code_inputs) > 1:
            analysis_options = ["Trend Analysis", "Anomaly Detection", "Comparison Mode"]
            analysis_choice = st.selectbox("3. Select Analysis:", analysis_options, key="analysis_selectbox")

    # --- Interactive Map Section ---
    # Provides a checkbox to toggle the map's visibility, reducing clutter.
    show_map = st.checkbox("Show Interactive Map Selector", value=True)
    if show_map and gdf is not None:
        # Display a spinner while the map data is being loaded.
        with st.spinner("Loading map data..."):
            # Caching the map data prevents reloading on every interaction.
            @st.cache_data
            def cached_get_prebuilt_data():
                return get_prebuilt_data_for_map()
            
            map_data = cached_get_prebuilt_data()
        
        # Render the map if data is available.
        if not map_data.empty:
            last_clicked_fips = create_interactive_map(gdf, map_data, index_choice)
            
            # If a county is clicked, add it to the selection and rerun the app.
            if last_clicked_fips and last_clicked_fips not in st.session_state.selected_fips:
                st.session_state.selected_fips.append(last_clicked_fips)
                st.rerun()

    if not fips_code_inputs:
        st.warning("Please select at least one county from the sidebar or map to begin.")
        st.stop()

    # --- Data Fetching for Analysis ---
    full_data = get_live_data_for_counties(fips_code_inputs)
    if full_data.empty:
        st.warning("No data could be fetched for the selected counties.")
        st.stop()

    # --- Main Panel Logic ---
    selected_county_names = [fips_options.get(fips, "Unknown") for fips in fips_code_inputs]
    
    if len(fips_code_inputs) == 1:
        st.header(selected_county_names[0])
        st.caption(f"Climate Index: {index_choice}")
        
        tabs = st.tabs(["Trend Analysis", "Anomaly Detection", "Seasonal Decomposition", "Autocorrelation", "Forecasting"])
        
        fips_code = fips_code_inputs[0]
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
                st.subheader("Forecasting Controls")
                c1, c2, c3 = st.columns(3)
                model_choice = c1.selectbox("Model:", ["ARIMA", "Prophet", "Both"], key="model_selectbox")
                forecast_horizon = c2.slider("Horizon (Months):", 6, 48, 24, 6, key="horizon_slider")
                scenario = c3.selectbox("Scenario:", ["Normal", "Wetter", "Drier"], key="scenario_selectbox", help="Simulate future conditions.")
                
                if model_choice == "Prophet":
                    fig = plot_forecasting_prophet(time_series, index_choice, forecast_horizon, scenario)
                elif model_choice == "ARIMA":
                    fig = plot_forecasting_arima(time_series, index_choice, forecast_horizon, scenario)
                else:
                    fig, metrics = plot_forecasting_both(time_series, index_choice, forecast_horizon, scenario)
                
                st.plotly_chart(fig, use_container_width=True)
                if model_choice == "Both":
                    st.subheader("Model Performance (Last 12 Months)")
                    m1, m2 = st.columns(2)
                    m1.markdown("##### ARIMA"); m1.metric("MAE", f"{metrics['arima_mae']:.4f}"); m1.metric("RMSE", f"{metrics['arima_rmse']:.4f}"); m1.metric("MAPE", f"{metrics['arima_mape']:.2%}")
                    m2.markdown("##### Prophet"); m2.metric("MAE", f"{metrics['prophet_mae']:.4f}"); m2.metric("RMSE", f"{metrics['prophet_rmse']:.4f}"); m2.metric("MAPE", f"{metrics['prophet_mape']:.2%}")

    elif len(fips_code_inputs) > 1:
        st.header(f"{analysis_choice}: {index_choice}")
        st.markdown(f"**Displaying data for:** `{', '.join(selected_county_names)}`")
        
        if analysis_choice == "Comparison Mode":
            plot_comparison_mode(full_data, fips_code_inputs, index_choice)
        else:
            fig = go.Figure()
            for fips_code in fips_code_inputs:
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
    
    st.sidebar.download_button(
        label="Download Selected Data (CSV)",
        data=full_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{index_choice}_data_{'_'.join(fips_code_inputs)}.csv",
        mime="text/csv",
        key="download_button",
    )

    st.markdown("---")
    st.markdown("Data Source: [NOAA National Centers for Environmental Information (NCEI)](https://www.ncei.noa.gov/access/monitoring/nadm/indices)")

if __name__ == "__main__":
    main()