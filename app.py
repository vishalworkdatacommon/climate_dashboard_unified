import streamlit as st
import warnings
import plotly.graph_objects as go
from datetime import datetime
import os
import subprocess

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

# --- Initial Setup: Check for pre-built data ---
PARQUET_PATH = "climate_indices.parquet"
if not os.path.exists(PARQUET_PATH):
    st.warning(f"Map data file not found. Automatically running build script... (This may take a few minutes)")
    try:
        # Use subprocess to run the build script
        process = subprocess.run(
            ["python3", "build_data.py"],
            capture_output=True,
            text=True,
            check=True
        )
        st.success("Map data file built successfully!")
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error("Failed to build the map data file. The application may not function correctly.")
        st.code(e.stderr)
        st.stop()


# --- Session State Initialization ---
if 'selected_fips' not in st.session_state:
    st.session_state.selected_fips = []


# --- Page Configuration ---
st.set_page_config(
    page_title="U.S. County-Level Drought Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suppress warnings for a cleaner app
warnings.filterwarnings("ignore")


def main() -> None:
    st.title("U.S. County-Level Drought Analysis")
    st.markdown(
        "Explore and compare key drought indices for any county in the United States. Data is fetched live from NOAA."
    )

    fips_options = get_county_options()
    gdf = get_geojson()

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

        # Use session state for the multiselect default
        fips_code_inputs = st.multiselect(
            "2. Search and Select Counties:",
            options=fips_options.index.tolist(),
            format_func=lambda x: fips_options.get(x, "Unknown County"),
            key="fips_multiselect",
            help="You can type to search, select multiple counties, or click on the map.",
            default=st.session_state.selected_fips
        )
        st.session_state.selected_fips = fips_code_inputs

        # Analysis selection is now in the main panel for single-county view
        analysis_choice = None
        if len(fips_code_inputs) > 1:
            analysis_options = ["Trend Analysis", "Anomaly Detection", "Comparison Mode"]
            analysis_choice = st.selectbox(
                "3. Select Analysis:", analysis_options, key="analysis_selectbox"
            )

    # --- Interactive Map ---
    show_map = st.checkbox("Show Interactive Map Selector")
    if show_map and gdf is not None:
        # Use the fast, pre-built data for the map's choropleth
        with st.spinner("Loading map data..."):
            map_data = get_prebuilt_data_for_map()
        
        if not map_data.empty:
            last_clicked_fips = create_interactive_map(gdf, map_data, index_choice)
            if last_clicked_fips and last_clicked_fips not in st.session_state.selected_fips:
                st.session_state.selected_fips.append(last_clicked_fips)
                st.rerun() # Rerun to update the multiselect widget

    if not fips_code_inputs:
        st.warning("Please select at least one county from the sidebar or map to begin.")
        st.stop()

    # --- Data Fetching for Analysis ---
    full_data = get_live_data_for_counties(fips_code_inputs)
    if full_data.empty:
        st.warning("No data could be fetched for the selected counties. Please try other selections or check back later.")
        st.stop()

    # --- Main Panel Logic ---
    selected_county_names = [fips_options.get(fips, "Unknown") for fips in fips_code_inputs]
    
    if len(fips_code_inputs) == 1:
        st.header(selected_county_names[0])
        st.caption(f"Climate Index: {index_choice}")
        # ... (rest of the single-county tabbed layout)
    elif len(fips_code_inputs) > 1:
        st.header(f"{analysis_choice}: {index_choice}")
        st.markdown(f"**Displaying data for:** `{', '.join(selected_county_names)}`")
        # ... (rest of the multi-county layout)

    # (The rest of the app logic for plotting and analysis remains the same)
    # ... (omitted for brevity, but it's the same as the previous version)
    if len(fips_code_inputs) == 1:
        # Single-county view with tabs
        (
            tab1, tab2, tab3, tab4, tab5
        ) = st.tabs([
            "Trend Analysis", "Anomaly Detection", "Seasonal Decomposition", 
            "Autocorrelation", "Forecasting"
        ])

        fips_code = fips_code_inputs[0]
        filtered_df = full_data[
            (full_data["countyfips"] == fips_code)
            & (full_data["index_type"] == index_choice)
        ]
        if not filtered_df.empty:
            time_series = filtered_df.set_index("date")["Value"].asfreq("MS")

            with tab1:
                fig = plot_trend_analysis(time_series, index_choice)
                st.plotly_chart(fig, use_container_width=True)
                display_historical_insights(time_series)
            with tab2:
                fig = plot_anomaly_detection(time_series, index_choice)
                st.plotly_chart(fig, use_container_width=True)
                display_historical_insights(time_series)
            with tab3:
                fig = plot_seasonal_decomposition(time_series, index_choice)
                st.plotly_chart(fig, use_container_width=True)
            with tab4:
                fig = plot_autocorrelation(time_series, index_choice)
                st.plotly_chart(fig, use_container_width=True)
            with tab5:
                st.subheader("Forecasting Controls")
                col1, col2, col3 = st.columns(3)
                with col1:
                    model_choice = st.selectbox("Model:", ["ARIMA", "Prophet", "Both"], key="model_selectbox")
                with col2:
                    forecast_horizon = st.slider("Horizon (Months):", 6, 48, 24, 6, key="horizon_slider")
                with col3:
                    scenario = st.selectbox("Scenario:", ["Normal", "Wetter", "Drier"], key="scenario_selectbox", help="Simulate future conditions.")

                if model_choice == "Prophet":
                    fig = plot_forecasting_prophet(time_series, index_choice, forecast_horizon, scenario)
                elif model_choice == "ARIMA":
                    fig = plot_forecasting_arima(time_series, index_choice, forecast_horizon, scenario)
                else:
                    fig, metrics = plot_forecasting_both(time_series, index_choice, forecast_horizon, scenario)
                
                st.plotly_chart(fig, use_container_width=True)
                if model_choice == "Both":
                    st.subheader("Model Performance on Historical Data (Last 12 Months)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### ARIMA"); st.metric("MAE", f"{metrics['arima_mae']:.4f}"); st.metric("RMSE", f"{metrics['arima_rmse']:.4f}"); st.metric("MAPE", f"{metrics['arima_mape']:.2%}")
                    with col2:
                        st.markdown("##### Prophet"); st.metric("MAE", f"{metrics['prophet_mae']:.4f}"); st.metric("RMSE", f"{metrics['prophet_rmse']:.4f}"); st.metric("MAPE", f"{metrics['prophet_mape']:.2%}")

    elif len(fips_code_inputs) > 1:
        # Multi-county view
        if analysis_choice == "Comparison Mode":
            plot_comparison_mode(full_data, fips_code_inputs, index_choice)
        else:
            fig = go.Figure()
            for fips_code in fips_code_inputs:
                filtered_df = full_data[
                    (full_data["countyfips"] == fips_code)
                    & (full_data["index_type"] == index_choice)
                ]
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
    
    # --- Download Button ---
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
