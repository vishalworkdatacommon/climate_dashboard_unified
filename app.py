# -*- coding: utf-8 -*-
import streamlit as st
import warnings
import plotly.graph_objects as go
from datetime import datetime

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
from ml_models import (
    plot_forecasting_arima,
    plot_forecasting_prophet,
    plot_forecasting_lightgbm,
    plot_forecasting_comparison,
    generate_ai_summary,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="U.S. County-Level Drought Analysis",
    page_icon=None,
    layout="wide",
)

# Suppress warnings for a cleaner app
warnings.filterwarnings("ignore")


def main() -> None:
    st.title("U.S. County-Level Drought Analysis")
    st.markdown(
        "Explore and compare key drought indices for any county in the United States. Data is fetched live from CDC."
    )

    with st.expander("About the Climate Indices"):
        st.markdown(
            """
            - **PDSI (Palmer Drought Severity Index):** Measures long-term drought based on temperature and precipitation data.
            - **SPI (Standardized Precipitation Index):** Shows how precipitation compares to the long-term average for a given period.
            - **SPEI (Standardized Precipitation-Evapotranspiration Index):** Similar to SPI, but also includes the effect of temperature on water demand.
            """
        )

    fips_options = get_county_options()

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
            st.success("Start Here! 👆")

        # Analysis selection is now in the main panel for single-county view
        analysis_choice = None
        if len(fips_code_inputs) > 1:
            analysis_options = ["Trend Analysis", "Anomaly Detection", "Comparison Mode"]
            analysis_choice = st.selectbox(
                "3. Select Analysis:", analysis_options, key="analysis_selectbox"
            )

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

    # --- Data Fetching for Selected Counties---
    full_data = get_live_data_for_counties(fips_code_inputs)

    if full_data.empty:
        st.warning("No data could be fetched for the selected counties. Please try other selections or check back later.")
        st.stop()

    # --- UI Rendering ---
    selected_county_names = [fips_options.get(fips, "Unknown") for fips in fips_code_inputs]
    
    # --- Main Panel Logic ---
    if len(fips_code_inputs) == 1:
        # Single-county view with tabs
        st.header(selected_county_names[0])
        st.caption(f"Climate Index: {index_choice}")

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
                col1, col2 = st.columns(2)
                with col1:
                    model_selection = st.selectbox(
                        "Model:", 
                        [
                            "ARIMA", "Prophet", "LightGBM", 
                            "ARIMA vs. Prophet", "ARIMA vs. LightGBM", "Prophet vs. LightGBM",
                            "All Models"
                        ], 
                        key="model_selection"
                    )
                with col2:
                    forecast_horizon = st.slider("Horizon (Months):", 6, 48, 24, 6, key="horizon_slider")

                # --- Scenario Analysis ---
                scenario_params = {} # Default to no scenario
                if st.toggle("Enable Scenario Analysis", key="scenario_toggle", help="Simulate future conditions by applying a shock or trend to the historical data before forecasting."):
                    st.subheader("Scenario Analysis Controls")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        scenario_type = st.selectbox("Scenario Type:", ["Sudden Shock", "Gradual Trend"], key="scenario_type")
                    with col2:
                        magnitude = st.slider("Magnitude of Change (%):", -50, 50, 10, 5, key="magnitude_slider")
                    with col3:
                        duration = st.slider("Duration of Change (Months):", 1, 12, 3, 1, key="duration_slider")

                    scenario_params = {
                        "type": scenario_type,
                        "magnitude": magnitude / 100.0,
                        "duration": duration
                    }
                
                model_map = {
                    "ARIMA": ["ARIMA"],
                    "Prophet": ["Prophet"],
                    "LightGBM": ["LightGBM"],
                    "ARIMA vs. Prophet": ["ARIMA", "Prophet"],
                    "ARIMA vs. LightGBM": ["ARIMA", "LightGBM"],
                    "Prophet vs. LightGBM": ["Prophet", "LightGBM"],
                    "All Models": ["ARIMA", "Prophet", "LightGBM"]
                }
                models_to_run = model_map[model_selection]

                forecast_df = None
                if len(models_to_run) == 1:
                    model_choice = models_to_run[0]
                    if model_choice == "Prophet":
                        fig, forecast_df = plot_forecasting_prophet(time_series, index_choice, forecast_horizon, scenario_params)
                    elif model_choice == "ARIMA":
                        fig, forecast_df = plot_forecasting_arima(time_series, index_choice, forecast_horizon, scenario_params)
                    elif model_choice == "LightGBM":
                        fig, forecast_df = plot_forecasting_lightgbm(time_series, index_choice, forecast_horizon, scenario_params)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- AI Summary for Single Model ---
                    st.subheader("AI-Powered Summary")
                    summary = generate_ai_summary(
                        county_name=selected_county_names[0], index_type=index_choice,
                        models_to_run=models_to_run, metrics={}, # No metrics for single model run
                        forecast_horizon=forecast_horizon, time_series=time_series,
                        forecast_df=forecast_df
                    )
                    st.markdown(summary)

                else:
                    fig, metrics, forecast_df = plot_forecasting_comparison(
                        time_series, index_choice, models_to_run, forecast_horizon, scenario_params
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- AI Summary ---
                    st.subheader("AI-Powered Summary")
                    summary = generate_ai_summary(
                        county_name=selected_county_names[0],
                        index_type=index_choice,
                        models_to_run=models_to_run,
                        metrics=metrics,
                        forecast_horizon=forecast_horizon,
                        time_series=time_series,
                        forecast_df=forecast_df
                    )
                    st.markdown(summary)

                    st.subheader("Model Performance (from Cross-Validation)")
                    cols = st.columns(len(models_to_run))
                    for i, model_name in enumerate(models_to_run):
                        with cols[i]:
                            st.markdown(f"##### {model_name}")
                            st.metric("MAE", f"{metrics.get(f'{model_name.lower()}_mae', 0):.4f}")
                            st.metric("RMSE", f"{metrics.get(f'{model_name.lower()}_rmse', 0):.4f}")
                            st.metric("MAPE", f"{metrics.get(f'{model_name.lower()}_mape', 0):.2%}")
                            st.metric("sMAPE", f"{metrics.get(f'{model_name.lower()}_smape', 0):.2%}")

    elif len(fips_code_inputs) > 1:
        # Multi-county view
        st.header(f"{analysis_choice}: {index_choice}")
        st.markdown(f"**Displaying data for:** `{', '.join(selected_county_names)}`")

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
    st.markdown("Data Source: [CDC Wonder](https://data.cdc.gov/resource)")

if __name__ == "__main__":
    main()