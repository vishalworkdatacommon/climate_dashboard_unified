# -*- coding: utf-8 -*-
import streamlit as st
import warnings
import plotly.graph_objects as go
from datetime import datetime

# --- Custom Modules ---
from data_loader import get_county_options, get_live_data_for_counties, get_geojson
from plotting import (
    plot_trend_analysis,
    plot_anomaly_detection,
    plot_seasonal_decomposition,
    plot_autocorrelation,
)
from ml_models import (
    plot_forecasting_arima,
    plot_forecasting_prophet,
    plot_forecasting_both,
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
        "Explore and compare key drought indices for any county in the United States. Data is fetched live from NOAA."
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
    # gdf = get_geojson() # Reserved for future map implementation

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
            "2. Search and Select Counties:",
            options=fips_options.index.tolist(),
            format_func=lambda x: fips_options.get(x, "Unknown County"),
            key="fips_multiselect",
            help="You can type to search for a county and select multiple counties for comparison.",
        )

        if len(fips_code_inputs) > 1:
            analysis_options = ["Trend Analysis", "Anomaly Detection"]
        else:
            analysis_options = [
                "Trend Analysis",
                "Anomaly Detection",
                "Seasonal Decomposition",
                "Autocorrelation",
                "Forecasting",
            ]
        analysis_choice = st.selectbox(
            "3. Select Analysis:", analysis_options, key="analysis_selectbox"
        )

        model_choice = "ARIMA"
        forecast_horizon = 24
        if analysis_choice == "Forecasting":
            model_choice = st.selectbox(
                "4. Select Forecasting Model:",
                ["ARIMA", "Prophet", "Both"],
                key="model_selectbox",
            )
            forecast_horizon = st.slider(
                "5. Select Forecast Horizon (Months):",
                min_value=6,
                max_value=48,
                value=24,
                step=6,
                key="horizon_slider",
            )

    if not fips_code_inputs:
        st.warning("Please select at least one county from the sidebar to begin.")
        st.stop()

    # --- Data Fetching ---
    full_data = get_live_data_for_counties(fips_code_inputs)

    if full_data.empty:
        st.warning("No data could be fetched for the selected counties. Please try other selections or check back later.")
        st.stop()

    # --- UI Rendering ---
    selected_county_names = [fips_options.get(fips, "Unknown") for fips in fips_code_inputs]
    st.header(f"{analysis_choice}: {index_choice}")
    st.markdown(f"**Displaying data for:** `{', '.join(selected_county_names)}`")

    with st.expander("About This Analysis"):
        analysis_explanations = {
            "Trend Analysis": "This chart shows the monthly index values over time, along with a 12-month rolling average to visualize long-term trends.",
            "Anomaly Detection": "This analysis highlights periods of extreme conditions by marking points that deviate significantly (more than two standard deviations) from the 12-month rolling average.",
            "Seasonal Decomposition": "This breaks down the time series into its core components: the long-term **Trend**, the annual **Seasonal** cycle, and irregular **Residuals**.",
            "Autocorrelation": "These plots (ACF and PACF) show how past values of the index correlate with future values, which is key for building forecasting models.",
            "Forecasting": f"This chart displays a {forecast_horizon}-month forecast using **auto-ARIMA** and/or Facebook's **Prophet** model. The shaded area is the 95% confidence interval.",
        }
        st.info(analysis_explanations.get(analysis_choice, "Analysis description not available."))

    # --- Download Button ---
    st.sidebar.download_button(
        label="Download Selected Data (CSV)",
        data=full_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{index_choice}_data_{'_'.join(fips_code_inputs)}.csv",
        mime="text/csv",
        key="download_button",
    )

    # --- Plotting ---
    fig = go.Figure()
    for fips_code in fips_code_inputs:
        filtered_df = full_data[
            (full_data["countyfips"] == fips_code)
            & (full_data["index_type"] == index_choice)
        ]
        if not filtered_df.empty:
            time_series = filtered_df.set_index("date")["Value"].asfreq("MS")
            county_name = fips_options.get(fips_code, "Unknown")

            if len(fips_code_inputs) == 1:
                if analysis_choice == "Forecasting":
                    if model_choice == "Prophet":
                        fig = plot_forecasting_prophet(time_series, index_choice, forecast_horizon)
                    elif model_choice == "ARIMA":
                        fig = plot_forecasting_arima(time_series, index_choice, forecast_horizon)
                    else:  # Both
                        fig, metrics = plot_forecasting_both(time_series, index_choice, forecast_horizon)
                else:
                    plot_function_mapping = {
                        "Trend Analysis": plot_trend_analysis,
                        "Anomaly Detection": plot_anomaly_detection,
                        "Seasonal Decomposition": plot_seasonal_decomposition,
                        "Autocorrelation": plot_autocorrelation,
                    }
                    fig = plot_function_mapping[analysis_choice](time_series, index_choice)
                break

            # Multi-county plots (limited to Trend and Anomaly)
            if analysis_choice == "Trend Analysis":
                fig.add_trace(go.Scatter(x=time_series.index, y=time_series, mode="lines", name=county_name))
            elif analysis_choice == "Anomaly Detection":
                rolling_mean = time_series.rolling(window=12).mean()
                rolling_std = time_series.rolling(window=12).std()
                anomalies = time_series[(time_series > rolling_mean + (2 * rolling_std)) | (time_series < rolling_mean - (2 * rolling_std))]
                fig.add_trace(go.Scatter(x=time_series.index, y=time_series, mode="lines", name=county_name))
                fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode="markers", name=f"{county_name} Anomaly", marker=dict(symbol="x")))

    if len(fips_code_inputs) > 1:
        fig.update_layout(
            title=f"{index_choice} {analysis_choice}",
            xaxis_title="Year",
            yaxis_title=f"{index_choice} Value",
            legend_title="Counties",
            template="plotly_white",
            font=dict(size=14),
            height=600,
        )

    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
        if "metrics" in locals() and analysis_choice == "Forecasting" and model_choice == "Both":
            st.subheader("Model Performance on Historical Data (Last 12 Months)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### ARIMA"); st.metric("MAE", f"{metrics['arima_mae']:.4f}"); st.metric("RMSE", f"{metrics['arima_rmse']:.4f}")
            with col2:
                st.markdown("##### Prophet"); st.metric("MAE", f"{metrics['prophet_mae']:.4f}"); st.metric("RMSE", f"{metrics['prophet_rmse']:.4f}")
    else:
        st.warning("No data available for the selected combination of counties and index. Please make another selection.")

    st.markdown("---")
    st.markdown("Data Source: [NOAA National Centers for Environmental Information (NCEI)](https://www.ncei.noa.gov/access/monitoring/nadm/indices)")

if __name__ == "__main__":
    main()