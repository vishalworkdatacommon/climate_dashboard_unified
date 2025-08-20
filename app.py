# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import warnings
import plotly.graph_objects as go
# from streamlit_folium import folium_static

# --- Custom Modules ---
from data_loader import get_live_data
from plotting import (
    plot_trend_analysis,
    plot_anomaly_detection,
    plot_seasonal_decomposition,
    plot_autocorrelation
)
from ml_models import (
    plot_forecasting_arima,
    plot_forecasting_prophet,
    plot_forecasting_both
)
# from map_view import create_map

# --- Page Configuration ---
st.set_page_config(
    page_title="U.S. County-Level Drought Analysis",
    page_icon="💧",
    layout="wide",
)

# Suppress warnings for a cleaner app
warnings.filterwarnings("ignore")

def main():
    st.title("💧 U.S. County-Level Drought Analysis")
    st.markdown("Explore and compare key drought indices for any county in the United States. This dashboard uses pre-built data, updated periodically.")

    full_data, fips_options, gdf, last_updated = get_live_data()

    with st.sidebar:
        st.header("Dashboard Controls")
        if last_updated:
            st.info(f"Data last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        if not full_data.empty and not fips_options.empty:
            index_choice = st.selectbox("1. Select Climate Index:", options=sorted(full_data['index_type'].unique()), key="index_selectbox")
            
            default_selection = [fips_options.index.tolist()[0]] if not fips_options.empty else []
            fips_code_inputs = st.multiselect("2. Select County/Counties:", options=fips_options.index.tolist(), format_func=lambda x: fips_options.loc[x]['display_name'], default=default_selection, key="fips_multiselect")

            if len(fips_code_inputs) > 1:
                analysis_options = ["Trend Analysis", "Anomaly Detection"]
            else:
                analysis_options = ["Trend Analysis", "Anomaly Detection", "Seasonal Decomposition", "Autocorrelation", "Forecasting"]
            analysis_choice = st.selectbox("3. Select Analysis:", analysis_options, key="analysis_selectbox")

            model_choice = "ARIMA"
            if analysis_choice == "Forecasting":
                model_choice = st.selectbox("4. Select Forecasting Model:", ["ARIMA", "Prophet", "Both"], key="model_selectbox")

            if fips_code_inputs:
                csv_data = full_data[full_data['countyfips'].isin(fips_code_inputs)]
                st.download_button(label="Download Selected Data (CSV)", data=csv_data.to_csv(index=False).encode('utf-8'), file_name=f"{index_choice}_data_{'_'.join(fips_code_inputs)}.csv", mime="text/csv", key="download_button")
        else:
            st.error("Data could not be loaded. The dashboard cannot be displayed.")
            st.stop()

    # --- Map-based County Selection ---
    # st.subheader("Select a Month to Display on the Map")
    # latest_date = full_data['date'].max()
    # selected_date = st.date_input("Date", value=latest_date, min_value=full_data['date'].min(), max_value=latest_date, key="date_selector")
    
    # if gdf is not None:
    #     folium_map = create_map(gdf, full_data, index_choice, selected_date)
    #     st.subheader("Click on a county to select it for analysis")
    #     folium_static(folium_map, width=1400)

    if not fips_code_inputs:
        st.warning("Please select at least one county from the sidebar to begin.")
        st.stop()

    selected_county_names = [fips_options.loc[fips]['display_name'] for fips in fips_code_inputs]
    st.header(f"{analysis_choice}: {index_choice}")
    st.markdown(f"**Displaying data for:** `{', '.join(selected_county_names)}`")

    with st.expander("About This Analysis"):
        analysis_explanations = {
            "Trend Analysis": "This chart shows the monthly index values over time, along with a 12-month rolling average. It helps you see the long-term trends and patterns in the data, smoothing out short-term fluctuations.",
            "Anomaly Detection": "This analysis highlights periods that were unusually wet or dry. The red dots mark months where the index value was significantly different (more than two standard deviations) from the 12-month rolling average, indicating extreme conditions.",
            "Seasonal Decomposition": "This technique breaks down the time series into three components: the long-term **Trend**, the repeating annual **Seasonal** cycle, and the irregular **Residual** noise. This helps to understand the underlying patterns driving the index values.",
            "Autocorrelation": "These plots show how correlated the index is with itself at different points in time. The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) are used in statistical modeling to identify how much past values influence future values.",
            "Forecasting": "This chart displays a 24-month forecast of the climate index. You can choose between two models: **auto-ARIMA**, a powerful statistical model for time series data, and **Prophet**, a forecasting tool from Facebook that excels with seasonal data. The shaded area represents the 95% confidence interval."
        }
        st.info(analysis_explanations[analysis_choice])

    fig = go.Figure()
    for fips_code in fips_code_inputs:
        filtered_df = full_data[(full_data['countyfips'] == fips_code) & (full_data['index_type'] == index_choice)]
        if not filtered_df.empty:
            time_series = filtered_df.set_index('date')['Value'].asfreq('MS')
            county_name = fips_options.loc[fips_code]['display_name']
            
            if len(fips_code_inputs) == 1:
                if analysis_choice == "Forecasting":
                    if model_choice == "Prophet":
                        fig = plot_forecasting_prophet(time_series, index_choice)
                    elif model_choice == "ARIMA":
                        fig = plot_forecasting_arima(time_series, index_choice)
                    elif model_choice == "ARIMA":
                        fig = plot_forecasting_arima(time_series, index_choice)
                    else: # Both
                        fig, metrics = plot_forecasting_both(time_series, index_choice)
                else:
                    plot_function_mapping = {
                        "Trend Analysis": plot_trend_analysis, "Anomaly Detection": plot_anomaly_detection,
                        "Seasonal Decomposition": plot_seasonal_decomposition, "Autocorrelation": plot_autocorrelation
                    }
                    fig = plot_function_mapping[analysis_choice](time_series, index_choice)
                break
            
            if analysis_choice == "Trend Analysis":
                rolling_avg = time_series.rolling(window=12).mean()
                fig.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name=f'{county_name}'))
                fig.add_trace(go.Scatter(x=rolling_avg.index, y=rolling_avg, mode='lines', name=f'{county_name} (Rolling Avg)', line=dict(dash='dash')))
            elif analysis_choice == "Anomaly Detection":
                rolling_mean = time_series.rolling(window=12).mean()
                rolling_std = time_series.rolling(window=12).std()
                anomalies = time_series[(time_series > rolling_mean + (2 * rolling_std)) | (time_series < rolling_mean - (2 * rolling_std))]
                fig.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name=f'{county_name}'))
                fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode='markers', name=f'{county_name} Anomaly', marker=dict(symbol='x')))

    if len(fips_code_inputs) > 1:
        fig.update_layout(title=f'{index_choice} {analysis_choice}', xaxis_title='Year', yaxis_title=f'{index_choice} Value', legend_title="Counties", template='plotly_white', font=dict(size=14), height=600)

    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
        if 'metrics' in locals():
            st.subheader("Model Performance on Historical Data (Last 12 Months)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### ARIMA")
                st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['arima_mae']:.4f}")
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics['arima_rmse']:.4f}")
            with col2:
                st.markdown("##### Prophet")
                st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['prophet_mae']:.4f}")
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics['prophet_rmse']:.4f}")

    else:
        st.warning("No data available for the selected combination of counties and index. Please make another selection.")

    st.markdown("---")
    st.markdown("Data Source: [NOAA National Centers for Environmental Information (NCEI)](https://www.ncei.noa.gov/access/monitoring/nadm/indices)")
    st.markdown("This application provides a comparative tool for various climate indices, allowing for a deeper understanding of regional climate patterns.")

if __name__ == "__main__":
    main()