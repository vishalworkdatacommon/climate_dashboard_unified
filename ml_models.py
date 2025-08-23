import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

from typing import Tuple, Dict, Literal

Scenario = Literal["Normal", "Wetter", "Drier"]

def _apply_scenario(ts: pd.Series, scenario: Scenario) -> pd.Series:
    """Adjusts the time series based on the selected scenario."""
    if scenario == "Normal":
        return ts
    
    # Adjust the last data point to simulate a change
    ts_adjusted = ts.copy()
    last_value = ts_adjusted.iloc[-1]
    std_dev = ts_adjusted.std()
    
    if scenario == "Wetter":
        ts_adjusted.iloc[-1] = last_value + (0.5 * std_dev)
    elif scenario == "Drier":
        ts_adjusted.iloc[-1] = last_value - (0.5 * std_dev)
        
    return ts_adjusted

@st.cache_resource
def plot_forecasting_arima(
    ts: pd.Series, index_type: str, n_periods: int = 24, scenario: Scenario = "Normal"
) -> go.Figure:
    
    ts_adjusted = _apply_scenario(ts, scenario)
    
    with st.spinner(f"Finding best ARIMA forecast model for '{scenario}' scenario..."):
        model = pm.auto_arima(
            ts_adjusted.dropna(),
            start_p=1, start_q=1, test="adf", max_p=3, max_q=3, m=12,
            d=None, seasonal=True, start_P=0, D=1, trace=False,
            error_action="ignore", suppress_warnings=True, stepwise=True,
        )
    
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_index = pd.date_range(start=ts_adjusted.index[-1], periods=n_periods + 1, freq="MS")[1:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"Historical Monthly {index_type}"))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode="lines", name="Forecast", line=dict(color="red", width=2.5)))
    fig.add_trace(go.Scatter(x=forecast_index, y=[c[0] for c in conf_int], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_index, y=[c[1] for c in conf_int], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255, 0, 0, 0.2)", name="95% Confidence Interval"))
    
    fig.update_layout(
        title=f"{index_type} ARIMA Forecast ({n_periods}-Month Horizon, {scenario} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )
    return fig

@st.cache_resource
def plot_forecasting_prophet(
    ts: pd.Series, index_type: str, n_periods: int = 24, scenario: Scenario = "Normal"
) -> go.Figure:
    
    ts_adjusted = _apply_scenario(ts, scenario)

    with st.spinner(f"Fitting Prophet model for '{scenario}' scenario..."):
        df = ts_adjusted.reset_index()
        df.rename(columns={"date": "ds", "Value": "y"}, inplace=True)
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
    
    future = model.make_future_dataframe(periods=n_periods, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast[forecast["ds"] > ts_adjusted.index.max()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"Historical Monthly {index_type}", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Forecast", line=dict(color="red", width=2.5)))
    fig.add_trace(go.Scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255, 0, 0, 0.2)", name="95% Confidence Interval"))

    fig.update_layout(
        title=f"{index_type} Prophet Forecast ({n_periods}-Month Horizon, {scenario} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )
    return fig

@st.cache_resource
def plot_forecasting_both(
    ts: pd.Series, index_type: str, n_periods: int = 24, scenario: Scenario = "Normal"
) -> Tuple[go.Figure, Dict[str, float]]:
    
    ts_adjusted = _apply_scenario(ts, scenario)

    with st.spinner(f"Running models for '{scenario}' scenario and calculating metrics..."):
        # --- 1. Calculate Performance Metrics on a Test Set ---
        train = ts[:-12]
        test = ts[-12:]

        # ARIMA eval
        arima_model_eval = pm.auto_arima(train.dropna(), m=12, seasonal=True, stepwise=True, suppress_warnings=True, error_action="ignore")
        arima_pred = arima_model_eval.predict(n_periods=12)

        # Prophet eval
        df_train = train.reset_index(); df_train.rename(columns={"date": "ds", "Value": "y"}, inplace=True)
        prophet_model_eval = Prophet(yearly_seasonality=True).fit(df_train)
        future_eval = prophet_model_eval.make_future_dataframe(periods=12, freq="MS")
        prophet_pred_df = prophet_model_eval.predict(future_eval)
        prophet_pred = prophet_pred_df["yhat"][-12:]

        # Calculate metrics
        metrics = {
            "arima_mae": mean_absolute_error(test, arima_pred),
            "arima_rmse": np.sqrt(mean_squared_error(test, arima_pred)),
            "arima_mape": mean_absolute_percentage_error(test, arima_pred),
            "prophet_mae": mean_absolute_error(test, prophet_pred),
            "prophet_rmse": np.sqrt(mean_squared_error(test, prophet_pred)),
            "prophet_mape": mean_absolute_percentage_error(test, prophet_pred),
        }

        # --- 2. Retrain Models on Full (Adjusted) Data for Final Forecast ---
        # ARIMA
        arima_model_final = pm.auto_arima(ts_adjusted.dropna(), m=12, seasonal=True, stepwise=True, suppress_warnings=True, error_action="ignore")
        arima_forecast, arima_conf_int = arima_model_final.predict(n_periods=n_periods, return_conf_int=True)
        arima_forecast_index = pd.date_range(start=ts_adjusted.index[-1], periods=n_periods + 1, freq="MS")[1:]

        # Prophet
        df_full = ts_adjusted.reset_index(); df_full.rename(columns={"date": "ds", "Value": "y"}, inplace=True)
        prophet_model_final = Prophet(yearly_seasonality=True).fit(df_full)
        future_final = prophet_model_final.make_future_dataframe(periods=n_periods, freq="MS")
        prophet_forecast = prophet_model_final.predict(future_final)
        prophet_forecast_future = prophet_forecast[prophet_forecast["ds"] > ts_adjusted.index.max()]

    # --- 3. Create Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"Historical {index_type}", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=arima_forecast_index, y=arima_forecast, mode="lines", name="ARIMA Forecast", line=dict(color="red", width=2.5)))
    fig.add_trace(go.Scatter(x=arima_forecast_index, y=[c[0] for c in arima_conf_int], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=arima_forecast_index, y=[c[1] for c in arima_conf_int], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255, 0, 0, 0.2)", name="ARIMA 95% CI"))
    fig.add_trace(go.Scatter(x=prophet_forecast_future["ds"], y=prophet_forecast_future["yhat"], mode="lines", name="Prophet Forecast", line=dict(color="green", width=2.5)))
    fig.add_trace(go.Scatter(x=prophet_forecast_future["ds"], y=prophet_forecast_future["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=prophet_forecast_future["ds"], y=prophet_forecast_future["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 255, 0, 0.2)", name="Prophet 95% CI"))

    fig.update_layout(
        title=f"{index_type} ARIMA vs. Prophet Forecast ({n_periods}-Month, {scenario} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )

    return fig, metrics