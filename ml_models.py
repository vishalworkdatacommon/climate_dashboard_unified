import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import lightgbm as lgb

from typing import Tuple, Dict, Literal

def smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error."""
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def _apply_scenario(ts: pd.Series, scenario_params: Dict) -> pd.Series:
    """Adjusts the time series based on the selected scenario."""
    ts_adjusted = ts.copy()
    magnitude = scenario_params.get("magnitude", 0)
    duration = scenario_params.get("duration", 1)
    scenario_type = scenario_params.get("type", "Sudden Shock")

    if magnitude == 0:
        return ts_adjusted

    last_value = ts_adjusted.iloc[-1]
    change = last_value * magnitude

    if scenario_type == "Sudden Shock":
        ts_adjusted.iloc[-1] += change
    elif scenario_type == "Gradual Trend":
        gradual_change = np.linspace(0, change, duration)
        if len(ts_adjusted) >= duration:
            ts_adjusted.iloc[-duration:] += gradual_change
        else:
            # If the series is shorter than the duration, apply over the whole series
            gradual_change = np.linspace(0, change, len(ts_adjusted))
            ts_adjusted += gradual_change
            
    return ts_adjusted

@st.cache_resource(show_spinner=False)
def plot_forecasting_arima(
    ts: pd.Series, index_type: str, n_periods: int = 24, scenario_params: Dict = None
) -> Tuple[go.Figure, pd.DataFrame]:
    
    ts_adjusted = _apply_scenario(ts, scenario_params or {})
    scenario_str = f"{scenario_params['type']} ({scenario_params['magnitude']*100:.0f}%)" if scenario_params and scenario_params.get('magnitude') != 0 else "Normal Forecast"
    
    with st.spinner(f"Searching for best ARIMA model for '{scenario_str}' scenario... This may take a moment."):
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
        title=f"{index_type} ARIMA Forecast ({n_periods}-Month Horizon, {scenario_str} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast})
    return fig, forecast_df

@st.cache_resource(show_spinner=False)
def plot_forecasting_prophet(
    ts: pd.Series, index_type: str, n_periods: int = 24, scenario_params: Dict = None
) -> Tuple[go.Figure, pd.DataFrame]:
    
    ts_adjusted = _apply_scenario(ts, scenario_params or {})
    scenario_str = f"{scenario_params['type']} ({scenario_params['magnitude']*100:.0f}%)" if scenario_params and scenario_params.get('magnitude') != 0 else "Normal Forecast"

    with st.spinner(f"Fitting Prophet model for '{scenario_str}' scenario..."):
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
        title=f"{index_type} Prophet Forecast ({n_periods}-Month Horizon, {scenario_str} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )
    return fig, forecast_future[['ds', 'yhat']]




def _create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time series features from a dataframe's date index.
    """
    df = df.copy()
    df['date'] = df.index
    
    # Date-based features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    return df[['month', 'year', 'quarter', 'dayofyear', 'weekofyear']]

@st.cache_resource(show_spinner=False)
def plot_forecasting_lightgbm(
    ts: pd.Series, index_type: str, n_periods: int = 24, scenario_params: Dict = None
) -> Tuple[go.Figure, pd.DataFrame]:
    
    ts_adjusted = _apply_scenario(ts, scenario_params or {})
    scenario_str = f"{scenario_params['type']} ({scenario_params['magnitude']*100:.0f}%)" if scenario_params and scenario_params.get('magnitude') != 0 else "Normal Forecast"
    
    with st.spinner(f"Training LightGBM model with advanced features for '{scenario_str}' scenario..."):
        # --- Feature Engineering ---
        df = ts_adjusted.to_frame(name='Value')
        
        # Lag features
        for lag in [1, 3, 6, 12]: df[f'lag_{lag}'] = df['Value'].shift(lag)
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = df['Value'].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['Value'].shift(1).rolling(window=window).std()
        df.dropna(inplace=True)

        X = pd.concat([_create_date_features(df), df.drop(columns=['Value'])], axis=1)
        y = df['Value']
        
        model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=1000, learning_rate=0.05,
                                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                                  verbose=-1, n_jobs=-1, seed=42)
        
        model.fit(X, y, eval_set=[(X, y)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        
        # --- Model Reliability Check ---
        train_pred = model.predict(X)
        train_smape = smape(y, train_pred)
        
        if np.isinf(train_smape) or pd.isna(train_smape):
            st.warning("LightGBM produced an unstable forecast. Falling back to a more robust ARIMA model.")
            return plot_forecasting_arima(ts, index_type, n_periods, scenario_params)

        # --- Forecasting ---
        forecast_values = []
        current_features = df.iloc[-1:].copy()

        for date in pd.date_range(start=ts_adjusted.index[-1], periods=n_periods + 1, freq="MS")[1:]:
            future_date_features = _create_date_features(pd.DataFrame(index=[date]))
            combined_features = pd.concat([future_date_features.reset_index(drop=True), current_features.drop(columns=['Value']).reset_index(drop=True)], axis=1)[X.columns]
            pred = model.predict(combined_features)[0]
            forecast_values.append(pred)
            
            new_row = current_features.iloc[[-1]].copy()
            new_row.index = [date]
            new_row['Value'] = pred
            full_series = pd.concat([df['Value'], pd.Series(forecast_values, index=pd.date_range(start=ts_adjusted.index[-1], periods=len(forecast_values) + 1, freq="MS")[1:])])
            for lag in [1, 3, 6, 12]: new_row[f'lag_{lag}'] = full_series.shift(lag).iloc[-1]
            for window in [3, 6, 12]:
                new_row[f'rolling_mean_{window}'] = full_series.shift(1).rolling(window=window).mean().iloc[-1]
                new_row[f'rolling_std_{window}'] = full_series.shift(1).rolling(window=window).std().iloc[-1]
            current_features = new_row

    future_dates = pd.date_range(start=ts_adjusted.index[-1], periods=n_periods + 1, freq="MS")[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"Historical Monthly {index_type}"))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, mode="lines", name="Forecast", line=dict(color="purple", width=2.5)))
    
    fig.update_layout(
        title=f"{index_type} LightGBM Forecast ({n_periods}-Month Horizon, {scenario_str} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    return fig, forecast_df


@st.cache_resource(show_spinner="Running cross-validation and generating forecasts...")
def plot_forecasting_comparison(
    ts: pd.Series, index_type: str, models: list, n_periods: int = 24, scenario_params: Dict = None
) -> Tuple[go.Figure, Dict[str, float], pd.DataFrame]:
    
    ts_adjusted = _apply_scenario(ts, scenario_params or {})
    scenario_str = f"{scenario_params['type']} ({scenario_params['magnitude']*100:.0f}%)" if scenario_params and scenario_params.get('magnitude') != 0 else "Normal Forecast"

    with st.spinner(f"Performing optimized cross-validation for {', '.join(models)}..."):
        # --- 1. Optimized Time Series Cross-Validation ---
        from sklearn.model_selection import TimeSeriesSplit
        
        # Use fewer splits for a faster, yet still indicative, cross-validation
        ts_cv = TimeSeriesSplit(n_splits=2, test_size=12)
        metrics = {model: [] for model in models}

        for train_index, test_index in ts_cv.split(ts):
            train = ts.iloc[train_index]
            test = ts.iloc[test_index]

            if "ARIMA" in models:
                try:
                    arima_model_eval = pm.auto_arima(train.dropna(), m=12, seasonal=True, stepwise=True, suppress_warnings=True, error_action="ignore")
                    arima_pred = arima_model_eval.predict(n_periods=len(test))
                    metrics["ARIMA"].append({"mae": mean_absolute_error(test, arima_pred), "rmse": np.sqrt(mean_squared_error(test, arima_pred)), "mape": mean_absolute_percentage_error(test, arima_pred), "smape": smape(test, arima_pred)})
                except Exception:
                    continue # Ignore failures in CV folds

            if "Prophet" in models:
                try:
                    df_train_prophet = train.reset_index().rename(columns={"date": "ds", "Value": "y"})
                    prophet_model_eval = Prophet(yearly_seasonality=True).fit(df_train_prophet)
                    future_eval_prophet = prophet_model_eval.make_future_dataframe(periods=len(test), freq="MS")
                    prophet_pred_df = prophet_model_eval.predict(future_eval_prophet)
                    prophet_pred = prophet_pred_df["yhat"][-len(test):]
                    metrics["Prophet"].append({"mae": mean_absolute_error(test, prophet_pred), "rmse": np.sqrt(mean_squared_error(test, prophet_pred)), "mape": mean_absolute_percentage_error(test, prophet_pred), "smape": smape(test, prophet_pred)})
                except Exception:
                    continue

            if "LightGBM" in models:
                try:
                    train_df = train.to_frame(name='Value')
                    for lag in [1, 3, 6, 12]: train_df[f'lag_{lag}'] = train_df['Value'].shift(lag)
                    for window in [3, 6, 12]:
                        train_df[f'rolling_mean_{window}'] = train_df['Value'].shift(1).rolling(window=window).mean()
                        train_df[f'rolling_std_{window}'] = train_df['Value'].shift(1).rolling(window=window).std()
                    train_df.dropna(inplace=True)
                    
                    X_train = pd.concat([_create_date_features(train_df), train_df.drop(columns=['Value'])], axis=1)
                    y_train = train_df['Value']
                    if X_train.empty: continue

                    # Use standard, good-performing parameters instead of a slow search
                    lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=500, learning_rate=0.05, verbose=-1, n_jobs=-1, seed=42)
                    lgbm.fit(X_train, y_train)
                    
                    test_predictions = []
                    history = train_df.copy()
                    for test_date in test.index:
                        date_features = _create_date_features(pd.DataFrame(index=[test_date]))
                        lag_rolling_features = history.drop(columns=['Value']).iloc[[-1]]
                        test_features = pd.concat([date_features.reset_index(drop=True), lag_rolling_features.reset_index(drop=True)], axis=1)[X_train.columns]
                        pred = lgbm.predict(test_features)[0]
                        test_predictions.append(pred)
                        
                        new_row = history.iloc[[-1]].copy()
                        new_row.index = [test_date]
                        new_row['Value'] = pred
                        full_series = pd.concat([history['Value'], pd.Series([pred], index=[test_date])])
                        for lag in [1, 3, 6, 12]: new_row[f'lag_{lag}'] = full_series.shift(lag).iloc[-1]
                        for window in [3, 6, 12]:
                            new_row[f'rolling_mean_{window}'] = full_series.shift(1).rolling(window=window).mean().iloc[-1]
                            new_row[f'rolling_std_{window}'] = full_series.shift(1).rolling(window=window).std().iloc[-1]
                        history = pd.concat([history, new_row])

                    metrics["LightGBM"].append({"mae": mean_absolute_error(test, test_predictions), "rmse": np.sqrt(mean_squared_error(test, test_predictions)), "mape": mean_absolute_percentage_error(test, test_predictions), "smape": smape(test, test_predictions)})
                except Exception:
                    continue

        # Average the metrics
        avg_metrics = {}
        for model, model_metrics in metrics.items():
            if model_metrics: # Ensure there are metrics to average
                avg_metrics[f"{model.lower()}_mae"] = np.mean([m["mae"] for m in model_metrics])
                avg_metrics[f"{model.lower()}_rmse"] = np.mean([m["rmse"] for m in model_metrics])
                avg_metrics[f"{model.lower()}_mape"] = np.mean([m["mape"] for m in model_metrics])
                avg_metrics[f"{model.lower()}_smape"] = np.mean([m["smape"] for m in model_metrics])
        
        metrics = avg_metrics

    # --- 2. Retrain Models on Full Data and Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name=f"Historical {index_type}", line=dict(color="blue")))
    
    final_forecast_df = pd.DataFrame() # To store the forecast of the best model for AI summary

    with st.spinner("Generating final forecasts..."):
        if "ARIMA" in models:
            _, arima_df = plot_forecasting_arima(ts, index_type, n_periods, scenario_params)
            fig.add_trace(go.Scatter(x=arima_df['ds'], y=arima_df['yhat'], mode="lines", name="ARIMA Forecast", line=dict(color="red", width=2.5)))
            if final_forecast_df.empty: final_forecast_df = arima_df

        if "Prophet" in models:
            _, prophet_df = plot_forecasting_prophet(ts, index_type, n_periods, scenario_params)
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['yhat'], mode="lines", name="Prophet Forecast", line=dict(color="green", width=2.5)))
            if final_forecast_df.empty: final_forecast_df = prophet_df

        if "LightGBM" in models:
            _, lightgbm_df = plot_forecasting_lightgbm(ts, index_type, n_periods, scenario_params)
            fig.add_trace(go.Scatter(x=lightgbm_df['ds'], y=lightgbm_df['yhat'], mode="lines", name="LightGBM Forecast", line=dict(color="purple", width=2.5)))
            if final_forecast_df.empty: final_forecast_df = lightgbm_df

    # --- 3. Create Plot ---
    fig.update_layout(
        title=f"{index_type} Models Forecast ({n_periods}-Month, {scenario_str} Scenario)",
        xaxis_title="Year", yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white", font=dict(size=14), height=600,
    )

    return fig, metrics, final_forecast_df

import google.generativeai as genai
import os

def generate_ai_summary(
    county_name: str,
    index_type: str,
    models_to_run: list,
    metrics: dict,
    forecast_horizon: int,
    time_series: pd.Series,
    forecast_df: pd.DataFrame
) -> str:
    """
    Generates a narrative summary of the forecast results using a GenAI model.
    """
    # --- API Key Configuration ---
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return (
            "**Warning:** `GOOGLE_API_KEY` environment variable not set. "
            "AI summary generation is disabled. Please set the key to enable this feature."
        )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        return f"**Error:** Could not configure the Generative AI model. Please check your API key. Details: {e}"

    # --- Prompt Engineering ---
    # Create a concise summary of the input data for the prompt
    latest_value = time_series.iloc[-1]
    forecast_start = forecast_df.iloc[0]['yhat']
    forecast_end = forecast_df.iloc[-1]['yhat']
    
    # Determine the best model
    best_model = "N/A"
    best_smape = float('inf')
    for model_name in models_to_run:
        smape_key = f"{model_name.lower()}_smape"
        if smape_key in metrics and metrics[smape_key] < best_smape:
            best_smape = metrics[smape_key]
            best_model = model_name

    prompt = f"""
    You are an expert climatologist analyzing drought conditions.
    Your task is to provide a concise, easy-to-understand, and data-driven summary of a new forecast.

    **Analysis Context:**
    - **Location:** {county_name}
    - **Climate Index:** {index_type} (A lower value means drier conditions)
    - **Models Used:** {', '.join(models_to_run)}
    - **Best Performing Model (by sMAPE):** {best_model} (sMAPE: {best_smape:.2f}%)
    - **Forecast Horizon:** {forecast_horizon} months

    **Key Data Points:**
    - **Most Recent Actual Value:** {latest_value:.2f}
    - **Forecasted Value (Start of Horizon):** {forecast_start:.2f}
    - **Forecasted Value (End of Horizon):** {forecast_end:.2f}

    **Your Task:**
    Based *only* on the data provided, write a 2-3 sentence narrative summary.
    1.  Start by stating the overall trend predicted by the forecast (e.g., "conditions are expected to become drier," "a slight improvement is forecasted," "conditions are expected to remain stable").
    2.  Mention the confidence in the forecast, referencing the best-performing model.
    3.  Conclude with a brief statement about what this trend implies for the county's drought situation.
    
    Do not include any information not present in the data provided. Be objective and professional.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**Error:** Failed to generate AI summary. Details: {e}"