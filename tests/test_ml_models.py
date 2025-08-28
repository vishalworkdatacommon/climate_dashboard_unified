import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytest
from ml_models import plot_forecasting_arima, smape, _apply_scenario

def test_smape_calculation():
    """Tests the smape function for accuracy."""
    y_true = pd.Series([100, 110, 120, 130])
    y_pred = pd.Series([105, 115, 125, 135])
    # Expected sMAPE: mean(2 * abs(5) / (abs(100)+abs(105)), ...) * 100
    expected_smape = np.mean([
        2 * 5 / (100 + 105),
        2 * 5 / (110 + 115),
        2 * 5 / (120 + 125),
        2 * 5 / (130 + 135)
    ]) * 100
    assert np.isclose(smape(y_true, y_pred), expected_smape)

def test_smape_with_zeros():
    """Tests the smape function with zero values."""
    y_true = pd.Series([0, 10])
    y_pred = pd.Series([0, 10])
    assert smape(y_true, y_pred) == 0.0

@pytest.fixture
def sample_time_series():
    """Creates a sample time series for testing scenarios."""
    return pd.Series(np.linspace(100, 110, 12), index=pd.date_range("2023-01-01", periods=12, freq="MS"))

def test_apply_scenario_no_change(sample_time_series):
    """Tests that the scenario function makes no change if magnitude is zero."""
    scenario_params = {"magnitude": 0}
    ts_adjusted = _apply_scenario(sample_time_series, scenario_params)
    pd.testing.assert_series_equal(sample_time_series, ts_adjusted)

def test_apply_scenario_sudden_shock(sample_time_series):
    """Tests the 'Sudden Shock' scenario."""
    scenario_params = {"type": "Sudden Shock", "magnitude": 0.10} # +10% shock
    ts_adjusted = _apply_scenario(sample_time_series, scenario_params)
    assert np.isclose(ts_adjusted.iloc[-1], sample_time_series.iloc[-1] * 1.10)
    # Check that other values are unchanged
    pd.testing.assert_series_equal(sample_time_series.iloc[:-1], ts_adjusted.iloc[:-1])

def test_apply_scenario_gradual_trend(sample_time_series):
    """Tests the 'Gradual Trend' scenario."""
    duration = 4
    magnitude = 0.20 # +20% trend
    scenario_params = {"type": "Gradual Trend", "magnitude": magnitude, "duration": duration}
    
    ts_adjusted = _apply_scenario(sample_time_series, scenario_params)
    
    last_original_value = sample_time_series.iloc[-1]
    total_change = last_original_value * magnitude
    expected_gradual_change = np.linspace(0, total_change, duration)
    
    # The last value should have the full change applied
    assert np.isclose(ts_adjusted.iloc[-1], sample_time_series.iloc[-1] + total_change)
    # The first value of the trend should have a small change
    assert np.isclose(ts_adjusted.iloc[-duration], sample_time_series.iloc[-duration] + expected_gradual_change[0])
    # Check that values before the trend are unchanged
    pd.testing.assert_series_equal(sample_time_series.iloc[:-duration], ts_adjusted.iloc[:-duration])

def test_plot_forecasting_arima_returns_figure_and_dataframe():
    """
    Tests if the plot_forecasting_arima function returns a Plotly Figure and a DataFrame.
    """
    # Create a simple time series for testing
    data = {"Value": [i for i in range(24)]}
    index = pd.to_datetime(
        [f"2022-{i}-01" for i in range(1, 13)] + [f"2023-{i}-01" for i in range(1, 13)]
    )
    ts = pd.Series(data["Value"], index=index)

    # Call the function
    fig, df = plot_forecasting_arima(ts, "Test Index")

    # Assert that the returned objects are of the correct type
    assert isinstance(fig, go.Figure)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'ds' in df.columns
    assert 'yhat' in df.columns