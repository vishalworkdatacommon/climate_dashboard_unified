import pandas as pd
import plotly.graph_objects as go
import pytest
from ml_models import plot_forecasting_arima

def test_plot_forecasting_arima_returns_figure():
    """
    Tests if the plot_forecasting_arima function returns a Plotly Figure object.
    """
    # Create a simple time series for testing
    data = {'Value': [i for i in range(24)]}
    index = pd.to_datetime([f'2022-{i}-01' for i in range(1, 13)] + [f'2023-{i}-01' for i in range(1, 13)])
    ts = pd.Series(data['Value'], index=index)
    
    # Call the function
    fig = plot_forecasting_arima(ts, 'Test Index')
    
    # Assert that the returned object is a Figure
    assert isinstance(fig, go.Figure)
