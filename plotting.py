import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf


def plot_trend_analysis(ts: pd.Series, index_type: str) -> go.Figure:
    rolling_avg = ts.rolling(window=12).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts,
            mode="lines",
            name=f"Monthly {index_type}",
            line=dict(color="lightblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_avg.index,
            y=rolling_avg,
            mode="lines",
            name="12-Month Rolling Average",
            line=dict(color="navy", width=2.5),
        )
    )
    fig.update_layout(
        title=f"{index_type} Trend Analysis",
        xaxis_title="Year",
        yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white",
        font=dict(size=14),
        height=600,
    )
    return fig


def plot_anomaly_detection(ts: pd.Series, index_type: str) -> go.Figure:
    rolling_mean = ts.rolling(window=12).mean()
    rolling_std = ts.rolling(window=12).std()
    upper_bound = rolling_mean + (2 * rolling_std)
    lower_bound = rolling_mean - (2 * rolling_std)
    anomalies = ts[(ts > upper_bound) | (ts < lower_bound)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts,
            mode="lines",
            name=f"Monthly {index_type}",
            line=dict(color="dodgerblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean,
            mode="lines",
            name="12-Month Rolling Mean",
            line=dict(color="orange", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalies.index,
            y=anomalies,
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=8, symbol="x"),
        )
    )
    fig.update_layout(
        title=f"{index_type} Anomaly Detection",
        xaxis_title="Year",
        yaxis_title=f"{index_type} Value",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white",
        font=dict(size=14),
        height=600,
    )
    return fig


def plot_seasonal_decomposition(ts: pd.Series, index_type: str) -> go.Figure:
    decomposition = seasonal_decompose(ts.dropna(), model="additive", period=12)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.observed.index,
            y=decomposition.observed,
            mode="lines",
            name="Observed",
            line=dict(color="dodgerblue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend,
            mode="lines",
            name="Trend",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal,
            mode="lines",
            name="Seasonal",
            line=dict(color="green"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.resid.index,
            y=decomposition.resid,
            mode="markers",
            name="Residual",
            marker=dict(color="red", size=4),
        ),
        row=4,
        col=1,
    )
    fig.update_layout(
        title_text=f"{index_type} Time Series Decomposition",
        height=700,
        showlegend=False,
        template="plotly_white",
        font=dict(size=14),
    )
    return fig


def plot_autocorrelation(ts: pd.Series, index_type: str) -> go.Figure:
    nlags = 40
    ts_dropna = ts.dropna()
    acf_values, confint_acf = acf(ts_dropna, nlags=nlags, alpha=0.05)
    pacf_values, confint_pacf = pacf(ts_dropna, nlags=nlags, alpha=0.05)
    ci_acf = confint_acf - acf_values[:, None]
    ci_pacf = confint_pacf - pacf_values[:, None]
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"),
    )
    fig.add_trace(
        go.Bar(x=list(range(1, nlags + 1)), y=acf_values[1:], name="ACF"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, nlags + 1)),
            y=ci_acf[1:, 0],
            mode="lines",
            line=dict(color="blue", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, nlags + 1)),
            y=ci_acf[1:, 1],
            mode="lines",
            line=dict(color="blue", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(0,0,255,0.1)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=list(range(1, nlags + 1)), y=pacf_values[1:], name="PACF"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, nlags + 1)),
            y=ci_pacf[1:, 0],
            mode="lines",
            line=dict(color="blue", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, nlags + 1)),
            y=ci_pacf[1:, 1],
            mode="lines",
            line=dict(color="blue", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(0,0,255,0.1)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title_text=f"Autocorrelation for {index_type}",
        height=600,
        template="plotly_white",
        font=dict(size=14),
    )
    return fig
