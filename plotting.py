import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import percentileofscore


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


def plot_comparison_mode(full_data: pd.DataFrame, fips_codes: list, index_choice: str):
    """
    Displays a comparison view for multiple counties, including a summary table
    and a correlation matrix.
    """
    st.subheader("Summary Statistics")

    # 1. Prepare data for comparison
    comparison_df = full_data[
        (full_data["countyfips"].isin(fips_codes)) &
        (full_data["index_type"] == index_choice)
    ]

    # Pivot the table to have counties as columns
    pivot_df = comparison_df.pivot(
        index='date',
        columns='display_name',
        values='Value'
    )

    # 2. Calculate Summary Statistics
    summary_stats = pivot_df.describe().loc[['mean', '50%', 'max', 'min']].T
    summary_stats.rename(columns={'50%': 'median'}, inplace=True)
    summary_stats = summary_stats.round(2)

    # Add sparkline data
    summary_stats['Trend'] = [pivot_df[col].dropna().tolist() for col in pivot_df.columns]

    st.dataframe(
        summary_stats,
        use_container_width=True,
        column_config={
            "Trend": st.column_config.LineChartColumn(
                "Historical Trend",
                width="medium",
            ),
        }
    )

    # 3. Calculate and display Correlation Matrix
    st.subheader("Correlation Matrix")
    st.info("This heatmap shows how similarly the drought indices for the selected counties change over time. A value of 1 means they are perfectly correlated, while a value of 0 means there is no correlation.")

    if len(fips_codes) > 1:
        correlation_matrix = pivot_df.corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(title=f"Correlation of {index_choice} Between Counties")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("At least two counties must be selected to display a correlation matrix.")


def display_historical_insights(ts: pd.Series):
    """
    Calculates and displays key historical insights for a given time series.
    """
    st.subheader("Historical Insights")
    
    if ts.empty:
        st.warning("Not enough data to generate historical insights.")
        return

    latest_value = ts.iloc[-1]
    latest_date = ts.index[-1]
    
    # 1. Percentile Rank
    percentile = percentileofscore(ts.dropna(), latest_value)
    
    # 2. Comparison to Monthly Average
    monthly_avg = ts[ts.index.month == latest_date.month].mean()
    diff_from_avg = latest_value - monthly_avg
    
    # 3. Longest Drought Period
    drought_threshold = -1.0
    drought_periods = (ts < drought_threshold).astype(int).groupby(ts.ge(drought_threshold).astype(int).cumsum()).cumsum()
    longest_drought = drought_periods.max()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Latest Value ({latest_date.strftime('%b %Y')})",
            value=f"{latest_value:.2f}"
        )
    with col2:
        st.metric(
            label="Historical Percentile",
            value=f"{percentile:.1f}%",
            help="The percentage of historical values that are less than or equal to the latest value."
        )
    with col3:
        st.metric(
            label=f"vs. Avg for {latest_date.strftime('%B')}",
            value=f"{diff_from_avg:+.2f}",
            help=f"The difference between the latest value and the historical average for all {latest_date.strftime('%B')}s."
        )
    
    st.metric(
        label="Longest Drought Period",
        value=f"{longest_drought} months",
        help=f"The longest consecutive period with an index value below {drought_threshold}."
    )


def plot_national_map(latest_data: pd.DataFrame, gdf: pd.DataFrame, index_type: str, fips_options: pd.Series) -> go.Figure:
    """
    Generates a national choropleth map of the latest drought index values.
    """
    # Convert fips_options Series to a DataFrame for merging
    county_names_df = fips_options.to_frame(name="display_name").reset_index()

    merged_gdf = gdf.merge(latest_data, on="countyfips", how="left")
    merged_gdf = merged_gdf.merge(county_names_df, on="countyfips", how="left")

    fig = px.choropleth_mapbox(
        merged_gdf,
        geojson=merged_gdf.geometry,
        locations=merged_gdf.index,
        color="Value",
        hover_name="display_name",
        hover_data={"Value": ":.2f", "countyfips": True},
        color_continuous_scale="RdYlBu",
        range_color=[-4, 4],
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.6,
        labels={"Value": f"Latest {index_type} Value"},
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=f"National Overview of Latest {index_type} Data",
    )
    return fig
