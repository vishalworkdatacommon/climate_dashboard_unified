# U.S. County-Level Climate Dashboard

![Update Climate Data](https://github.com/vishalworkdatacommon/climate_dashboard_unified/actions/workflows/update_climate_data.yml/badge.svg)

An interactive web application for analyzing, visualizing, and forecasting key drought indices for any county in the United States. This dashboard provides a powerful tool for researchers, policymakers, and climate enthusiasts to explore historical trends and predict future conditions.

<!-- TODO: Add a high-quality screenshot or GIF of the dashboard in action -->
![Dashboard Screenshot](placeholder.png)

---

## Features

*   **Interactive Map Selector:** Explore a choropleth map of the U.S. to visualize regional drought patterns and select counties by clicking on them.
*   **Live Data for Analysis:** While the map uses a fast, pre-built dataset, all detailed analyses are performed on up-to-the-second data fetched live from the NOAA API.
*   **Multiple Analysis Modes:**
    *   **Trend Analysis:** View monthly index values with a 12-month rolling average.
    *   **Anomaly Detection:** Automatically identify periods of extreme drought or wetness.
    *   **Seasonal Decomposition:** Break down the time series into trend, seasonal, and residual components.
    *   **Autocorrelation:** Analyze the ACF and PACF plots to understand the data's underlying structure.
*   **Dedicated Comparison Mode:** Select two or more counties to see a detailed comparison, including:
    *   A summary statistics table (mean, median, max, min).
    *   Trend sparklines for quick visual comparison.
    *   A correlation matrix heatmap to see how similarly the counties behave.
*   **Advanced Forecasting:**
    *   Compare the performance of **ARIMA** and **Prophet** models.
    *   Adjust the forecast horizon from 6 to 48 months.
    *   Run **Scenario Analysis** to see how "Wetter" or "Drier" conditions might impact the forecast.
*   **Automated Historical Insights:** For any single county, get instant context on the latest data point, including its historical percentile rank and comparison to the monthly average.
*   **Performant Caching:** A smart file-based caching system makes the app highly responsive for frequently accessed counties.
*   **Downloadable Data:** Easily download the selected data as a CSV file for offline analysis.

## Data Sources

Data is sourced live from **NOAA's National Centers for Environmental Information (NCEI)**. The dashboard utilizes the following indices:

*   **PDSI (Palmer Drought Severity Index):** Measures long-term drought based on temperature and precipitation.
*   **SPI (Standardized Precipitation Index):** Compares precipitation to the long-term average.
*   **SPEI (Standardized Precipitation-Evapotranspiration Index):** Similar to SPI, but also accounts for temperature's effect on water demand.

## Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [GeoPandas](https://geopandas.org/)
*   **Data Visualization:** [Plotly](https://plotly.com/), [Folium](https://python-visualization.github.io/folium/)
*   **Forecasting Models:** [pmdarima (auto-ARIMA)](http://alkaline-ml.com/pmdarima/), [Prophet](https://facebook.github.io/prophet/)
*   **API Interaction:** [Requests](https://requests.readthedocs.io/en/latest/)

## 🚀 Running Locally

To run the dashboard on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vishalworkdatacommon/climate_dashboard_unified.git
    cd climate_dashboard_unified
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the map data file:**
    This is a crucial one-time step to create the pre-built data file needed for the interactive map.
    ```bash
    python3 build_data.py
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Project Structure

```
.
├── app.py                   # Main Streamlit application logic
├── data_loader.py           # Handles API calls, caching, and data loading
├── plotting.py              # Contains functions for generating charts and insights
├── ml_models.py             # Contains the forecasting models and logic
├── map_view.py              # Logic for the interactive Folium map
├── build_data.py            # Script to generate the pre-built map data
├── requirements.txt         # Project dependencies
├── .streamlit/config.toml   # Custom theme configuration
└── cache/                   # Directory for cached data files (gitignored)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a pull request.

## License

This project is licensed under the MIT License.