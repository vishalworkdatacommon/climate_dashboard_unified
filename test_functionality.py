import pandas as pd
from data_loader import get_live_data_for_counties
from ml_models import plot_forecasting_arima, plot_forecasting_prophet

def run_test():
    print("--- Starting Functionality Test ---")

    # Test data loading
    print("Testing data loading...")
    county_fips = ["01001"]  # Autauga County, Alabama
    try:
        data = get_live_data_for_counties(county_fips)
        print(f"Successfully loaded data for FIPS {county_fips}.")
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter data for a single index for forecasting
    pdsi_data = data[data['index_type'] == 'PDSI'].copy()
    pdsi_data.set_index('date', inplace=True)
    pdsi_data = pdsi_data[['Value']]


    # Test ARIMA model
    print("\nTesting ARIMA model...")
    try:
        arima_fig = plot_forecasting_arima(pdsi_data['Value'], 'PDSI', 12)
        print("Successfully generated ARIMA forecast.")
    except Exception as e:
        print(f"Error with ARIMA model: {e}")
        return

    # Test Prophet model
    print("\nTesting Prophet model...")
    try:
        prophet_fig = plot_forecasting_prophet(pdsi_data['Value'], 'PDSI', 12)
        print("Successfully generated Prophet forecast.")
    except Exception as e:
        print(f"Error with Prophet model: {e}")
        return

    print("\n--- Functionality Test Successful ---")

if __name__ == "__main__":
    run_test()
