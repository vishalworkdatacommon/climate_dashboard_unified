# build_national_view.py
import pandas as pd
import requests
import re
from config import DATA_URLS, FIPS_PATH
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def _fetch_index_data(index_type: str) -> tuple[str, pd.DataFrame | None]:
    """
    Performs a single, bulk download of the entire dataset for one index type.
    Returns the index type along with the dataframe.
    """
    print(f"  - Starting bulk download for {index_type}...")
    base_url = DATA_URLS[index_type]
    query = "?$limit=50000000" 
    
    try:
        response = requests.get(base_url + query, timeout=300) # 5 minute timeout for very large files
        if response.status_code == 200:
            print(f"  - Bulk download for {index_type} complete.")
            # Use regex to parse the JSON-like response
            if index_type == 'SPEI':
                # SPEI has a different format
                data = re.findall(r'\{"state":"(.*?)","county":"(.*?)","fips":"(\d{5})","year":"(\d{4})","month":"(\d{1,2})","spei":"(.*?)"\}', response.text)
                df = pd.DataFrame(data, columns=["state", "county", "fips", "year", "month", "spei"])
            else:
                # SPI and PDSI have the same format
                value_col = {"SPI": "spi", "PDSI": "pdsi"}[index_type]
                data = re.findall(r'\{"year":"(\d{4})","month":"(\d{1,2})","statefips":"(\d{1,2})","countyfips":"(\d{5})","'+value_col+'":"(.*?)"\}', response.text)
                df = pd.DataFrame(data, columns=["year", "month", "statefips", "countyfips", value_col])

            print(f"Columns for {index_type}: {df.columns.tolist()}")
            return index_type, df
    except requests.RequestException as e:
        print(f"  - ERROR: Bulk download for {index_type} failed: {e}")
        return index_type, None
    return index_type, None

def main():
    """
    Main function to build the national latest data file.
    This script is designed to be run by an automated process like a GitHub Action.
    """
    print("--- Starting National Data Build Process (Bulk Download Strategy) ---")
    
    all_indices_data = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {executor.submit(_fetch_index_data, it): it for it in DATA_URLS.keys()}
        
        for future in as_completed(future_to_index):
            index_type, full_index_df = future.result()

            if full_index_df is None:
                continue

            print(f"  - Processing latest data for {index_type}...")
            fips_col = {"SPEI": "fips", "SPI": "countyfips", "PDSI": "countyfips"}[index_type]
            
            full_index_df['date'] = pd.to_datetime(full_index_df['year'].astype(str) + '-' + full_index_df['month'].astype(str).str.zfill(2))
            latest_indices = full_index_df.loc[full_index_df.groupby(fips_col)['date'].idxmax()].index
            latest_df = full_index_df.loc[latest_indices]

            value_col = {"SPEI": "spei", "SPI": "spi", "PDSI": "pdsi"}[index_type]
            if "fips" in latest_df.columns: latest_df.rename(columns={"fips": "countyfips"}, inplace=True)
            latest_df.rename(columns={value_col: "Value"}, inplace=True)
            latest_df["countyfips"] = latest_df["countyfips"].astype(str).str.zfill(5)
            latest_df["index_type"] = index_type
            all_indices_data.append(latest_df[["countyfips", "Value", "index_type"]])

    if not all_indices_data:
        print("--- FATAL: Build failed. No data could be downloaded. Aborting. ---")
        exit(1)

    final_df = pd.concat(all_indices_data, ignore_index=True)
    
    # --- Final, Definitive Type Casting ---
    # Force the 'Value' column to be float64 to ensure schema consistency.
    final_df['Value'] = pd.to_numeric(final_df['Value'], errors='coerce')
    final_df.dropna(subset=['Value'], inplace=True)
    final_df = final_df.astype({'Value': 'float64'})

    final_pivot = final_df.pivot(index='countyfips', columns='index_type', values='Value')
    final_pivot.reset_index(inplace=True)
    
    output_path = "national_latest.parquet"
    final_pivot.to_parquet(output_path)
    
    print(f"--- National Data Build Complete ---")
    print(f"Data saved to {output_path}")
    print(f"Total counties processed: {len(final_pivot)}")

if __name__ == "__main__":
    main()
