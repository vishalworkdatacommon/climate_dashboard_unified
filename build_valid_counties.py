# build_valid_counties.py
import pandas as pd
import requests
import time
from config import DATA_URLS, RAW_FIPS_PATH
from concurrent.futures import ThreadPoolExecutor, as_completed

import io

def get_counties_for_index(index_type: str) -> set:
    """Fetches all unique county FIPS codes for a single index type with retries."""
    print(f"Fetching county list for {index_type}...")
    base_url = DATA_URLS[index_type]
    fips_col = {"SPEI": "fips", "SPI": "countyfips", "PDSI": "countyfips"}[index_type]
    
    # A SoQL query to get distinct FIPS codes
    query = f"?$select=DISTINCT {fips_col}"
    
    for attempt in range(3): # Try up to 3 times
        try:
            response = requests.get(base_url + query, timeout=180)
            if response.status_code == 200:
                # The response is a CSV, so we read it into a pandas DataFrame
                csv_data = io.StringIO(response.text)
                df = pd.read_csv(csv_data)
                # FIPS codes are in the column specified by fips_col
                df.rename(columns={fips_col: "countyfips"}, inplace=True)
                df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)
                print(f"  - Found {len(df)} unique counties for {index_type}.")
                return set(df["countyfips"])
        except (requests.RequestException, pd.errors.ParserError) as e:
            print(f"  - ATTEMPT {attempt + 1} FAILED for {index_type}: {e}")
            if attempt < 2: # Don't sleep on the last attempt
                print("  - Retrying in 30 seconds...")
                time.sleep(30)

    print(f"  - ERROR: Could not fetch county list for {index_type} after 3 attempts.")
    return set()

def main():
    """
    Main function to find the intersection of counties available for all indices
    and create a new, validated FIPS data file.
    """
    print("--- Starting Valid County Build Process ---")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(get_counties_for_index, it) for it in DATA_URLS.keys()]
        results = [future.result() for future in as_completed(futures)]

    if not all(results):
        print("--- FATAL: Could not fetch county list for at least one index. Aborting. ---")
        exit(1)

    # Find the intersection of all sets
    common_fips = set.intersection(*results)
    
    if not common_fips:
        print("--- FATAL: No common counties found across all indices. Aborting. ---")
        exit(1)
        
    print(f"\nFound {len(common_fips)} counties that have data for ALL three indices.")

    # Load the original FIPS data to get county names and states
    full_fips_df = pd.read_csv(RAW_FIPS_PATH, dtype=str)
    full_fips_df["countyfips"] = full_fips_df["state_fips"].str.zfill(2) + full_fips_df["county_fips"].str.zfill(3)

    # Filter the full list down to only the common, valid FIPS codes
    valid_fips_df = full_fips_df[full_fips_df["countyfips"].isin(common_fips)]
    
    output_path = "valid_counties.csv"
    valid_fips_df.to_csv(output_path, index=False)
    
    print("--- Valid County Build Complete ---")
    print(f"Clean county list saved to {output_path}")

if __name__ == "__main__":
    main()
