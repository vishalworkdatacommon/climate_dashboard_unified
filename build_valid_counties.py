# build_valid_counties.py
import pandas as pd
import requests
from config import DATA_URLS, RAW_FIPS_PATH
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_counties_for_index(index_type: str) -> set:
    """Fetches all unique county FIPS codes for a single index type."""
    print(f"Fetching county list for {index_type}...")
    base_url = DATA_URLS[index_type]
    fips_col = {"SPEI": "fips", "SPI": "countyfips", "PDSI": "countyfips"}[index_type]
    
    # A SoQL query to get distinct FIPS codes
    query = f"?$select=DISTINCT {fips_col}"
    
    try:
        response = requests.get(base_url + query, timeout=180)
        if response.status_code == 200:
            # Use the json library to parse the response
            # Use the json library to parse the response
            data = response.json()
            df = pd.DataFrame(data)
            # FIPS codes are in the column specified by fips_col
            df.rename(columns={fips_col: "countyfips"}, inplace=True)
            df["countyfips"] = df["countyfips"].astype(str).str.zfill(5)
            print(f"  - Found {len(df)} unique counties for {index_type}.")
            return set(df["countyfips"])
    except requests.RequestException as e:
        print(f"  - ERROR: Could not fetch county list for {index_type}: {e}")
        return set()
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
