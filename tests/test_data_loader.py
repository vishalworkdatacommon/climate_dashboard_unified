# -*- coding: utf-8 -*-
import os
import pandas as pd
import pytest
import hashlib
from unittest.mock import patch, MagicMock
from data_loader import get_live_data_for_counties, CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)

def mock_api_response_func(url, params=None):
    mock_response = MagicMock()
    mock_response.status_code = 200
    index_type = "PDSI"
    if "spi" in url: index_type = "SPI"
    elif "spei" in url: index_type = "SPEI"
    value_col_map = {"PDSI": "pdsi", "SPI": "spi", "SPEI": "spei"}
    fips_col_map = {"PDSI": "countyfips", "SPI": "countyfips", "SPEI": "fips"}
    data = {
        "year": ["2023"], "month": ["1"],
        value_col_map[index_type]: [1.0],
        fips_col_map[index_type]: ["01001"]
    }
    csv_string = pd.DataFrame(data).to_csv(index=False)
    mock_response.text = csv_string
    mock_response.raise_for_status = MagicMock()
    return mock_response

def test_get_live_data_for_counties_api_call():
    with patch('requests.get', side_effect=mock_api_response_func) as mock_get, \
         patch('data_loader.get_county_options') as mock_get_county_options, \
         patch('schemas.climate_data_schema.validate', side_effect=lambda x: x):
        
        mock_get_county_options.return_value = pd.Series({"01001": "Autauga, AL"})
        
        fips_codes = ["01001"]
        cache_key = hashlib.md5("".join(sorted(fips_codes)).encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
        if os.path.exists(cache_file):
            os.remove(cache_file)

        df = get_live_data_for_counties(fips_codes)

        assert mock_get.called
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df['display_name'].iloc[0] == "Autauga, AL"
        assert os.path.exists(cache_file)
        
        if os.path.exists(cache_file):
            os.remove(cache_file)

def test_get_live_data_for_counties_cache_hit():
    with patch('requests.get') as mock_get:
        fips_codes = ["99999"]
        cache_key = hashlib.md5("".join(sorted(fips_codes)).encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
        
        sample_data = {
            'date': pd.to_datetime(['2024-01-01']), 'Value': [1.0],
            'index_type': ['PDSI'], 'countyfips': [fips_codes[0]]
        }
        pd.DataFrame(sample_data).to_parquet(cache_file)

        df = get_live_data_for_counties(fips_codes)

        assert not mock_get.called
        assert df['Value'].iloc[0] == 1.0

        if os.path.exists(cache_file):
            os.remove(cache_file)
