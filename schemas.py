# schemas.py
"""Data validation schemas for the climate dashboard application."""

import pandera as pa

# Using the older DataFrameSchema syntax for broader compatibility
climate_data_schema = pa.DataFrameSchema({
    "date": pa.Column(pa.DateTime, nullable=False),
    "countyfips": pa.Column(str, nullable=False),
    "Value": pa.Column(float, nullable=True),
    "index_type": pa.Column(str, pa.Check.isin(["SPEI", "SPI", "PDSI"])),
    "display_name": pa.Column(str, nullable=False),
})

fips_data_schema = pa.DataFrameSchema({
    "countyfips": pa.Column(str, nullable=False),
    "county_name": pa.Column(str, nullable=False),
    "state": pa.Column(str, nullable=False),
})