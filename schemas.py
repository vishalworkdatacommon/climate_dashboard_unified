# schemas.py
"""Data validation schemas for the climate dashboard application."""

import pandera as pa
from pandera.typing import Series

class ClimateDataSchema(pa.SchemaModel):
    """Schema for the main climate data DataFrame."""
    date: Series[pa.DateTime] = pa.Field(nullable=False)
    countyfips: Series[str] = pa.Field(nullable=False)
    Value: Series[float] = pa.Field(nullable=True)
    index_type: Series[str] = pa.Field(isin=["SPEI", "SPI", "PDSI"])
    display_name: Series[str] = pa.Field(nullable=False)

class FipsDataSchema(pa.SchemaModel):
    """Schema for the FIPS county data DataFrame."""
    countyfips: Series[str] = pa.Field(nullable=False)
    county_name: Series[str] = pa.Field(nullable=False)
    state: Series[str] = pa.Field(nullable=False)
