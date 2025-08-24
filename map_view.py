
import streamlit as st
import pandas as pd
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
import branca.colormap as cm
from streamlit_folium import st_folium
from geopandas import GeoDataFrame
from typing import Optional, Dict, Any

# --- Constants ---
DEFAULT_MAP_LOCATION = [39.8283, -98.5795]
DEFAULT_ZOOM = 4
NO_DATA_COLOR = "#808080"
COLOR_SCALE = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
COLOR_INDEX = [-3, -1.5, 0, 1.5, 3]

# --- Helper Functions ---

@st.cache_data
def get_colormap(index_type: str, vmin: float = -3, vmax: float = 3) -> cm.LinearColormap:
    """Creates and returns a branca colormap for the map legend."""
    return cm.LinearColormap(
        colors=COLOR_SCALE,
        index=COLOR_INDEX,
        vmin=vmin,
        vmax=vmax,
        caption=f"Value for {index_type}"
    )

def style_function(feature: Dict[str, Any], colormap: cm.LinearColormap) -> Dict[str, Any]:
    """
    Applies styling to a GeoJSON feature based on its 'Value' property.
    Handles missing data by applying a default color.
    """
    value = feature['properties'].get('Value')
    return {
        'fillColor': colormap(value) if pd.notna(value) else NO_DATA_COLOR,
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7
    }

# --- Main Map Creation Function ---

def create_interactive_map(
    gdf: GeoDataFrame, 
    data: pd.DataFrame, 
    index_type: str
) -> Optional[str]:
    """
    Creates and displays an interactive Folium map with a choropleth layer.

    Args:
        gdf: A GeoDataFrame containing county geometries and FIPS codes.
        data: A DataFrame with the time-series data for the selected index.
        index_type: The climate index being displayed (e.g., "PDSI").

    Returns:
        The FIPS code of the last clicked county, or None if no county was clicked.
    """
    # 1. Prepare data for the map
    if data.empty:
        st.warning("No data available to display on the map.")
        return None
        
    latest_date = data['date'].max()
    map_data = data[
        (data["index_type"] == index_type) & (data["date"] == latest_date)
    ]
    
    # Ensure 'date' column is in a JSON-serializable format
    if "date" in map_data.columns:
        map_data["date"] = map_data["date"].astype(str)

    merged_gdf = gdf.merge(map_data, on="countyfips", how="left")

    # 2. Create the base map
    m = folium.Map(location=DEFAULT_MAP_LOCATION, zoom_start=DEFAULT_ZOOM)
    
    # 3. Create and add the colormap legend
    colormap = get_colormap(index_type)
    m.add_child(colormap)

    # 4. Create and add the GeoJson layer with tooltips and popups
    tooltip = GeoJsonTooltip(
        fields=["display_name", "Value", "date"],
        aliases=["County:", f"{index_type} Value:", "Date:"],
        localize=True,
        sticky=False,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """
    )
    
    popup = GeoJsonPopup(
        fields=["display_name"],
        aliases=[""],
        localize=True,
    )

    geojson_layer = GeoJson(
        merged_gdf,
        style_function=lambda feature: style_function(feature, colormap),
        tooltip=tooltip,
        popup=popup,
        name="counties"
    )
    geojson_layer.add_to(m)

    # 5. Display the map and capture user interaction
    map_output = st_folium(m, width='100%', height=500, returned_objects=[])
    
    last_clicked_fips = None
    if map_output and map_output.get("last_object_clicked_popup"):
        clicked_name = map_output["last_object_clicked_popup"]
        fips_series = merged_gdf[merged_gdf['display_name'] == clicked_name]['countyfips']
        if not fips_series.empty:
            last_clicked_fips = fips_series.iloc[0]
            
    return last_clicked_fips
