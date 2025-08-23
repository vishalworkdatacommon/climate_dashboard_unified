import streamlit as st
import pandas as pd
import folium
import branca.colormap as cm
from streamlit_folium import st_folium

def create_interactive_map(gdf, data, index_type):
    """
    Creates and displays an interactive Folium map with a choropleth layer.
    Returns the FIPS code of the last clicked county.
    """
    # 1. Prepare the data for the map
    latest_date = data['date'].max()
    map_data = data[
        (data["index_type"] == index_type) & (data["date"] == latest_date)
    ]
    merged_gdf = gdf.merge(map_data, on="countyfips", how="left")

    # Convert Timestamp to string to avoid JSON serialization errors
    if "date" in merged_gdf.columns:
        merged_gdf["date"] = merged_gdf["date"].astype(str)

    # 2. Create the map
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # 3. Create a colormap
    colormap = cm.LinearColormap(
        colors=['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
        index=[-3, -1.5, 0, 1.5, 3],
        vmin=-3, vmax=3,
        caption=f"{index_type} Value"
    )
    m.add_child(colormap)

    # 4. Add the choropleth layer
    choropleth = folium.Choropleth(
        geo_data=merged_gdf,
        data=merged_gdf,
        columns=["countyfips", "Value"],
        key_on="feature.properties.countyfips",
        fill_color="YlOrRd", # This will be overridden by the GeoJson style function
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{index_type} Value",
        highlight=True,
    ).add_to(m)

    # 5. Add tooltips and click functionality
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Value']) if pd.notna(feature['properties']['Value']) else '#808080',
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["display_name", "Value"],
            aliases=["County:", f"{index_type} Value:"],
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["display_name"],
            aliases=[""],
            localize=True,
        )
    ).add_to(m)

    # 6. Display the map and capture output
    map_output = st_folium(m, width='100%', height=500)
    
    last_clicked_fips = None
    if map_output and map_output.get('last_object_clicked_popup'):
        # Extract FIPS code from the popup content (which is the display_name)
        clicked_name = map_output['last_object_clicked_popup']
        fips_series = merged_gdf[merged_gdf['display_name'] == clicked_name]['countyfips']
        if not fips_series.empty:
            last_clicked_fips = fips_series.iloc[0]
            
    return last_clicked_fips