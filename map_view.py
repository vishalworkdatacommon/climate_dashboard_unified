import streamlit as st
import pandas as pd
import folium


def get_color(value):
    if pd.isna(value):
        return "#808080"  # Gray for no data
    if value <= -2:
        return "#d7191c"  # Severe Drought
    elif value <= -1.5:
        return "#fdae61"  # Moderate Drought
    elif value <= -1.0:
        return "#ffffbf"  # Mild Drought
    elif value >= 2:
        return "#1a9641"  # Very Wet
    elif value >= 1.5:
        return "#a6d96a"  # Moderately Wet
    else:
        return "#ffffff"  # Normal


def create_map(gdf, data, index_type, date):
    st.subheader(f"U.S. Drought Conditions for {date.strftime('%B %Y')}")

    map_data = data[
        (data["index_type"] == index_type)
        & (data["date"].dt.strftime("%Y-%m") == date.strftime("%Y-%m"))
    ]

    merged_gdf = gdf.merge(map_data, on="countyfips", how="left")

    # Convert Timestamp to string to avoid JSON serialization errors
    if "date" in merged_gdf.columns:
        merged_gdf["date"] = merged_gdf["date"].astype(str)

    merged_gdf["color"] = merged_gdf["Value"].apply(get_color)

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    folium.Choropleth(
        geo_data=merged_gdf,
        name="choropleth",
        data=merged_gdf,
        columns=["countyfips", "Value"],
        key_on="feature.id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{index_type} Value",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Add tooltips
    def style_function(feature):
        return {
            "fillColor": feature["properties"]["color"],
            "color": "black",
            "fillOpacity": 0.7,
            "weight": 0.2,
        }

    folium.GeoJson(
        merged_gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["display_name", "Value"],
            aliases=["County:", f"{index_type} Value:"],
            localize=True,
        ),
    ).add_to(m)

    return m
