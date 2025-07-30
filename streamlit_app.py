# File: app.py
# The "Face" of the EWARS-ID system - Final Production Version
# Data Loading Update: Now fetches data from a private GitHub repository.
# FINAL FIX: Implements a robust standardization logic to ensure CSV data loads correctly.

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
import io
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="EWARS-ID Dengue Forecast by M Arief Widagdo",
    page_icon="🇮🇩",
    layout="wide"
)

# --- NEW: Robust Standardization Function ---
def standardize_name(name):
    """
    Creates a reliable, simple key for merging by standardizing names from both
    the forecast CSV and the GeoJSON file.
    - Converts to lowercase
    - Removes all prefixes (kabupaten, kota, etc.)
    - Removes all spaces and non-alphanumeric characters.
    
    Examples:
    'Kota Bandung' -> 'bandung'
    'KOTA BEKASI'  -> 'bekasi'
    'Kabupaten Aceh Barat' -> 'acehbarat'
    'AcehBarat' -> 'acehbarat'
    """
    if not isinstance(name, str):
        return None
    
    name_lower = name.lower().strip()
    
    # List of prefixes to remove. Ordered to handle variations.
    prefixes = ['kabupaten administrasi', 'kota administrasi', 'kabupaten', 'kota', 'kab.']
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            # Replace only the first occurrence of the prefix
            name_lower = name_lower.replace(prefix, '', 1).strip()
            
    # Remove all remaining spaces and non-alphanumeric characters for a clean key
    return re.sub(r'[^a-z0-9]', '', name_lower)

# --- Caching Functions ---
@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """
    Loads and prepares all data from a private GitHub repo using a robust
    standardization method to ensure data is merged correctly.
    """
    try:
        # --- Securely fetch files from GitHub ---
        github_token = st.secrets["GITHUB_TOKEN"]
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3.raw",
        }
        
        forecast_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{forecast_path}"
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        
        geojson_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{geojson_path}"
        geojson_response = requests.get(geojson_url, headers=headers)
        geojson_response.raise_for_status()

        # --- Read the downloaded content ---
        forecast_df = pd.read_csv(
            io.StringIO(forecast_response.text),
            parse_dates=['Date'],
            date_parser=pd.to_datetime
        )
        forecast_df.rename(columns={
            'Kabupaten_Standard': 'kabupaten_standard',
            'Predicted_Cases': 'predicted_cases',
            'Population': 'population'
        }, inplace=True)
        
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))
        
        # --- FIX: Create the reliable merge key on BOTH dataframes ---
        forecast_df['merge_key'] = forecast_df['kabupaten_standard'].apply(standardize_name)
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'].apply(standardize_name)
        
        # --- Merge the full forecast dataset with the GeoDataFrame ---
        # This keeps all map shapes and attaches forecast data where the keys match.
        merged_gdf = kabupaten_gdf.merge(forecast_df, on='merge_key', how='left')

        # --- Add Debugging to show match success ---
        st.sidebar.header("Data Loading Status")
        matched_count = merged_gdf['kabupaten_standard'].nunique()
        total_count = forecast_df['kabupaten_standard'].nunique()
        st.sidebar.success(f"Successfully matched {matched_count} of {total_count} forecast regions to the map.")

        unmatched_keys = set(forecast_df['merge_key']) - set(kabupaten_gdf['merge_key'])
        unmatched_names = forecast_df[forecast_df['merge_key'].isin(unmatched_keys)]['kabupaten_standard'].unique()
        if len(unmatched_names) > 0:
            with st.sidebar.expander("View Unmatched Regions"):
                st.write(unmatched_names)
        
        # Return the fully merged data and the original forecast dataframe
        return merged_gdf, forecast_df
        
    except Exception as e:
        st.error(f"FATAL ERROR: An error occurred during data loading or processing: {e}")
        return None, None

def create_map(gdf):
    """Creates the Folium map with choropleth layer and popups."""
    # Ensure data is numeric for calculation
    gdf['predicted_cases_numeric'] = pd.to_numeric(gdf['predicted_cases'], errors='coerce').fillna(0)
    gdf['population_numeric'] = pd.to_numeric(gdf['population'], errors='coerce').fillna(0)
    
    gdf['incidence_rate'] = gdf.apply(
        lambda row: (row['predicted_cases_numeric'] / row['population_numeric']) * 100000 if row['population_numeric'] > 0 else 0,
        axis=1
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=['merge_key', 'predicted_cases_numeric'],
        key_on='feature.properties.merge_key',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name='Predicted Dengue Cases (Next Week)',
        name='Predicted Cases',
        nan_fill_color='white' # Color for regions with no data
    ).add_to(m)

    gdf['popup_html'] = gdf.apply(
        lambda row: f"""<div style="font-family: sans-serif;">
            <h4>📍 {row.get('kabupaten_standard', row['NAME_2'])}</h4>
            <p><b>Forecast Week:</b> {row.get('Date', pd.NaT).strftime('%Y-%m-%d') if pd.notna(row.get('Date')) else 'N/A'}</p>
            <p><b>Predicted Cases:</b> {int(row['predicted_cases_numeric'])}</p>
            <p><b>Population:</b> {int(row['population_numeric'])}</p>
            <p><b>Incidence Rate (/100k):</b> {row['incidence_rate']:.2f}</p>
        </div>""", axis=1)

    folium.GeoJson(
        gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
        tooltip=folium.features.GeoJsonTooltip(fields=['NAME_2'], aliases=['Region:']),
        popup=folium.features.GeoJsonPopup(fields=['popup_html'], aliases=[''])
    ).add_to(m)
    
    return m

# --- Main App Layout ---
st.title("🇮🇩 EWARS-ID: Dengue Forecast Dashboard")
st.markdown("An operational prototype for near real-time dengue fever forecasting by M Arief Widagdo.")

OWNER = "AriefWidagdo"
PRIVATE_REPO_NAME = "data-raw"
FORECAST_FILE_PATH = "january_2024_predictions.csv"
GEOJSON_PATH = "gadm41_IDN_2.json"

# Load data once
merged_data, forecast_data = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)

if merged_data is not None and forecast_data is not None:
    
    unique_dates = sorted(forecast_data['Date'].unique())
    
    if unique_dates:
        selected_date = st.selectbox(
            "Select Forecast Date for Map View:",
            unique_dates,
            format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
        )
        
        # Filter the already-merged data for the selected date
        map_ready_gdf = merged_data[
            (merged_data['Date'] == selected_date) | (merged_data['Date'].isnull())
        ].copy()
        
        # For shapes that had no match, ensure essential columns exist and are filled
        map_ready_gdf['population'].fillna(0, inplace=True)
        map_ready_gdf['predicted_cases'].fillna(0, inplace=True)

    else:
        st.warning("No forecast dates found in the data.")
        selected_date = None
        # Use the base merged data for the map, showing all shapes but no case data
        map_ready_gdf = merged_data.copy()
        map_ready_gdf['population'].fillna(0, inplace=True)
        map_ready_gdf['predicted_cases'].fillna(0, inplace=True)


    # --- Sidebar Controls ---
    st.sidebar.header("Forecast Trajectory")
    kabupaten_list = sorted(forecast_data['kabupaten_standard'].unique())
    selected_kabupaten = st.sidebar.selectbox("Select a Kabupaten/Kota:", kabupaten_list)
    
    st.sidebar.subheader(f"4-Week Forecast for {selected_kabupaten}")
    trajectory_df = forecast_data[forecast_data['kabupaten_standard'] == selected_kabupaten].copy()
    
    if not trajectory_df.empty:
        population_for_kab = trajectory_df['population'].iloc[0]
        if population_for_kab > 0:
            trajectory_df['incidence_rate'] = (trajectory_df['predicted_cases'] / population_for_kab) * 100000
        else:
            trajectory_df['incidence_rate'] = 0
        
        chart_df = trajectory_df[['Date', 'predicted_cases']].set_index('Date')
        st.sidebar.line_chart(chart_df, y='predicted_cases')
        
        display_df = trajectory_df[['Date', 'predicted_cases', 'incidence_rate']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df.rename(columns={
            'Date': 'Week', 'predicted_cases': 'Forecast Cases', 'incidence_rate': 'Incidence Rate (/100k)'
        }, inplace=True)
        st.sidebar.dataframe(display_df.set_index('Week'), use_container_width=True)


    # --- Main Map Section ---
    st.subheader(f"National Risk Overview (Forecast for {selected_date.strftime('%Y-%m-%d') if selected_date else 'N/A'})")
    
    # FIX: Use a dynamic key for the map to force re-render on date change
    map_object = create_map(map_ready_gdf)
    st_folium(map_object, width='100%', height=550, key=f"map_{selected_date}")

    # --- Top 10 Regions Section ---
    if selected_date:
        top10_df = map_ready_gdf[map_ready_gdf['Date'] == selected_date].nlargest(10, 'predicted_cases')
        
        if not top10_df.empty:
            top10_df['incidence_rate'] = top10_df.apply(
                lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
                axis=1
            )
            top10_df_display = top10_df[['NAME_2', 'predicted_cases', 'incidence_rate']].copy()
            top10_df_display.rename(columns={
                'NAME_2': 'Region', 'predicted_cases': 'Predicted Cases', 'incidence_rate': 'Incidence Rate (/100k)'
            }, inplace=True)
            st.subheader(f"Top 10 Highest Predicted Cases for {selected_date.strftime('%Y-%m-%d')}")
            st.dataframe(top10_df_display, hide_index=True, use_container_width=True)

else:
    st.error("Data files could not be loaded or processed. Please check the logs in the sidebar.")
