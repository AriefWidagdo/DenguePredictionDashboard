# File: app.py - Updated with robust name matching
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="EWARS-ID Dengue Forecast by M Arief Widagdo",
    page_icon="🇮🇩",
    layout="wide"
)

# --- NEW: Helper Function for Name Standardization ---
def standardize_name(name):
    """
    Standardizes Indonesian administrative names to a common, prefix-less format.
    - Converts to lowercase
    - Removes common prefixes like 'kabupaten', 'kota', 'kab.', 'adm.', etc.
    - Strips extra whitespace.
    Example: 'Kabupaten Aceh Barat' -> 'aceh barat'
             'KOTA JAMBI' -> 'jambi'
             'Kota Administrasi Jakarta Timur' -> 'jakarta timur'
    """
    if not isinstance(name, str):
        return None
    
    name_lower = name.lower()
    # List of prefixes to remove, ordered to handle variations
    prefixes = ['kota administrasi', 'kabupaten', 'kota', 'kab.', 'adm.']
    
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            # Remove prefix and strip leading/trailing whitespace
            return name_lower[len(prefix):].strip()
            
    return name_lower.strip()

@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """
    Loads and prepares all data with a robust name matching strategy.
    """
    try:
        # --- Fetch files from GitHub ---
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

        # --- Read the data ---
        forecast_df = pd.read_csv(
            io.StringIO(forecast_response.text),
            parse_dates=['Date'],
            date_parser=pd.to_datetime
        )
        
        # Rename columns to a consistent format
        forecast_df.rename(columns={
            'Kabupaten_Standard': 'kabupaten_standard',
            'Predicted_Cases': 'predicted_cases',
            'Population': 'population'
        }, inplace=True)
        
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))
        
        # --- NEW: Create robust merge key ---
        # Create a standardized, prefix-less key for both dataframes
        forecast_df['merge_key'] = forecast_df['kabupaten_standard'].apply(standardize_name)
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'].apply(standardize_name)
        
        # --- Merge DataFrames ---
        # Perform a left merge to keep all GeoJSON shapes and add forecast data where available
        merged_gdf = kabupaten_gdf.merge(forecast_df, on='merge_key', how='left')

        # --- Debug: Show matching status ---
        st.write("### Debug: Name Matching Status")
        
        # Check how many unique regions from the forecast file were successfully matched
        forecast_regions_total = forecast_df['kabupaten_standard'].nunique()
        matched_regions_count = merged_gdf['kabupaten_standard'].nunique()
        
        st.write(f"Matched forecast regions with GeoJSON: {matched_regions_count}/{forecast_regions_total}")
        
        # Show unmatched regions from the forecast file
        matched_keys = set(merged_gdf.dropna(subset=['kabupaten_standard'])['merge_key'])
        unmatched_forecast_df = forecast_df[~forecast_df['merge_key'].isin(matched_keys)]
        unmatched_regions = unmatched_forecast_df['kabupaten_standard'].unique()

        if len(unmatched_regions) > 0:
            st.write("Unmatched forecast regions:")
            st.write(pd.DataFrame(unmatched_regions, columns=['value']))

        # --- Final Data Preparation ---
        # Sort and get the first week's forecast for the initial map view
        forecast_df = forecast_df.sort_values(by=['kabupaten_standard', 'Date'])
        first_week_df = forecast_df.loc[forecast_df.groupby('kabupaten_standard')['Date'].idxmin()].copy()
        
        # Re-merge to ensure we only have the first week's data for the initial map
        map_ready_gdf = kabupaten_gdf.merge(first_week_df, on='merge_key', how='left')
        
        # Fill missing values for regions in GeoJSON without forecast data
        map_ready_gdf['predicted_cases'].fillna(0, inplace=True)
        map_ready_gdf['population'].fillna(0, inplace=True)
        
        # Use the original GeoJSON name (NAME_2) as a fallback for display
        map_ready_gdf['kabupaten_display'] = map_ready_gdf['kabupaten_standard'].fillna(map_ready_gdf['NAME_2'])
        
        # Ensure forecast_week_str is available for popups
        map_ready_gdf['forecast_week_str'] = map_ready_gdf['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
        
        # Prepare final GDF for the map
        map_ready_gdf.rename(columns={'kabupaten_display': 'kabupaten'}, inplace=True)
        
        return map_ready_gdf, forecast_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def create_map(gdf):
    """Creates the Folium map with choropleth layer and popups."""
    # Calculate incidence rate, handling potential division by zero
    gdf['incidence_rate'] = gdf.apply(
        lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
        axis=1
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    # Choropleth for predicted cases
    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=['merge_key', 'predicted_cases'],
        key_on='feature.properties.merge_key',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name='Predicted Dengue Cases (Next Week)',
        name='Predicted Cases'
    ).add_to(m)

    # Create popup HTML content
    gdf['popup_html'] = gdf.apply(
        lambda row: f"""<div style="font-family: sans-serif;">
            <h4>📍 {row['kabupaten']}</h4>
            <p><b>Forecast Week:</b> {row['forecast_week_str']}</p>
            <p><b>Predicted Cases:</b> {int(row['predicted_cases'])}</p>
            <p><b>Population:</b> {int(row['population'])}</p>
            <p><b>Incidence Rate (/100k):</b> {row['incidence_rate']:.2f}</p>
        </div>""", axis=1)

    # Invisible GeoJson layer for tooltips and popups
    folium.GeoJson(
        gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
        tooltip=folium.features.GeoJsonTooltip(fields=['kabupaten'], aliases=['Region:']),
        popup=folium.features.GeoJsonPopup(fields=['popup_html'], aliases=[''])
    ).add_to(m)
    
    return m

# --- Main App Layout ---
st.title("🇮🇩 EWARS-ID: Dengue Forecast Dashboard")
st.markdown("An operational prototype for near real-time dengue fever forecasting by M Arief Widagdo.")

# --- Load Data ---
OWNER = "AriefWidagdo"
PRIVATE_REPO_NAME = "data-raw"
FORECAST_FILE_PATH = "january_2024_predictions.csv"
GEOJSON_PATH = "gadm41_IDN_2.json"

map_data, full_forecast_df = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)

# --- Display Map and Data Table ---
if map_data is not None and full_forecast_df is not None and not map_data.empty:
    st.success("Data loaded and matched successfully!")
    
    # Create and display the map
    dengue_map = create_map(map_data)
    st_folium(dengue_map, width='100%', height=600)

    # Display the full forecast data in an expandable section
    with st.expander("View Full Forecast Data Table"):
        st.dataframe(full_forecast_df)
else:
    st.error("Could not load or process the data. Please check the debug messages above.")
