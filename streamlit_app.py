# File: app.py - Updated with proper name matching
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

# --- NEW: Helper Functions for Name Conversion ---
def convert_spaced_to_camelcase(spaced_name):
    """
    Converts spaced Kabupaten_Standard format to CamelCase ADM2 format
    Examples: 'ACEH BARAT' -> 'AcehBarat'
    """
    if not isinstance(spaced_name, str):
        return None
    
    clean_name = spaced_name.strip().upper()
    words = clean_name.split()
    camel_case = ''.join([word.title() for word in words])
    
    return camel_case

def convert_camelcase_to_gadm(camel_name):
    """
    Converts CamelCase to GADM NAME_2 format
    Examples: 'AcehBarat' -> 'Kabupaten Aceh Barat'
    """
    if not isinstance(camel_name, str):
        return None
    
    # Determine if it's a city (Kota) or regency (Kabupaten)
    if camel_name.startswith("Kota"):
        prefix = "Kota"
        core_name = camel_name[4:]  # Remove "Kota"
    else:
        prefix = "Kabupaten"
        core_name = camel_name
    
    # Insert spaces before capital letters
    spaced_core = re.sub(r'(?<!^)(?=[A-Z])', ' ', core_name)
    
    return f"{prefix} {spaced_core}"

# --- Load ADM2 reference list ---
@st.cache_data
def load_adm2_reference():
    """Load the ADM2 reference list"""
    # Your ADM2 list from the document
    adm2_list = [
        "AcehBarat", "AcehBaratDaya", "AcehBesar", "AcehJaya", "AcehSelatan", 
        "AcehSingkil", "AcehTamiang", "AcehTengah", "AcehTenggara", "AcehTimur", 
        "AcehUtara", "Agam", "Alor", "Ambon", "Asahan", "Asmat", "Badung", 
        "Balangan", "Balikpapan", "BandaAceh", "BandarLampung", "Bandung", 
        "BandungBarat", "Banggai", "BanggaiKepulauan", "Bangka", "BangkaBarat", 
        "BangkaSelatan", "BangkaTengah", "Bangkalan", "Bangli", "Banjar", 
        "BanjarBaru", "Banjarmasin", "Banjarnegara", "Bantaeng", "Bantul", 
        "BanyuAsin", "Banyumas", "Banyuwangi", "BaritoKuala", "BaritoSelatan",
        # ... (include the full list from your document)
    ]
    return adm2_list

@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """
    Loads and prepares all data with proper name matching
    """
    try:
        # Load ADM2 reference
        adm2_list = load_adm2_reference()
        
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
        
        # Rename columns to match expected format
        forecast_df.rename(columns={
            'Kabupaten_Standard': 'kabupaten_standard',
            'Predicted_Cases': 'predicted_cases',
            'Population': 'population'
        }, inplace=True)
        
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))
        
        # --- NEW: Create proper name matching ---
        # Step 1: Convert spaced format to CamelCase
        forecast_df['adm2_format'] = forecast_df['kabupaten_standard'].apply(convert_spaced_to_camelcase)
        
        # Step 2: Convert CamelCase to GADM format for GeoJSON matching
        forecast_df['gadm_format'] = forecast_df['adm2_format'].apply(convert_camelcase_to_gadm)
        
        # Step 3: Create mapping keys
        forecast_df['merge_key'] = forecast_df['gadm_format']  # Use GADM format for merging
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2']   # GeoJSON uses NAME_2
        
        # --- Debug: Show matching status ---
        st.write("### Debug: Name Matching Status")
        
        # Check which regions from forecast match ADM2 list
        forecast_adm2_matches = forecast_df['adm2_format'].isin(adm2_list)
        matched_count = forecast_adm2_matches.sum()
        total_count = len(forecast_df['adm2_format'].unique())
        
        st.write(f"Forecast regions matching ADM2 list: {matched_count}/{total_count}")
        
        # Show unmatched regions from forecast
        unmatched_forecast = forecast_df[~forecast_adm2_matches]['kabupaten_standard'].unique()
        if len(unmatched_forecast) > 0:
            st.write("Unmatched forecast regions:")
            st.write(unmatched_forecast[:10])  # Show first 10
        
        # Check GeoJSON matching
        geojson_matches = kabupaten_gdf['NAME_2'].isin(forecast_df['gadm_format'])
        geojson_matched_count = geojson_matches.sum()
        geojson_total_count = len(kabupaten_gdf)
        
        st.write(f"GeoJSON regions with forecast data: {geojson_matched_count}/{geojson_total_count}")
        
        # Sort and prepare data
        forecast_df = forecast_df.sort_values(by=['kabupaten_standard', 'Date'])
        first_week_df = forecast_df.loc[forecast_df.groupby('kabupaten_standard')['Date'].idxmin()].copy()
        first_week_df['forecast_week_str'] = first_week_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Merge with GeoJSON
        merged_gdf = kabupaten_gdf.merge(first_week_df, on='merge_key', how='left')
        
        # Fill missing values
        merged_gdf['predicted_cases'] = merged_gdf['predicted_cases'].fillna(0)
        merged_gdf['population'] = merged_gdf['population'].fillna(0)
        merged_gdf['kabupaten_display'] = merged_gdf['kabupaten_standard'].fillna(merged_gdf['NAME_2'])
        
        # Prepare final GDF
        map_ready_gdf = merged_gdf[[
            'geometry', 'merge_key', 'kabupaten_display', 'predicted_cases', 
            'population', 'NAME_2', 'forecast_week_str'
        ]].copy()
        map_ready_gdf.rename(columns={'kabupaten_display': 'kabupaten'}, inplace=True)
        
        return map_ready_gdf, forecast_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- Rest of your existing functions (create_map, etc.) remain the same ---
def create_map(gdf):
    # Your existing create_map function stays the same
    gdf['incidence_rate'] = gdf.apply(
        lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
        axis=1
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

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

    gdf['popup_html'] = gdf.apply(
        lambda row: f"""<div style="font-family: sans-serif;">
            <h4>📍 {row['kabupaten']}</h4>
            <p><b>Forecast Week:</b> {row['forecast_week_str']}</p>
            <p><b>Predicted Cases:</b> {int(row['predicted_cases'])}</p>
            <p><b>Population:</b> {int(row['population'])}</p>
            <p><b>Incidence Rate (/100k):</b> {row['incidence_rate']:.2f}</p>
        </div>""", axis=1)

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

# --- Rest of your existing app logic remains the same ---
if map_data is not None and full_forecast_df is not None and not map_data.empty:
    # Your existing app logic continues here...
    pass
else:
    st.error("Could not load or match the data properly.")
