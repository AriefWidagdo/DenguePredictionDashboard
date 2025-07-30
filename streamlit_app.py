# File: app.py
# The "Face" of the EWARS-ID system - Final Production Version
# Data Loading Update: Now fetches data from a private GitHub repository.

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

# --- Caching Functions ---
### MODIFIED ###: This function has been completely rewritten to securely load data from GitHub Privatr Repo.
@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """
    Loads and prepares all data from a private GitHub repo,
    returning a map-ready GDF and a full forecast DF.
    """
    try:
        # --- Securely fetch files from GitHub ---
        github_token = st.secrets["GITHUB_TOKEN"]
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3.raw",
        }
        
        # Download forecast data
        forecast_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{forecast_path}"
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        
        # Download geojson data
        geojson_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{geojson_path}"
        geojson_response = requests.get(geojson_url, headers=headers)
        geojson_response.raise_for_status()

        # --- Read the downloaded content ---
        forecast_df = pd.read_csv(io.BytesIO(forecast_response.content), parse_dates=['forecast_week'])
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))
        
        # --- Processing logic  ---
        forecast_df['forecast_week_str'] = forecast_df['forecast_week'].dt.strftime('%Y-%m-%d')
        first_week_df = forecast_df.loc[forecast_df.groupby('kabupaten')['forecast_week'].idxmin()]

        def standardize_name(name):
            return str(name).upper().strip().replace("KABUPATEN ", "").replace("KOTA ", "")

        first_week_df['merge_key'] = first_week_df['kabupaten'].apply(standardize_name)
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'].apply(standardize_name)
        
        merged_gdf = kabupaten_gdf.merge(first_week_df, on='merge_key', how='left')
        
        merged_gdf['predicted_cases'] = merged_gdf['predicted_cases'].fillna(0)
        merged_gdf['kabupaten'] = merged_gdf['kabupaten'].fillna(merged_gdf['NAME_2'])
        
        map_ready_gdf = merged_gdf[['geometry', 'merge_key', 'kabupaten', 'predicted_cases', 'NAME_2']]
        
        return map_ready_gdf, forecast_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"FATAL ERROR: Could not fetch data from GitHub. Check your token and repo details. Error: {e}")
        return None, None
    except KeyError:
        st.error("FATAL ERROR: GITHUB_TOKEN not found in secrets. Please add it to your Streamlit app settings.")
        return None, None
    except Exception as e:
        st.error(f"FATAL ERROR: An error occurred during data processing: {e}")
        return None, None

#map function using Folium's Choropleth
def create_map(gdf):
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
    ).add_to(m)

    gdf['popup_html'] = gdf.apply(
        lambda row: f"""<div style="font-family: sans-serif;">
            <h4>📍 {row['kabupaten']}</h4>
            <p><b>Predicted Cases (Next Week):</b> {int(row['predicted_cases'])}</p>
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
FORECAST_FILE_PATH = "january_2024_predictions.csv"  # The path to the file inside the private repo
GEOJSON_PATH = "gadm41_IDN_2.json"         # The path to the file inside the private repo


map_data, full_forecast_df = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)


if map_data is not None and full_forecast_df is not None:
    
    # --- NEW LAYOUT: ALL CONTROLS AND CHARTS IN THE SIDEBAR ---
    st.sidebar.header("Forecast Trajectory")
    
    kabupaten_list = sorted(full_forecast_df['kabupaten'].unique())
    selected_kabupaten = st.sidebar.selectbox(
        "Select a Kabupaten/Kota to inspect its forecast:",
        kabupaten_list
    )
    
    # Move the 4-week forecast elements into the sidebar
    st.sidebar.subheader(f"4-Week Forecast for {selected_kabupaten}")
    
    trajectory_df = full_forecast_df[full_forecast_df['kabupaten'] == selected_kabupaten]
    
    chart_df = trajectory_df[['forecast_week', 'predicted_cases']].set_index('forecast_week')
    st.sidebar.line_chart(chart_df)
    
    display_df = trajectory_df[['forecast_week_str', 'predicted_cases']]
    display_df.rename(columns={'forecast_week_str': 'Week', 'predicted_cases': 'Forecast'}, inplace=True)
    st.sidebar.dataframe(display_df.set_index('Week'))

    # The map now takes up the entire main page area
    st.subheader("National Risk Overview (Forecast for Next Week)")
    map_object = create_map(map_data)
    st_folium(map_object, returned_objects=[], width='100%', height=550, key="overview_map")

else:
    st.error("Data files could not be loaded. Please check the logs for more information.")
