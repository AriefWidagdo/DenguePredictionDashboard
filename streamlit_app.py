# File: app.py
# The "Face" of the EWARS-ID system - Final Production Version
# Data Loading Update: Now fetches data from a private GitHub repository.
# Updated to use january_2024_predictions.csv and show Population & Incidence Rate.
# Uses internal logic to convert Kabupaten_Standard names to match GeoJSON NAME_2.

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

# --- NEW: More Robust Helper Function for Name Conversion ---
def convert_kabupaten_standard_to_name2(kab_standard_name):
    """
    Converts a 'Kabupaten_Standard' name (e.g., 'AcehBarat', 'KOTA JAMBI', 'Bekasi')
    to the GADM 'NAME_2' format (e.g., 'Kabupaten Aceh Barat', 'Kota Jambi', 'Kota Bekasi').

    This version is robust and handles various input formats and special administrative names.
    """
    if not isinstance(kab_standard_name, str):
        return None

    # Step 1: Standardize the input to a predictable CamelCase format.
    words = kab_standard_name.strip().title().split()
    camel_case_name = "".join(words)

    # Step 2: Handle special cases and known cities/regencies that need explicit mapping.
    # This is crucial for distinguishing between cities and regencies with the same core name.
    special_cases = {
        # FIX: Explicitly map "Bekasi" to "Kota Bekasi" and add "KabupatenBekasi"
        "Bekasi": "Kota Bekasi",
        "KabupatenBekasi": "Kabupaten Bekasi",
        "Bogor": "Kota Bogor",
        "KabupatenBogor": "Kabupaten Bogor",
        "Jambi": "Kota Jambi",
        "Gunungsitoli": "Kota Gunungsitoli",
        "Ternate": "Kota Ternate",
        "TidoreKepulauan": "Kota Tidore Kepulauan",
        "JakartaTimur": "Kota Administrasi Jakarta Timur",
        "JakartaSelatan": "Kota Administrasi Jakarta Selatan",
        "JakartaBarat": "Kota Administrasi Jakarta Barat",
        "JakartaUtara": "Kota Administrasi Jakarta Utara",
        "JakartaPusat": "Kota Administrasi Jakarta Pusat",
        "Batam": "Kota Batam",
        "Ambon": "Kota Ambon",
        "Balikpapan": "Kota Balikpapan",
        "Banjarmasin": "Kota Banjarmasin",
        "BandarLampung": "Kota Bandar Lampung"
        # Add other known city/regency pairs here as needed
    }
    if camel_case_name in special_cases:
        return special_cases[camel_case_name]

    # Step 3: Apply the original logic for general cases (now on the clean camel_case_name).
    if camel_case_name.startswith("Kota"):
        prefix = "Kota"
        core_name = camel_case_name[4:]
    else:
        prefix = "Kabupaten"
        core_name = camel_case_name

    # Insert spaces before capital letters to format the core name.
    spaced_core_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', core_name).strip()

    return f"{prefix} {spaced_core_name}"

# --- Caching Functions ---
@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """
    Loads and prepares all data from a private GitHub repo,
    returning a map-ready GDF and a full forecast DF.
    Converts kabupaten names internally for matching.
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

        # --- Processing logic with the NEW robust Name Conversion ---
        forecast_df['converted_name_2'] = forecast_df['kabupaten_standard'].apply(convert_kabupaten_standard_to_name2)

        forecast_df = forecast_df.sort_values(by=['kabupaten_standard', 'Date'])
        first_week_df = forecast_df.loc[forecast_df.groupby('kabupaten_standard')['Date'].idxmin()].copy()
        first_week_df['forecast_week_str'] = first_week_df['Date'].dt.strftime('%Y-%m-%d')

        # Prepare for merging
        first_week_df['merge_key'] = first_week_df['converted_name_2']
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2']

        merged_gdf = kabupaten_gdf.merge(first_week_df, on='merge_key', how='left')

        merged_gdf['predicted_cases'].fillna(0, inplace=True)
        merged_gdf['population'].fillna(0, inplace=True)
        merged_gdf['kabupaten_display'] = merged_gdf['kabupaten_standard'].fillna(merged_gdf['NAME_2'])

        map_ready_gdf = merged_gdf[['geometry', 'merge_key', 'kabupaten_display', 'predicted_cases', 'population', 'NAME_2', 'forecast_week_str']].copy()
        map_ready_gdf.rename(columns={'kabupaten_display': 'kabupaten'}, inplace=True)

        return map_ready_gdf, forecast_df

    except requests.exceptions.RequestException as e:
        st.error(f"FATAL ERROR: Could not fetch data from GitHub. Check token and repo details. Error: {e}")
        return None, None
    except KeyError as e:
        st.error(f"FATAL ERROR: Missing key in secrets or data. Error: {e}. Check GITHUB_TOKEN or CSV column names.")
        return None, None
    except Exception as e:
        st.error(f"FATAL ERROR: An error occurred during data processing: {e}")
        return None, None

# Map function using Folium's Choropleth
def create_map(gdf):
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
            <p><b>Forecast Week:</b> {row.get('forecast_week_str', 'N/A')}</p>
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

OWNER = "AriefWidagdo"
PRIVATE_REPO_NAME = "data-raw"
FORECAST_FILE_PATH = "january_2024_predictions.csv"
GEOJSON_PATH = "gadm41_IDN_2.json"

map_data, full_forecast_df = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)

if map_data is not None and full_forecast_df is not None and not map_data.empty:

    unique_dates = sorted(full_forecast_df['Date'].unique())
    if unique_dates:
        selected_date = st.selectbox(
            "Select Forecast Date for Map View:",
            unique_dates,
            format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
        )

        map_data_for_date_df = full_forecast_df[full_forecast_df['Date'] == selected_date].copy()

        map_data_for_date_df['merge_key'] = map_data_for_date_df['kabupaten_standard'].apply(convert_kabupaten_standard_to_name2)

        kabupaten_gdf_base = map_data[['geometry', 'NAME_2', 'merge_key']].drop_duplicates(subset=['merge_key'])

        map_ready_gdf_for_date = kabupaten_gdf_base.merge(map_data_for_date_df, on='merge_key', how='left')

        map_ready_gdf_for_date['predicted_cases'].fillna(0, inplace=True)
        map_ready_gdf_for_date['population'].fillna(0, inplace=True)
        map_ready_gdf_for_date['kabupaten_display'] = map_ready_gdf_for_date['kabupaten_standard'].fillna(map_ready_gdf_for_date['NAME_2'])
        map_ready_gdf_for_date['forecast_week_str'] = selected_date.strftime('%Y-%m-%d')

        map_ready_gdf_for_date = map_ready_gdf_for_date[['geometry', 'merge_key', 'kabupaten_display', 'predicted_cases', 'population', 'NAME_2', 'forecast_week_str']].copy()
        map_ready_gdf_for_date.rename(columns={'kabupaten_display': 'kabupaten'}, inplace=True)

    else:
        st.warning("No forecast dates found in the data.")
        selected_date = None
        map_ready_gdf_for_date = map_data

    st.sidebar.header("Forecast Trajectory")
    kabupaten_list = sorted(full_forecast_df['kabupaten_standard'].unique())
    selected_kabupaten = st.sidebar.selectbox("Select a Kabupaten/Kota:", kabupaten_list)

    st.sidebar.subheader(f"4-Week Forecast for {selected_kabupaten}")
    trajectory_df = full_forecast_df[full_forecast_df['kabupaten_standard'] == selected_kabupaten].copy()

    if not trajectory_df.empty and 'population' in trajectory_df.columns and trajectory_df['population'].iloc[0] > 0:
        population_for_kab = trajectory_df['population'].iloc[0]
        trajectory_df['incidence_rate'] = (trajectory_df['predicted_cases'] / population_for_kab) * 100000
    else:
        trajectory_df['incidence_rate'] = 0

    chart_df = trajectory_df[['Date', 'predicted_cases']].set_index('Date')
    st.sidebar.line_chart(chart_df, y='predicted_cases')

    display_df = trajectory_df[['Date', 'predicted_cases', 'incidence_rate']].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df.rename(columns={
        'Date': 'Week',
        'predicted_cases': 'Forecast Cases',
        'incidence_rate': 'Incidence Rate (/100k)'
    }, inplace=True)
    st.sidebar.dataframe(display_df.set_index('Week'), use_container_width=True)

    st.subheader(f"National Risk Overview (Forecast for {selected_date.strftime('%Y-%m-%d') if selected_date else 'Next Week'})")
    if selected_date:
        map_object = create_map(map_ready_gdf_for_date)
    else:
        map_object = create_map(map_data)
    st_folium(map_object, returned_objects=[], width='100%', height=550, key="overview_map")

    if selected_date and not map_ready_gdf_for_date.empty:
        top10_df = map_ready_gdf_for_date.nlargest(10, 'predicted_cases')[['kabupaten', 'predicted_cases', 'population']]
        top10_df['incidence_rate'] = top10_df.apply(
            lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
            axis=1
        )
        top10_df_display = top10_df[['kabupaten', 'predicted_cases', 'incidence_rate']].copy()
        top10_df_display.rename(columns={
            'kabupaten': 'Region',
            'predicted_cases': 'Predicted Cases',
            'incidence_rate': 'Incidence Rate (/100k)'
        }, inplace=True)
        st.subheader(f"Top 10 Highest Predicted Cases for {selected_date.strftime('%Y-%m-%d')}")
        st.dataframe(top10_df_display, hide_index=True, use_container_width=True)
    else:
        st.info("Select a date to view the Top 10 regions.")

else:
    if map_data is not None and (map_data.empty or map_data['predicted_cases'].fillna(0).sum() == 0):
         st.error("Data files loaded, but no forecast data could be matched to map regions. Please check the name conversion logic or data formats.")
    else:
         st.error("Data files could not be loaded. Please check the logs for more information.")
