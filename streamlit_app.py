# File: streamlit_app.py
# The "Face" of the EWARS-ID system - Final Production Version
# FINAL FIXES: Resolves JSON serialization error, data loading warnings, and excessive re-runs.
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

# --- Session State Initialization ---
# Use session state to store loaded data and avoid re-fetching on every interaction
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'loaded_forecast' not in st.session_state:
    st.session_state.loaded_forecast = None

# --- Helper Function for Name Standardization ---
@st.cache_data
def standardize_name(name):
    """
    Creates a reliable, simple key for merging by standardizing names from both
    the forecast CSV and the GeoJSON file.
    """
    if not isinstance(name, str):
        return None
    name_lower = name.lower().strip()
    prefixes = ['kabupaten administrasi', 'kota administrasi', 'kabupaten', 'kota', 'kab.']
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            name_lower = name_lower.replace(prefix, '', 1).strip()
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
        # --- FIX: Corrected the URL construction by removing extra spaces ---
        forecast_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{forecast_path}"
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        geojson_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{geojson_path}"
        geojson_response = requests.get(geojson_url, headers=headers)
        geojson_response.raise_for_status()
        # --- Read the downloaded content ---
        # FIX: Removed deprecated 'date_parser' argument.
        forecast_df = pd.read_csv(
            io.StringIO(forecast_response.text),
            parse_dates=['Date']
        )
        forecast_df.rename(columns={
            'Kabupaten_Standard': 'kabupaten_standard',
            'Predicted_Cases': 'predicted_cases',
            'Population': 'population'
        }, inplace=True)
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))
        # --- Create the reliable merge key on BOTH dataframes ---
        forecast_df['merge_key'] = forecast_df['kabupaten_standard'].apply(standardize_name)
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'].apply(standardize_name)
        # Merge the full forecast dataset with the GeoDataFrame
        merged_gdf = kabupaten_gdf.merge(forecast_df, on='merge_key', how='left')
        return merged_gdf, forecast_df
    except Exception as e:
        st.error(f"FATAL ERROR: An error occurred during data loading or processing: {e}")
        return None, None

def create_map(gdf):
    """
    Creates the Folium map with a choropleth layer and popups.
    This version prevents the JSON serialization error.
    """
    # Work on a copy to avoid modifying the original data
    map_gdf = gdf.copy()
    # --- CRITICAL FIX: Convert Date column to string BEFORE passing to Folium ---
    if 'Date' in map_gdf.columns and pd.api.types.is_datetime64_any_dtype(map_gdf['Date']):
        map_gdf['forecast_week_str'] = map_gdf['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
        # Drop the original timestamp column as it's no longer needed and causes the error
        map_gdf = map_gdf.drop(columns=['Date'])
    else:
        map_gdf['forecast_week_str'] = 'N/A'
    # Ensure data is numeric for calculations
    map_gdf['predicted_cases_numeric'] = pd.to_numeric(map_gdf['predicted_cases'], errors='coerce').fillna(0)
    map_gdf['population_numeric'] = pd.to_numeric(map_gdf['population'], errors='coerce').fillna(0)
    map_gdf['incidence_rate'] = map_gdf.apply(
        lambda row: (row['predicted_cases_numeric'] / row['population_numeric']) * 100000 if row['population_numeric'] > 0 else 0,
        axis=1
    )
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")
    folium.Choropleth(
        geo_data=map_gdf,
        data=map_gdf,
        columns=['merge_key', 'predicted_cases_numeric'],
        key_on='feature.properties.merge_key',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name='Predicted Dengue Cases (Next Week)',
        name='Predicted Cases',
        nan_fill_color='lightgray' # Make missing data more visible
    ).add_to(m)
    # Generate popup HTML using the string date and numeric values
    map_gdf['popup_html'] = map_gdf.apply(
        lambda row: f"""<div style="font-family: sans-serif;">
            <h4>📍 {row.get('kabupaten_standard', row['NAME_2'])}</h4>
            <p><b>Forecast Week:</b> {row['forecast_week_str']}</p>
            <p><b>Predicted Cases:</b> {int(row['predicted_cases_numeric'])}</p>
            <p><b>Population:</b> {int(row['population_numeric']) if not pd.isna(row['population_numeric']) else 'N/A'}</p>
            <p><b>Incidence Rate (/100k):</b> {row['incidence_rate']:.2f}</p>
        </div>""", axis=1)
    folium.GeoJson(
        map_gdf,
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

# --- Load Data Once ---
# Check session state first, then load if needed
if st.session_state.loaded_data is None or st.session_state.loaded_forecast is None:
    with st.spinner("Loading data from GitHub..."):
        merged_data, forecast_data = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)
        st.session_state.loaded_data = merged_data
        st.session_state.loaded_forecast = forecast_data
else:
    merged_data = st.session_state.loaded_data
    forecast_data = st.session_state.loaded_forecast

if merged_data is not None and forecast_data is not None:
    unique_dates = sorted(forecast_data['Date'].unique())
    if unique_dates:
        # Use session state for the selected date to avoid resetting on map interaction
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = unique_dates[0] # Default to first date
        # Update session state only when user selects a new date
        selected_date_from_selectbox = st.selectbox(
            "Select Forecast Date for Map View:",
            unique_dates,
            index=unique_dates.index(st.session_state.selected_date) if st.session_state.selected_date in unique_dates else 0,
            format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
        )
        # Only update session state if the selection actually changed
        if selected_date_from_selectbox != st.session_state.selected_date:
            st.session_state.selected_date = selected_date_from_selectbox
        selected_date = st.session_state.selected_date
        # Filter the merged data for the selected date, keeping unmatched regions for a full map
        map_ready_gdf = merged_data[
            (merged_data['Date'] == selected_date) | (pd.isnull(merged_data['Date']))
        ].copy()
    else:
        st.warning("No forecast dates found in the data.")
        selected_date = None
        map_ready_gdf = merged_data.copy()

    # --- Sidebar Controls ---
    st.sidebar.header("Forecast Trajectory")
    kabupaten_list = sorted(forecast_data['kabupaten_standard'].unique())
    # Use session state for selected kabupaten in sidebar too
    if 'selected_kabupaten' not in st.session_state:
         st.session_state.selected_kabupaten = kabupaten_list[0] if kabupaten_list else None
    selected_kabupaten_from_selectbox = st.sidebar.selectbox(
        "Select a Kabupaten/Kota:",
        kabupaten_list,
        index=kabupaten_list.index(st.session_state.selected_kabupaten) if st.session_state.selected_kabupaten in kabupaten_list else 0
    )
    if selected_kabupaten_from_selectbox != st.session_state.selected_kabupaten:
        st.session_state.selected_kabupaten = selected_kabupaten_from_selectbox
    selected_kabupaten = st.session_state.selected_kabupaten

    if selected_kabupaten:
        st.sidebar.subheader(f"4-Week Forecast for {selected_kabupaten}")
        trajectory_df = forecast_data[forecast_data['kabupaten_standard'] == selected_kabupaten].copy()
        if not trajectory_df.empty:
            population_for_kab = pd.to_numeric(trajectory_df['population'].iloc[0], errors='coerce')
            if pd.notna(population_for_kab) and population_for_kab > 0:
                trajectory_df['incidence_rate'] = (pd.to_numeric(trajectory_df['predicted_cases'], errors='coerce') / population_for_kab) * 100000
            else:
                trajectory_df['incidence_rate'] = 0
            chart_df = trajectory_df[['Date', 'predicted_cases']].set_index('Date')
            # Ensure predicted_cases is numeric for the chart
            chart_df['predicted_cases'] = pd.to_numeric(chart_df['predicted_cases'], errors='coerce')
            st.sidebar.line_chart(chart_df, y='predicted_cases')
            display_df = trajectory_df[['Date', 'predicted_cases', 'incidence_rate']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df.rename(columns={
                'Date': 'Week',
                'predicted_cases': 'Forecast Cases',
                'incidence_rate': 'Incidence Rate (/100k)'
            }, inplace=True)
            # Ensure columns are of correct type for display
            display_df['Forecast Cases'] = pd.to_numeric(display_df['Forecast Cases'], errors='coerce')
            st.sidebar.dataframe(display_df.set_index('Week'), use_container_width=True)

    # --- Main Map Section ---
    st.subheader(f"National Risk Overview (Forecast for {selected_date.strftime('%Y-%m-%d') if selected_date else 'N/A'})")
    # --- KEY FIX: Use `returned_objects=[]` and a stable key ---
    # Setting `returned_objects=[]` tells st_folium we don't need interaction data back,
    # which can sometimes prevent re-runs. A stable key based on the *data* helps too,
    # but we use the selected date which changes when needed.
    map_object = create_map(map_ready_gdf)
    st_folium(
        map_object,
        width='100%',
        height=550,
        key=f"map_{selected_date.strftime('%Y%m%d') if selected_date else 'default'}",
        returned_objects=[] # Crucial for preventing re-runs on interaction
    )

    # --- NEW SECTION: Display Totals Below the Map ---
    if selected_date:
        # Filter data for the selected date to calculate totals
        totals_df = map_ready_gdf[map_ready_gdf['Date'] == selected_date].copy()

        if not totals_df.empty:
            # Ensure columns are numeric
            totals_df['predicted_cases'] = pd.to_numeric(totals_df['predicted_cases'], errors='coerce')
            totals_df['population'] = pd.to_numeric(totals_df['population'], errors='coerce')

            # Calculate Totals
            total_population = totals_df['population'].sum()
            total_predicted_cases = totals_df['predicted_cases'].sum()
            # Calculate weighted average incidence rate
            # Weighted by population to avoid simple average of rates
            valid_incidence_rows = totals_df.dropna(subset=['population', 'predicted_cases'])
            valid_incidence_rows = valid_incidence_rows[valid_incidence_rows['population'] > 0]
            if not valid_incidence_rows.empty:
                 total_weighted_cases = valid_incidence_rows['predicted_cases'].sum()
                 total_weighted_population = valid_incidence_rows['population'].sum()
                 average_incidence_rate = (total_weighted_cases / total_weighted_population) * 100000
            else:
                 average_incidence_rate = 0

            # Display Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Population (Forecast Regions)", f"{total_population:,.0f}")
            col2.metric("Total Predicted Cases (Next Week)", f"{total_predicted_cases:,.0f}")
            col3.metric("Weighted Avg Incidence Rate (/100k)", f"{average_incidence_rate:.2f}")

        else:
            st.info("No data available for totals calculation on the selected date.")
    else:
        st.info("Please select a forecast date to see totals.")

    # --- Top 10 Regions Section ---
    if selected_date:
        # Create a clean dataframe for top 10 calculation
        top10_df = map_ready_gdf[map_ready_gdf['predicted_cases'].notna()].copy()
        top10_df['predicted_cases'] = pd.to_numeric(top10_df['predicted_cases'], errors='coerce')
        top10_df['population'] = pd.to_numeric(top10_df['population'], errors='coerce')
        top10_df_clean = top10_df.dropna(subset=['predicted_cases']).copy()
        if not top10_df_clean.empty:
            top10_df_clean = top10_df_clean.nlargest(10, 'predicted_cases')
            top10_df_clean['incidence_rate'] = top10_df_clean.apply(
                lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
                axis=1
            )
            top10_df_display = top10_df_clean[['NAME_2', 'predicted_cases', 'incidence_rate']].copy()
            top10_df_display.rename(columns={
                'NAME_2': 'Region',
                'predicted_cases': 'Predicted Cases',
                'incidence_rate': 'Incidence Rate (/100k)'
            }, inplace=True)
            # Ensure numeric types for display
            top10_df_display['Predicted Cases'] = top10_df_display['Predicted Cases'].astype(int)
            st.subheader(f"Top 10 Highest Predicted Cases for {selected_date.strftime('%Y-%m-%d')}")
            st.dataframe(top10_df_display, hide_index=True, use_container_width=True)
        else:
            st.info("No valid data available for the Top 10 list for the selected date.")
else:
    st.error("Data files could not be loaded. Please check deployment logs for more information.")
