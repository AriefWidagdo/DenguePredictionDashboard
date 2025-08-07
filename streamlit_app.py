# File: streamlit_app.py
# The "Face" of the EWARS-ID system - Enhanced Production Version
# Enhanced with better summary metrics, PAST weather integration, and improved features
# Includes PAST weather data (1 month ago) in map popups and sidebar
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
import io
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Attempt to import dateutil for better month subtraction
try:
    from dateutil.relativedelta import relativedelta
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    st.warning("The `python-dateutil` library is not installed. Using approximate month calculation.")

# --- Page Configuration ---
st.set_page_config(
    page_title="EWARS-ID Dengue Forecast by M Arief Widagdo",
    page_icon="🇮🇩",
    layout="wide"
)

# --- Session State Initialization ---
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'loaded_forecast' not in st.session_state:
    st.session_state.loaded_forecast = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None # This will now be a dict {region_name: {date: weather_info}}

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

# --- Weather Data Function (Modified to fetch PAST weather for a specific date) ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_data(region_centroids=None, target_date=None):
    """
    Fetch PAST weather data for a list of region centroids for a specific date (1 month ago).
    If region_centroids is None or target_date is None, returns an empty dict.
    """
    if not region_centroids or target_date is None:
        return {}

    try:
        api_key = st.secrets.get("WEATHER_API_KEY", "")
        if not api_key:
            st.warning("Weather API key not found in secrets.")
            return {}

        # Calculate the date one month ago
        if HAS_DATEUTIL:
            try:
                past_date = target_date - relativedelta(months=1)
            except Exception:
                # Fallback if relativedelta fails for some reason
                past_date = target_date - timedelta(days=30)
        else:
            # Approximate 1 month as 30 days
            past_date = target_date - timedelta(days=30)

        # Convert past_date to Unix timestamp (API expects UTC timestamp)
        # API expects timestamp at 00:00 UTC of the target day for historical data
        past_date_utc_start = datetime(past_date.year, past_date.month, past_date.day)
        timestamp = int(past_date_utc_start.timestamp())

        weather_data = {}
        for location in region_centroids:
            region_name = location['name']
            try:
                lat = float(location['lat'])
                lon = float(location['lon'])
            except (ValueError, TypeError, KeyError):
                st.warning(f"Invalid coordinates for {region_name}: lat={location.get('lat')}, lon={location.get('lon')}")
                continue

            # Use One Call API 3.0 Timemachine endpoint for historical data
            url = f"http://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={api_key}&units=metric"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # The timemachine API returns hourly data for the requested day
                    # We'll take the data closest to noon (12:00) if available, otherwise the first entry
                    hourly_data = data.get('data', [])
                    if hourly_data:
                        # Find the entry closest to noon (index 12 if hourly from 00:00)
                        # A simple approach: take the first entry (00:00 data) or find noon
                        target_hour_index = min(range(len(hourly_data)), key=lambda i: abs(hourly_data[i].get('dt', 0) - (timestamp + 12*3600)))
                        selected_hour = hourly_data[target_hour_index]

                        weather_data[region_name] = {
                            'region_name': region_name,
                            'temperature': selected_hour['temp'],
                            'humidity': selected_hour['humidity'],
                            'description': selected_hour['weather'][0]['description'].title() if selected_hour.get('weather') else 'N/A',
                            'feels_like': selected_hour.get('feels_like', selected_hour['temp']), # feels_like might not be in historical
                            'weather_date': past_date.strftime('%Y-%m-%d') # Store the date for display
                        }
                    else:
                        st.warning(f"No hourly weather data found for {region_name} on {past_date.strftime('%Y-%m-%d')}")
                elif response.status_code == 401:
                    st.error("Invalid API key for OpenWeatherMap or access to historical data not available with your subscription.")
                    return {} # Stop trying if auth fails
                else:
                    st.warning(f"Failed to fetch weather for {region_name} (Status {response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error fetching weather for {region_name}: {e}")
            except KeyError as e:
                st.error(f"Unexpected data structure in weather response for {region_name}: Missing key {e}")
            except Exception as e:
                st.error(f"Error processing weather data for {region_name}: {e}")

        return weather_data
    except Exception as e:
        st.error(f"Error in get_weather_data: {e}")
        return {}

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
            parse_dates=['Date']
        )
        forecast_df.rename(columns={
            'Kabupaten_Standard': 'kabupaten_standard',
            'Predicted_Cases': 'predicted_cases',
            'Population': 'population'
        }, inplace=True)
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))

        # --- Project GeoDataFrame for accurate centroid calculation (optional but good practice) ---
        # Using EPSG:32748 (UTM Zone 48S) for Indonesia. Adjust if needed for northern regions.
        # If projection causes issues, skip this step and calculate centroids in WGS84.
        try:
            # Check if it's already projected or geographic
            if kabupaten_gdf.crs is None or kabupaten_gdf.crs.is_geographic:
                 # If geographic (like WGS84), project it
                 kabupaten_gdf_projected = kabupaten_gdf.to_crs(epsg=32748)
            else:
                 # If already projected, use as is for centroid calculation
                 kabupaten_gdf_projected = kabupaten_gdf

            # Calculate centroids in the projected CRS
            centroids_projected = kabupaten_gdf_projected.geometry.centroid
            # Transform centroids back to WGS84 (lat/lon) for OpenWeatherMap API
            centroids_wgs84 = centroids_projected.to_crs(epsg=4326)
            # Extract lat and lon
            kabupaten_gdf['lat'] = centroids_wgs84.y
            kabupaten_gdf['lon'] = centroids_wgs84.x
        except Exception as proj_error:
            # Fallback: Calculate centroids directly in WGS84 if projection fails
            st.warning("Could not project GeoDataFrame for centroid calculation. Calculating centroids in WGS84. Accuracy might vary.")
            centroids_wgs84 = kabupaten_gdf.geometry.centroid
            kabupaten_gdf['lat'] = centroids_wgs84.y
            kabupaten_gdf['lon'] = centroids_wgs84.x


        # --- Create the reliable merge key on BOTH dataframes ---
        forecast_df['merge_key'] = forecast_df['kabupaten_standard'].apply(standardize_name)
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'].apply(standardize_name)
        # Merge the full forecast dataset with the GeoDataFrame
        merged_gdf = kabupaten_gdf.merge(forecast_df, on='merge_key', how='left')
        return merged_gdf, forecast_df
    except Exception as e:
        st.error(f"FATAL ERROR: An error occurred during data loading or processing: {e}")
        return None, None

def create_map(gdf, region_weather_data=None, weather_date_str="N/A"):
    """
    Creates the Folium map with a choropleth layer and popups.
    Enhanced version with better styling, risk levels, AND REGIONAL PAST WEATHER.
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
    # Create risk categories
    map_gdf['risk_level'] = pd.cut(
        map_gdf['incidence_rate'],
        bins=[0, 10, 25, 50, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
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
        nan_fill_color='lightgray'
    ).add_to(m)

    # Enhanced popup HTML with risk level, better styling, AND PAST WEATHER
    def generate_popup_html(row):
        region_name = row.get('kabupaten_standard', row['NAME_2'])
        html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 250px;">
            <h4 style="margin-bottom: 10px; color: #2c3e50;">📍 {region_name}</h4>
            <div style="border-left: 4px solid #e74c3c; padding-left: 12px; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                <p style="margin: 5px 0;"><strong>📅 Forecast Week:</strong> {row['forecast_week_str']}</p>
                <p style="margin: 5px 0;"><strong>🦟 Predicted Cases:</strong> {int(row['predicted_cases_numeric'])}</p>
                <p style="margin: 5px 0;"><strong>👥 Population:</strong> {int(row['population_numeric']):,} {'' if not pd.isna(row['population_numeric']) else 'N/A'}</p>
                <p style="margin: 5px 0;"><strong>📊 Incidence Rate:</strong> {row['incidence_rate']:.2f}/100k</p>
                <p style="margin: 5px 0;"><strong>⚠️ Risk Level:</strong>
                    <span style="color: {'#e74c3c' if row['risk_level'] in ['High', 'Very High'] else '#f39c12' if row['risk_level'] == 'Medium' else '#27ae60'}; font-weight: bold;">
                        {row['risk_level']}
                    </span>
                </p>
                <!-- NEW: PAST Weather Section -->
                <p style="margin: 5px 0; padding-top: 10px; border-top: 1px dashed #ccc;">
                    <strong>🌤️ Weather (1 Month Ago - {weather_date_str}):</strong><br>
        """
        # Add weather data if available
        if region_weather_data and isinstance(region_weather_data, dict):
            weather_info = region_weather_data.get(region_name)
            if weather_info:
                html += f"""
                    {weather_info['temperature']:.1f}°C ({weather_info['description']})<br>
                    <small>💧 Humidity: {weather_info['humidity']}%</small>
                """
            else:
                html += "Weather data for this region is unavailable."
        else:
             html += "Weather data currently unavailable."

        html += """
                </p>
            </div>
        </div>
        """
        return html

    map_gdf['popup_html'] = map_gdf.apply(generate_popup_html, axis=1)

    folium.GeoJson(
        map_gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
        tooltip=folium.features.GeoJsonTooltip(fields=['NAME_2'], aliases=['Region:']),
        popup=folium.features.GeoJsonPopup(fields=['popup_html'], aliases=[''])
    ).add_to(m)
    return m

def create_trend_chart(forecast_data):
    """Create a trend chart showing national forecast over time."""
    if forecast_data is None or forecast_data.empty:
        return None
    # Aggregate by date
    trend_data = forecast_data.groupby('Date').agg({
        'predicted_cases': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'population': lambda x: pd.to_numeric(x, errors='coerce').sum()
    }).reset_index()
    trend_data['incidence_rate'] = (trend_data['predicted_cases'] / trend_data['population']) * 100000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['predicted_cases'],
        mode='lines+markers',
        name='Total Predicted Cases',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8, color='#e74c3c'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Cases:</b> %{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title={
            'text': 'National Dengue Forecast Trend',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Total Predicted Cases',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    return fig

# --- Main App Layout ---
st.title("🇮🇩 EWARS-ID: Enhanced Dengue Forecast Dashboard")
st.markdown("*An operational prototype for near real-time dengue fever forecasting by M Arief Widagdo*")

# Configuration
OWNER = "AriefWidagdo"
PRIVATE_REPO_NAME = "data-raw"
FORECAST_FILE_PATH = "january_2024_predictions.csv"
GEOJSON_PATH = "gadm41_IDN_2.json"

# --- Load Data Once ---
if st.session_state.loaded_data is None or st.session_state.loaded_forecast is None:
    with st.spinner("🔄 Loading data from GitHub..."):
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
            st.session_state.selected_date = unique_dates[0]
        # Date selector with enhanced formatting
        selected_date_from_selectbox = st.selectbox(
            "📅 Select Forecast Date for Analysis:",
            unique_dates,
            index=unique_dates.index(st.session_state.selected_date) if st.session_state.selected_date in unique_dates else 0,
            format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d (%A)')
        )
        if selected_date_from_selectbox != st.session_state.selected_date:
            st.session_state.selected_date = selected_date_from_selectbox
        selected_date = st.session_state.selected_date
        # Filter the merged data for the selected date
        map_ready_gdf = merged_data[
            (merged_data['Date'] == selected_date) | (pd.isnull(merged_data['Date']))
        ].copy()
    else:
        st.warning("⚠️ No forecast dates found in the data.")
        selected_date = None
        map_ready_gdf = merged_data.copy()

    # --- Fetch PAST Weather Data for Regions ---
    region_weather_data = None
    weather_date_str = "N/A"
    if selected_date:
        # Calculate the weather date (1 month ago)
        if HAS_DATEUTIL:
            try:
                weather_date = selected_date - relativedelta(months=1)
            except Exception:
                weather_date = selected_date - timedelta(days=30)
        else:
            weather_date = selected_date - timedelta(days=30)
        weather_date_str = weather_date.strftime('%Y-%m-%d')

        # Check if weather data for this date is already cached
        cache_key = f"weather_{weather_date_str}"
        if st.session_state.weather_data is None or cache_key not in st.session_state.weather_data:
            with st.spinner(f"🌦️ Loading regional weather data for {weather_date_str}..."):
                # Prepare list of regions with centroids for weather API
                region_list = map_ready_gdf[['kabupaten_standard', 'lat', 'lon']].rename(columns={'kabupaten_standard': 'name'}).to_dict('records')
                fetched_weather_data = get_weather_data(region_list, selected_date) # Pass the selected_date for calculation
                # Store in session state under the date key
                if st.session_state.weather_data is None:
                    st.session_state.weather_data = {}
                st.session_state.weather_data[cache_key] = fetched_weather_data
                region_weather_data = fetched_weather_data
        else:
            region_weather_data = st.session_state.weather_data[cache_key]
    else:
        region_weather_data = {}


    # --- Main Map Section (MOVED UP) ---
    st.subheader(f"🗺️ Interactive Risk Map - {selected_date.strftime('%Y-%m-%d') if selected_date else 'N/A'}")
    # Pass weather data and the calculated weather date string
    map_object = create_map(map_ready_gdf, region_weather_data, weather_date_str)
    st_folium(
        map_object,
        width='100%',
        height=600,
        key=f"enhanced_map_{selected_date.strftime('%Y%m%d') if selected_date else 'default'}",
        returned_objects=[]
    )

    # --- ENHANCED NATIONAL SUMMARY SECTION (MOVED DOWN) ---
    st.subheader(f"📊 National Summary for {selected_date.strftime('%Y-%m-%d (%A)') if selected_date else 'N/A'}")
    if selected_date:
        # Use forecast_data directly for more reliable filtering
        summary_data = forecast_data[
            (forecast_data['Date'] == selected_date) &
            (forecast_data['predicted_cases'].notna()) &
            (forecast_data['population'].notna())
        ].copy()
        if not summary_data.empty:
            # Ensure numeric data
            summary_data['predicted_cases'] = pd.to_numeric(summary_data['predicted_cases'], errors='coerce')
            summary_data['population'] = pd.to_numeric(summary_data['population'], errors='coerce')
            # Remove rows that became NaN after conversion
            summary_data = summary_data.dropna(subset=['predicted_cases', 'population'])
            if not summary_data.empty:
                # Calculate metrics
                total_population = summary_data['population'].sum()
                total_predicted_cases = summary_data['predicted_cases'].sum()
                regions_with_data = len(summary_data)
                if total_population > 0:
                    average_incidence_rate = (total_predicted_cases / total_population) * 100000
                else:
                    average_incidence_rate = 0
                # Display metrics in enhanced format
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="🏘️ Total Population",
                        value=f"{total_population:,.0f}",
                        help="Total population in regions with forecast data"
                    )
                with col2:
                    st.metric(
                        label="🦟 Predicted Cases",
                        value=f"{total_predicted_cases:,.0f}",
                        help="Total predicted dengue cases for next week"
                    )
                with col3:
                    st.metric(
                        label="📈 Avg Incidence Rate",
                        value=f"{average_incidence_rate:.2f}/100k",
                        help="Weighted average incidence rate per 100,000 population"
                    )
                with col4:
                    st.metric(
                        label="🗺️ Regions Covered",
                        value=f"{regions_with_data}",
                        help="Number of regions with forecast data"
                    )
                # Risk level distribution
                summary_data['incidence_rate'] = (summary_data['predicted_cases'] / summary_data['population']) * 100000
                summary_data['risk_level'] = pd.cut(
                    summary_data['incidence_rate'],
                    bins=[0, 10, 25, 50, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    include_lowest=True
                )
                risk_counts = summary_data['risk_level'].value_counts()
                if not risk_counts.empty:
                    st.subheader("🚨 Risk Level Distribution")
                    risk_cols = st.columns(len(risk_counts))
                    risk_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Very High': '🔴'}
                    for i, (level, count) in enumerate(risk_counts.items()):
                        if i < len(risk_cols):
                            risk_cols[i].metric(
                                label=f"{risk_colors.get(level, '⚪')} {level} Risk",
                                value=f"{count} region{'s' if count != 1 else ''}",
                                help=f"Regions classified as {level.lower()} risk"
                            )
            else:
                st.warning("⚠️ No valid numeric data available for calculations on the selected date.")
        else:
            st.info("ℹ️ No forecast data available for the selected date.")
    # --- NATIONAL TREND CHART (MOVED DOWN) ---
    if not forecast_data.empty:
        st.subheader("📈 National Forecast Trend")
        trend_fig = create_trend_chart(forecast_data)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        # Weather data section
        st.subheader("🌤️ Weather Information (1 Month Ago)")
        if selected_date and st.button("Refresh Weather Data", help=f"Update weather conditions for {weather_date_str}"):
            with st.spinner(f"Fetching weather data for {weather_date_str}..."):
                # Prepare list of regions with centroids for weather API
                region_list = map_ready_gdf[['kabupaten_standard', 'lat', 'lon']].rename(columns={'kabupaten_standard': 'name'}).to_dict('records')
                # Force refresh by clearing cache for this date
                cache_key = f"weather_{weather_date_str}"
                if st.session_state.weather_data is not None and cache_key in st.session_state.weather_data:
                     del st.session_state.weather_data[cache_key]
                fetched_weather_data = get_weather_data(region_list, selected_date)
                if st.session_state.weather_data is None:
                    st.session_state.weather_data = {}
                st.session_state.weather_data[cache_key] = fetched_weather_data
                region_weather_data = fetched_weather_data # Update local variable for immediate use
                st.success(f"Weather data refreshed for {weather_date_str}!")
                # Note: st.rerun() might be needed if the map doesn't update automatically, but often Streamlit handles it.

        # Display weather status (summary or error)
        if region_weather_data is not None and isinstance(region_weather_data, dict) and region_weather_data:
            st.success(f"PAST Weather data loaded for {len(region_weather_data)} regions (Date: {weather_date_str})!")
            # Optionally, display a summary or sample of weather data
            # For now, just a success message is shown
        else:
            if 'WEATHER_API_KEY' not in st.secrets:
                st.info("Add `WEATHER_API_KEY` to secrets to enable weather data")
            elif selected_date:
                st.warning(f"PAST Weather data currently unavailable for {weather_date_str}.")
            else:
                st.info("Select a forecast date to load weather data.")

        st.markdown("---")
        # Region analysis
        st.subheader("🔍 Region Analysis")
        kabupaten_list = sorted(forecast_data['kabupaten_standard'].unique())
        if 'selected_kabupaten' not in st.session_state and kabupaten_list:
            st.session_state.selected_kabupaten = kabupaten_list[0]
        if kabupaten_list:
            selected_kabupaten = st.selectbox(
                "Select a Region:",
                kabupaten_list,
                index=kabupaten_list.index(st.session_state.selected_kabupaten) if st.session_state.selected_kabupaten in kabupaten_list else 0
            )
            if selected_kabupaten != st.session_state.get('selected_kabupaten'):
                st.session_state.selected_kabupaten = selected_kabupaten
            # Region-specific analysis
            if selected_kabupaten and selected_date:
                trajectory_df = forecast_data[forecast_data['kabupaten_standard'] == selected_kabupaten].copy()
                if not trajectory_df.empty:
                    st.markdown(f"### 📊 {selected_kabupaten}")
                    # Convert to numeric
                    trajectory_df['predicted_cases'] = pd.to_numeric(trajectory_df['predicted_cases'], errors='coerce')
                    population_for_kab = pd.to_numeric(trajectory_df['population'].iloc[0], errors='coerce')
                    if pd.notna(population_for_kab) and population_for_kab > 0:
                        trajectory_df['incidence_rate'] = (trajectory_df['predicted_cases'] / population_for_kab) * 100000
                    else:
                        trajectory_df['incidence_rate'] = 0
                    # Current week metrics
                    current_data = trajectory_df[trajectory_df['Date'] == selected_date]
                    if not current_data.empty:
                        current_cases = current_data['predicted_cases'].iloc[0]
                        current_incidence = current_data['incidence_rate'].iloc[0]
                        st.metric("🦟 Cases This Week", f"{current_cases:.0f}")
                        st.metric("📊 Incidence Rate", f"{current_incidence:.2f}/100k")

                        # --- NEW: Display PAST Weather for Selected Region in Sidebar ---
                        if region_weather_data and isinstance(region_weather_data, dict):
                            region_weather = region_weather_data.get(selected_kabupaten)
                            if region_weather:
                                st.markdown(f"🌤️ **Weather (1 Month Ago - {region_weather['weather_date']}):**")
                                col1, col2 = st.columns(2)
                                col1.metric("🌡️ Temperature", f"{region_weather['temperature']:.1f}°C")
                                col2.metric("💧 Humidity", f"{region_weather['humidity']}%")
                                st.caption(f"{region_weather['description']}")
                            else:
                                st.info(f"Weather data for {selected_kabupaten} on {weather_date_str} not available.")
                        # --- END NEW WEATHER DISPLAY ---

                    # Mini trend chart
                    chart_df = trajectory_df[['Date', 'predicted_cases']].set_index('Date')
                    st.line_chart(chart_df, height=200)
                    # Data table
                    display_df = trajectory_df[['Date', 'predicted_cases', 'incidence_rate']].copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    display_df.rename(columns={
                        'Date': 'Week',
                        'predicted_cases': 'Forecast Cases',
                        'incidence_rate': 'Rate (/100k)'
                    }, inplace=True)
                    st.dataframe(
                        display_df.set_index('Week'),
                        use_container_width=True,
                        column_config={
                            "Forecast Cases": st.column_config.NumberColumn(format="%.0f"),
                            "Rate (/100k)": st.column_config.NumberColumn(format="%.2f")
                        }
                    )
    # --- Top 10 Regions Section ---
    if selected_date:
        st.subheader(f"🏆 Top 10 Highest Risk Regions for {selected_date.strftime('%Y-%m-%d')}")
        top10_df = forecast_data[forecast_data['Date'] == selected_date].copy()
        if not top10_df.empty:
            top10_df['predicted_cases'] = pd.to_numeric(top10_df['predicted_cases'], errors='coerce')
            top10_df['population'] = pd.to_numeric(top10_df['population'], errors='coerce')
            top10_df_clean = top10_df.dropna(subset=['predicted_cases']).copy()
            if not top10_df_clean.empty:
                top10_df_display = top10_df_clean.nlargest(10, 'predicted_cases')
                top10_df_display['incidence_rate'] = top10_df_display.apply(
                    lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
                    axis=1
                )
                # Add risk level
                top10_df_display['risk_level'] = pd.cut(
                    top10_df_display['incidence_rate'],
                    bins=[0, 10, 25, 50, float('inf')],
                    labels=['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Very High'],
                    include_lowest=True
                )
                display_columns = ['kabupaten_standard', 'predicted_cases', 'incidence_rate', 'risk_level']
                final_display_df = top10_df_display[display_columns].copy()
                final_display_df.rename(columns={
                    'kabupaten_standard': 'Region',
                    'predicted_cases': 'Predicted Cases',
                    'incidence_rate': 'Incidence Rate (/100k)',
                    'risk_level': 'Risk Level'
                }, inplace=True)
                # Format data
                final_display_df['Predicted Cases'] = final_display_df['Predicted Cases'].astype(int)
                final_display_df['Incidence Rate (/100k)'] = final_display_df['Incidence Rate (/100k)'].round(2)
                st.dataframe(
                    final_display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Predicted Cases": st.column_config.NumberColumn(
                            "Predicted Cases",
                            format="%d"
                        ),
                        "Incidence Rate (/100k)": st.column_config.NumberColumn(
                            "Incidence Rate (/100k)",
                            format="%.2f"
                        )
                    }
                )
            else:
                st.info("ℹ️ No valid data available for the Top 10 list for the selected date.")
        else:
            st.info("ℹ️ No data available for the selected date.")
else:
    st.error("❌ Data files could not be loaded. Please check deployment logs for more information.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>EWARS-ID Enhanced Dashboard</strong></p>
        <p>Developed by M Arief Widagdo • Enhanced with PAST weather integration and advanced analytics</p>
        <p>🦟 Dengue Fever Early Warning and Response System for Indonesia 🇮🇩</p>
    </div>
    """,
    unsafe_allow_html=True
)
