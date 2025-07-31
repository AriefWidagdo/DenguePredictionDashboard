# File: streamlit_app.py
# Enhanced EWARS-ID system with improved features and weather integration
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
    st.session_state.weather_data = None

# --- Helper Function for Name Standardization ---
@st.cache_data
def standardize_name(name):
    """Creates a reliable, simple key for merging by standardizing names."""
    if not isinstance(name, str):
        return None
    name_lower = name.lower().strip()
    prefixes = ['kabupaten administrasi', 'kota administrasi', 'kabupaten', 'kota', 'kab.']
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            name_lower = name_lower.replace(prefix, '', 1).strip()
    return re.sub(r'[^a-z0-9]', '', name_lower)

# --- Weather Data Function ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_data():
    """Fetch weather data for major Indonesian cities."""
    try:
        # You'll need to get a free API key from OpenWeatherMap
        api_key = st.secrets.get("WEATHER_API_KEY", "")
        if not api_key:
            return None
            
        cities = [
            {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456},
            {"name": "Surabaya", "lat": -7.2575, "lon": 112.7521},
            {"name": "Medan", "lat": 3.5952, "lon": 98.6722},
            {"name": "Bandung", "lat": -6.9175, "lon": 107.6191},
            {"name": "Makassar", "lat": -5.1477, "lon": 119.4327}
        ]
        
        weather_data = []
        for city in cities:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={city['lat']}&lon={city['lon']}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_data.append({
                    'city': city['name'],
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description']
                })
        return pd.DataFrame(weather_data) if weather_data else None
    except Exception as e:
        st.warning(f"Weather data unavailable: {str(e)}")
        return None

# --- Data Loading Function ---
@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """Loads and prepares all data from GitHub repo."""
    try:
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
        
        # Create merge keys
        forecast_df['merge_key'] = forecast_df['kabupaten_standard'].apply(standardize_name)
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'].apply(standardize_name)
        
        # Merge datasets
        merged_gdf = kabupaten_gdf.merge(forecast_df, on='merge_key', how='left')
        return merged_gdf, forecast_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def create_enhanced_map(gdf):
    """Creates an enhanced Folium map with better styling and features."""
    map_gdf = gdf.copy()
    
    # Handle date conversion
    if 'Date' in map_gdf.columns and pd.api.types.is_datetime64_any_dtype(map_gdf['Date']):
        map_gdf['forecast_week_str'] = map_gdf['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
        map_gdf = map_gdf.drop(columns=['Date'])
    else:
        map_gdf['forecast_week_str'] = 'N/A'
    
    # Ensure numeric data
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
    
    # Initialize map
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")
    
    # Add choropleth layer
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
    
    # Enhanced popups with risk level
    map_gdf['popup_html'] = map_gdf.apply(
        lambda row: f"""
        <div style="font-family: sans-serif; min-width: 200px;">
            <h4 style="margin-bottom: 10px;">📍 {row.get('kabupaten_standard', row['NAME_2'])}</h4>
            <div style="border-left: 4px solid #ff6b6b; padding-left: 10px;">
                <p><strong>Forecast Week:</strong> {row['forecast_week_str']}</p>
                <p><strong>Predicted Cases:</strong> {int(row['predicted_cases_numeric'])}</p>
                <p><strong>Population:</strong> {int(row['population_numeric']) if not pd.isna(row['population_numeric']) else 'N/A'}</p>
                <p><strong>Incidence Rate:</strong> {row['incidence_rate']:.2f}/100k</p>
                <p><strong>Risk Level:</strong> <span style="color: {'red' if row['risk_level'] in ['High', 'Very High'] else 'orange' if row['risk_level'] == 'Medium' else 'green'}">{row['risk_level']}</span></p>
            </div>
        </div>
        """, axis=1
    )
    
    folium.GeoJson(
        map_gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
        tooltip=folium.features.GeoJsonTooltip(fields=['NAME_2'], aliases=['Region:']),
        popup=folium.features.GeoJsonPopup(fields=['popup_html'], aliases=[''])
    ).add_to(m)
    
    return m

def create_summary_metrics(data, selected_date):
    """Create summary metrics for the selected date."""
    if data is None or selected_date is None:
        return None, None, None, None
    
    # Filter data for selected date and valid predictions
    summary_data = data[
        (data['Date'] == selected_date) & 
        (data['predicted_cases'].notna()) &
        (data['population'].notna())
    ].copy()
    
    if summary_data.empty:
        return 0, 0, 0, 0
    
    # Convert to numeric
    summary_data['predicted_cases'] = pd.to_numeric(summary_data['predicted_cases'], errors='coerce')
    summary_data['population'] = pd.to_numeric(summary_data['population'], errors='coerce')
    
    # Calculate metrics
    total_population = summary_data['population'].sum()
    total_predicted_cases = summary_data['predicted_cases'].sum()
    regions_with_data = len(summary_data)
    avg_incidence_rate = (total_predicted_cases / total_population * 100000) if total_population > 0 else 0
    
    return total_population, total_predicted_cases, avg_incidence_rate, regions_with_data

def create_trend_chart(forecast_data):
    """Create a trend chart showing national forecast over time."""
    if forecast_data is None or forecast_data.empty:
        return None
    
    # Aggregate by date
    trend_data = forecast_data.groupby('Date').agg({
        'predicted_cases': 'sum',
        'population': 'sum'
    }).reset_index()
    
    trend_data['incidence_rate'] = (trend_data['predicted_cases'] / trend_data['population']) * 100000
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['predicted_cases'],
        mode='lines+markers',
        name='Total Predicted Cases',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='National Dengue Forecast Trend',
        xaxis_title='Date',
        yaxis_title='Total Predicted Cases',
        template='plotly_white',
        height=400
    )
    
    return fig

# --- Main App Layout ---
st.title("🇮🇩 EWARS-ID: Enhanced Dengue Forecast Dashboard")
st.markdown("An operational prototype for near real-time dengue fever forecasting by M Arief Widagdo.")

# Configuration
OWNER = "AriefWidagdo"
PRIVATE_REPO_NAME = "data-raw"
FORECAST_FILE_PATH = "january_2024_predictions.csv"
GEOJSON_PATH = "gadm41_IDN_2.json"

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # Load weather data
    if st.button("🌤️ Refresh Weather Data"):
        st.session_state.weather_data = get_weather_data()
    
    # Display weather if available
    if st.session_state.weather_data is not None:
        st.subheader("Current Weather")
        weather_df = st.session_state.weather_data
        for _, row in weather_df.iterrows():
            st.metric(
                label=f"{row['city']}",
                value=f"{row['temperature']:.1f}°C",
                delta=f"{row['humidity']}% humidity"
            )
    elif st.session_state.weather_data is None and 'WEATHER_API_KEY' not in st.secrets:
        st.info("Add WEATHER_API_KEY to secrets to enable weather data")

# --- Load Data ---
if st.session_state.loaded_data is None or st.session_state.loaded_forecast is None:
    with st.spinner("Loading data from GitHub..."):
        merged_data, forecast_data = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)
        st.session_state.loaded_data = merged_data
        st.session_state.loaded_forecast = forecast_data
else:
    merged_data = st.session_state.loaded_data
    forecast_data = st.session_state.loaded_forecast

if merged_data is not None and forecast_data is not None:
    # Date selection
    unique_dates = sorted(forecast_data['Date'].unique())
    if unique_dates:
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = unique_dates[0]
        
        selected_date = st.selectbox(
            "📅 Select Forecast Date:",
            unique_dates,
            index=unique_dates.index(st.session_state.selected_date) if st.session_state.selected_date in unique_dates else 0,
            format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d (%A)')
        )
        
        if selected_date != st.session_state.selected_date:
            st.session_state.selected_date = selected_date
    
    # Filter data for selected date
    map_ready_gdf = merged_data[
        (merged_data['Date'] == selected_date) | (pd.isnull(merged_data['Date']))
    ].copy()
    
    # --- Enhanced Summary Section ---
    st.subheader(f"📊 National Summary for {selected_date.strftime('%Y-%m-%d')}")
    
    total_pop, total_cases, avg_incidence, regions_count = create_summary_metrics(
        forecast_data, selected_date
    )
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏘️ Total Population",
            value=f"{total_pop:,.0f}",
            help="Total population in regions with forecast data"
        )
    
    with col2:
        st.metric(
            label="🦟 Predicted Cases",
            value=f"{total_cases:,.0f}",
            help="Total predicted dengue cases for next week"
        )
    
    with col3:
        st.metric(
            label="📈 Avg Incidence Rate",
            value=f"{avg_incidence:.2f}/100k",
            help="Weighted average incidence rate per 100,000 population"
        )
    
    with col4:
        st.metric(
            label="🗺️ Regions Covered",
            value=f"{regions_count}",
            help="Number of regions with forecast data"
        )
    
    # Add risk level distribution
    if not forecast_data.empty:
        risk_data = forecast_data[forecast_data['Date'] == selected_date].copy()
        if not risk_data.empty:
            risk_data['predicted_cases'] = pd.to_numeric(risk_data['predicted_cases'], errors='coerce')
            risk_data['population'] = pd.to_numeric(risk_data['population'], errors='coerce')
            risk_data['incidence_rate'] = (risk_data['predicted_cases'] / risk_data['population']) * 100000
            
            risk_data['risk_level'] = pd.cut(
                risk_data['incidence_rate'],
                bins=[0, 10, 25, 50, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
            
            risk_counts = risk_data['risk_level'].value_counts()
            
            st.subheader("🚨 Risk Level Distribution")
            risk_cols = st.columns(len(risk_counts))
            colors = ['🟢', '🟡', '🟠', '🔴']
            
            for i, (level, count) in enumerate(risk_counts.items()):
                if i < len(risk_cols):
                    risk_cols[i].metric(
                        label=f"{colors[i]} {level} Risk",
                        value=f"{count} regions"
                    )
    
    # --- National Trend Chart ---
    st.subheader("📈 National Forecast Trend")
    trend_fig = create_trend_chart(forecast_data)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)
    
    # --- Enhanced Map Section ---
    st.subheader(f"🗺️ Interactive Risk Map")
    map_object = create_enhanced_map(map_ready_gdf)
    st_folium(
        map_object,
        width='100%',
        height=600,
        key=f"enhanced_map_{selected_date.strftime('%Y%m%d')}",
        returned_objects=[]
    )
    
    # --- Sidebar Region Analysis ---
    with st.sidebar:
        st.header("🔍 Region Analysis")
        kabupaten_list = sorted(forecast_data['kabupaten_standard'].unique())
        
        if 'selected_kabupaten' not in st.session_state:
            st.session_state.selected_kabupaten = kabupaten_list[0] if kabupaten_list else None
        
        selected_kabupaten = st.selectbox(
            "Select Region:",
            kabupaten_list,
            index=kabupaten_list.index(st.session_state.selected_kabupaten) if st.session_state.selected_kabupaten in kabupaten_list else 0
        )
        
        if selected_kabupaten:
            trajectory_df = forecast_data[forecast_data['kabupaten_standard'] == selected_kabupaten].copy()
            
            if not trajectory_df.empty:
                st.subheader(f"📊 {selected_kabupaten}")
                
                # Create mini chart
                trajectory_df['predicted_cases'] = pd.to_numeric(trajectory_df['predicted_cases'], errors='coerce')
                population = pd.to_numeric(trajectory_df['population'].iloc[0], errors='coerce')
                
                if pd.notna(population) and population > 0:
                    trajectory_df['incidence_rate'] = (trajectory_df['predicted_cases'] / population) * 100000
                else:
                    trajectory_df['incidence_rate'] = 0
                
                # Mini metrics
                current_week_data = trajectory_df[trajectory_df['Date'] == selected_date]
                if not current_week_data.empty:
                    current_cases = current_week_data['predicted_cases'].iloc[0]
                    current_incidence = current_week_data['incidence_rate'].iloc[0]
                    
                    st.metric("Cases This Week", f"{current_cases:.0f}")
                    st.metric("Incidence Rate", f"{current_incidence:.2f}/100k")
                
                # Line chart
                chart_data = trajectory_df[['Date', 'predicted_cases']].set_index('Date')
                st.line_chart(chart_data, height=200)
                
                # Data table
                display_df = trajectory_df[['Date', 'predicted_cases', 'incidence_rate']].copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df.columns = ['Week', 'Cases', 'Rate/100k']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # --- Top Regions Table ---
    st.subheader(f"🏆 Top 10 Highest Risk Regions")
    
    top_regions_data = forecast_data[forecast_data['Date'] == selected_date].copy()
    if not top_regions_data.empty:
        top_regions_data['predicted_cases'] = pd.to_numeric(top_regions_data['predicted_cases'], errors='coerce')
        top_regions_data['population'] = pd.to_numeric(top_regions_data['population'], errors='coerce')
        top_regions_data['incidence_rate'] = (top_regions_data['predicted_cases'] / top_regions_data['population']) * 100000
        
        top_regions_data = top_regions_data.dropna(subset=['predicted_cases']).nlargest(10, 'predicted_cases')
        
        # Add risk level
        top_regions_data['risk_level'] = pd.cut(
            top_regions_data['incidence_rate'],
            bins=[0, 10, 25, 50, float('inf')],
            labels=['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Very High'],
            include_lowest=True
        )
        
        display_columns = ['kabupaten_standard', 'predicted_cases', 'incidence_rate', 'risk_level']
        display_df = top_regions_data[display_columns].copy()
        display_df.columns = ['Region', 'Predicted Cases', 'Incidence Rate (/100k)', 'Risk Level']
        display_df['Predicted Cases'] = display_df['Predicted Cases'].astype(int)
        display_df['Incidence Rate (/100k)'] = display_df['Incidence Rate (/100k)'].round(2)
        
        st.dataframe(
            display_df,
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
        st.info("No data available for the selected date.")

else:
    st.error("❌ Unable to load data. Please check the configuration and try again.")

# --- Footer ---
st.markdown("---")
st.markdown("*EWARS-ID Dashboard - Enhanced version with weather integration and improved analytics*")
