# File: app.py
# The "Face" of the EWARS-ID system - Final Production Version
# Data Loading Update: Now fetches data from a private GitHub repository.
# Updated to use january_2024_predictions.csv and show Population & Incidence Rate.
# Uses RapidFuzz for matching Kabupaten_Standard to NAME_2.

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests 
import io       
# --- Import RapidFuzz ---
from rapidfuzz import process, fuzz

# --- Page Configuration ---
st.set_page_config(
    page_title="EWARS-ID Dengue Forecast by M Arief Widagdo",
    page_icon="🇮🇩",
    layout="wide"
)

# --- Caching Functions ---
@st.cache_data
def load_data(owner, repo, forecast_path, geojson_path):
    """
    Loads and prepares all data from a private GitHub repo,
    returning a map-ready GDF and a full forecast DF.
    Uses RapidFuzz to match kabupaten names.
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
        
        # Download geojson data
        geojson_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{geojson_path}"
        geojson_response = requests.get(geojson_url, headers=headers)
        geojson_response.raise_for_status()

        # --- Read the downloaded content ---
        # --- UPDATED: Read with new column names and parse Date ---
        forecast_df = pd.read_csv(
            io.StringIO(forecast_response.text), # Use text for CSV
            parse_dates=['Date'],
            date_parser=pd.to_datetime # Explicitly parse dates
        )
        # Rename columns to match internal logic or for clarity
        forecast_df.rename(columns={
            'Kabupaten_Standard': 'kabupaten',
            'Predicted_Cases': 'predicted_cases',
            'Population': 'population'
        }, inplace=True)
        
        kabupaten_gdf = gpd.read_file(io.BytesIO(geojson_response.content))
        
        # --- Processing logic with Fuzzy Matching ---
        
        # Get unique lists of names
        csv_kabupaten_names = forecast_df['kabupaten'].unique().tolist()
        geojson_kabupaten_names = kabupaten_gdf['NAME_2'].unique().tolist()

        # Create a mapping dictionary using RapidFuzz
        name_mapping = {}
        unmatched_from_csv = []
        threshold = 90 # Minimum similarity score to consider a match

        for csv_name in csv_kabupaten_names:
            # Find the best match in the GeoJSON names
            best_match, score, _ = process.extractOne(csv_name, geojson_kabupaten_names, scorer=fuzz.WRatio)
            if score >= threshold:
                name_mapping[csv_name] = best_match
            else:
                unmatched_from_csv.append(csv_name)
                # Optionally, map unmatched names to None or a placeholder
                name_mapping[csv_name] = None 

        # Report unmatched names
        if unmatched_from_csv:
            st.warning(f"Could not find high-confidence matches for {len(unmatched_from_csv)} kabupaten/kota names from the CSV using fuzzy matching (threshold={threshold}). They will not appear on the map.")
            # Optionally, display the list in an expander for debugging
            # with st.expander("Unmatched CSV Names"):
            #     st.write(unmatched_from_csv)

        # Add the matched GeoJSON name to the forecast dataframe
        forecast_df['matched_name_2'] = forecast_df['kabupaten'].map(name_mapping)

        # Filter out rows where matching failed
        forecast_df_matched = forecast_df.dropna(subset=['matched_name_2']).copy()

        # Sort by kabupaten and Date
        forecast_df_matched = forecast_df_matched.sort_values(by=['kabupaten', 'Date'])
        # Take the first forecast date for each kabupaten for the map (assumes first is the primary forecast week)
        first_week_df = forecast_df_matched.loc[forecast_df_matched.groupby('kabupaten')['Date'].idxmin()].copy()
        # Add a column for the forecast week string for display
        first_week_df['forecast_week_str'] = first_week_df['Date'].dt.strftime('%Y-%m-%d')

        # Prepare GeoDataFrame for merging
        # Use the matched NAME_2 from the CSV as the key
        first_week_df['merge_key'] = first_week_df['matched_name_2'] 
        kabupaten_gdf['merge_key'] = kabupaten_gdf['NAME_2'] # Use NAME_2 directly from GeoJSON
        
        merged_gdf = kabupaten_gdf.merge(first_week_df, on='merge_key', how='left')
        
        merged_gdf['predicted_cases'] = merged_gdf['predicted_cases'].fillna(0)
        merged_gdf['population'] = merged_gdf['population'].fillna(0) # Fill population as well
        # kabupaten name can come from either source, prefer the original standardized one
        merged_gdf['kabupaten_display'] = merged_gdf['kabupaten'].fillna(merged_gdf['NAME_2']) 
        
        # --- UPDATED: Include 'population' and 'matched_name_2' in the map-ready GDF ---
        map_ready_gdf = merged_gdf[['geometry', 'merge_key', 'kabupaten_display', 'predicted_cases', 'population', 'NAME_2', 'forecast_week_str']].copy()
        map_ready_gdf.rename(columns={'kabupaten_display': 'kabupaten'}, inplace=True) # Rename for consistency downstream
        
        return map_ready_gdf, forecast_df_matched # Return the matched forecast data
        
    except requests.exceptions.RequestException as e:
        st.error(f"FATAL ERROR: Could not fetch data from GitHub. Check your token and repo details. Error: {e}")
        return None, None
    except KeyError as e:
        st.error(f"FATAL ERROR: Missing key in secrets or data. Error: {e}. Please add GITHUB_TOKEN to your Streamlit app settings or check CSV column names.")
        return None, None
    except Exception as e:
        st.error(f"FATAL ERROR: An error occurred during data processing: {e}")
        return None, None

#map function using Folium's Choropleth
def create_map(gdf):
    # --- UPDATED: Add Incidence Rate calculation (per 100,000 population) ---
    # Avoid division by zero
    gdf['incidence_rate'] = gdf.apply(
        lambda row: (row['predicted_cases'] / row['population']) * 100000 if row['population'] > 0 else 0,
        axis=1
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        # --- UPDATED: Use 'predicted_cases' for choropleth ---
        columns=['merge_key', 'predicted_cases'], 
        key_on='feature.properties.merge_key',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name='Predicted Dengue Cases (Next Week)',
        name='Predicted Cases'
    ).add_to(m)

    # --- UPDATED: Popup to include Population and Incidence Rate ---
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
FORECAST_FILE_PATH = "january_2024_predictions.csv"  # The path to the file inside the private repo
GEOJSON_PATH = "gadm41_IDN_2.json"         # The path to the file inside the private repo


map_data, full_forecast_df = load_data(OWNER, PRIVATE_REPO_NAME, FORECAST_FILE_PATH, GEOJSON_PATH)


if map_data is not None and full_forecast_df is not None:
    
    # --- Get the unique forecast dates available in the data ---
    unique_dates = sorted(full_forecast_df['Date'].unique())
    if unique_dates:
        selected_date = st.selectbox(
            "Select Forecast Date for Map View:",
            unique_dates,
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
        # Filter map data for the selected date
        # Use the full forecast data to get the correct data for the selected date
        map_data_for_date = full_forecast_df[full_forecast_df['Date'] == selected_date].copy()
        
        # Need to merge this filtered data with the GeoJSON GDF again for mapping
        # Re-run the matching logic or reuse the mapping if names are consistent per date
        # Simpler approach: Re-merge the filtered data with the base GeoJSON GDF
        kabupaten_gdf_base = map_data[['geometry', 'NAME_2']].drop_duplicates() # Get base GDF geometry and NAME_2
        # --- Fuzzy Matching for the selected date data ---
        csv_kabupaten_names_for_date = map_data_for_date['kabupaten'].unique().tolist()
        geojson_kabupaten_names_base = kabupaten_gdf_base['NAME_2'].unique().tolist()

        name_mapping_for_date = {}
        threshold = 90
        for csv_name in csv_kabupaten_names_for_date:
             # Use the already found match if available, otherwise re-match (less efficient but robust)
             # Let's assume name_mapping is not easily accessible here, so re-match
             # A more efficient way would be to pass the name_mapping from load_data or store it.
             # For simplicity, re-running matching here.
            best_match, score, _ = process.extractOne(csv_name, geojson_kabupaten_names_base, scorer=fuzz.WRatio)
            if score >= threshold:
                name_mapping_for_date[csv_name] = best_match
            else:
                 # Handle unmatched - perhaps skip or use original (risk of mismatch)
                 # Let's skip for map view if no good match
                 name_mapping_for_date[csv_name] = None

        map_data_for_date['matched_name_2'] = map_data_for_date['kabupaten'].map(name_mapping_for_date)
        map_data_for_date_filtered = map_data_for_date.dropna(subset=['matched_name_2'])

        # Merge with base GDF
        map_data_for_date_filtered['merge_key'] = map_data_for_date_filtered['matched_name_2']
        kabupaten_gdf_base['merge_key'] = kabupaten_gdf_base['NAME_2']
        map_ready_gdf_for_date = kabupaten_gdf_base.merge(map_data_for_date_filtered, on='merge_key', how='left')
        map_ready_gdf_for_date['predicted_cases'] = map_ready_gdf_for_date['predicted_cases'].fillna(0)
        map_ready_gdf_for_date['population'] = map_ready_gdf_for_date['population'].fillna(0)
        map_ready_gdf_for_date['kabupaten_display'] = map_ready_gdf_for_date['kabupaten'].fillna(map_ready_gdf_for_date['NAME_2'])
        map_ready_gdf_for_date['forecast_week_str'] = selected_date.strftime('%Y-%m-%d')
        map_ready_gdf_for_date = map_ready_gdf_for_date[['geometry', 'merge_key', 'kabupaten_display', 'predicted_cases', 'population', 'NAME_2', 'forecast_week_str']].copy()
        map_ready_gdf_for_date.rename(columns={'kabupaten_display': 'kabupaten'}, inplace=True)

    else:
        st.warning("No forecast dates found in the data.")
        selected_date = None
        map_ready_gdf_for_date = map_data # Fallback to original map data

    # --- NEW LAYOUT: ALL CONTROLS AND CHARTS IN THE SIDEBAR ---
    st.sidebar.header("Forecast Trajectory")
    
    # Use kabupaten names from the full forecast data for the selector
    kabupaten_list = sorted(full_forecast_df['kabupaten'].unique())
    selected_kabupaten = st.sidebar.selectbox(
        "Select a Kabupaten/Kota to inspect its forecast:",
        kabupaten_list
    )
    
    # Move the 4-week forecast elements into the sidebar
    st.sidebar.subheader(f"4-Week Forecast for {selected_kabupaten}")
    
    trajectory_df = full_forecast_df[full_forecast_df['kabupaten'] == selected_kabupaten].copy()
    # Calculate incidence rate for the sidebar chart/dataframe
    if not trajectory_df.empty and 'population' in trajectory_df.columns:
        population_for_kab = trajectory_df['population'].iloc[0] # Assuming population is constant
        trajectory_df['incidence_rate'] = (trajectory_df['predicted_cases'] / population_for_kab) * 100000
    else:
        trajectory_df['incidence_rate'] = 0
    
    # Chart with dual axis or two charts might be complex, showing predicted cases
    chart_df = trajectory_df[['Date', 'predicted_cases']].set_index('Date')
    st.sidebar.line_chart(chart_df, y='predicted_cases')
    
    display_df = trajectory_df[['Date', 'predicted_cases', 'incidence_rate']].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d') # Format date for display
    display_df.rename(columns={'Date': 'Week', 'predicted_cases': 'Forecast Cases', 'incidence_rate': 'Incidence Rate (/100k)'}, inplace=True)
    st.sidebar.dataframe(display_df.set_index('Week'), use_container_width=True)


    # --- Main Map Section ---
    st.subheader(f"National Risk Overview (Forecast for {selected_date.strftime('%Y-%m-%d') if selected_date else 'Next Week'})")
    if selected_date and not map_ready_gdf_for_date.empty:
        map_object = create_map(map_ready_gdf_for_date)
    else:
        map_object = create_map(map_data) # Fallback
    st_folium(map_object, returned_objects=[], width='100%', height=550, key="overview_map")

    # --- Top 10 Regions Section ---
    if selected_date and not map_ready_gdf_for_date.empty:
        # Use the data for the selected date for top 10
        top10_df = map_ready_gdf_for_date.nlargest(10, 'predicted_cases')[['kabupaten', 'predicted_cases', 'population']]
        # Calculate incidence rate for top 10 as well
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
    st.error("Data files could not be loaded. Please check the logs for more information.")
