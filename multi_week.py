# File: generate_multi_week_forecast.py
# CORRECTED VERSION: This script fixes the critical autoregressive loop bug
# to ensure stable multi-step forecasting.

import pandas as pd
import numpy as np
import joblib
import time
import os

# --- CONFIGURATION ---
INPUT_FILE = 'model_ready_data.csv'
MODEL_STORAGE_PATH = 'trained_models'
OUTPUT_FORECAST_FILE = 'multi_week_forecast.csv'
FORECAST_HORIZON_WEEKS = 4
SCALING_FACTOR = 10.0

# --- THE STABLE FEATURE LIST (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
FEATURES = [
    'mean_mdpl', 'week_of_year', 'month',
    'total_cases_lag_1', 'total_cases_lag_2', 'total_cases_lag_3', 'total_cases_lag_4',
    'precipitation_sum_lag_1', 'precipitation_sum_lag_2', 'precipitation_sum_lag_3', 'precipitation_sum_lag_4',
    'temperature_2m_mean_lag_1', 'temperature_2m_mean_lag_2', 'temperature_2m_mean_lag_3', 'temperature_2m_mean_lag_4',
    'total_cases_rolling_mean_4wk', 'temp_rolling_mean_4wk'
]

# --- MAIN SCRIPT ---

def generate_multi_step_forecasts():
    print(f"--- Generating {FORECAST_HORIZON_WEEKS}-Week Autoregressive Forecast ---")
    start_time = time.time()
    
    print(f"  -> Loading master data from '{INPUT_FILE}' to get the last known state...")
    try:
        df = pd.read_csv(INPUT_FILE, parse_dates=['week'])
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Input file not found: '{INPUT_FILE}'. Please run the data preparation and training scripts first.")
        return

    all_forecasts = []
    all_kabupaten = sorted(df['kabupaten'].unique())
    
    print(f"  -> Preparing to forecast for {len(all_kabupaten)} locations...")

    for i, kabupaten_name in enumerate(all_kabupaten):
        progress = f"({i+1}/{len(all_kabupaten)})"
        print(f"\r     {progress} Forecasting for: {kabupaten_name.ljust(30)}", end="")
        
        model_filename = f"{MODEL_STORAGE_PATH}/{kabupaten_name.replace(' ', '_').upper()}.joblib"
        try:
            model = joblib.load(model_filename)
        except FileNotFoundError:
            continue

        last_known_row = df[df['kabupaten'] == kabupaten_name].sort_values('week').tail(1)
        if last_known_row.empty:
            continue
            
        current_features = last_known_row[FEATURES].copy()
        last_known_week = last_known_row['week'].iloc[0]

        # Autoregressive Loop
        for step in range(1, FORECAST_HORIZON_WEEKS + 1):
            # Predict the next step using the current feature set
            log_prediction = model.predict(current_features)
            
            # --- FIX 1 (OUTPUT): Inverse transform for the FINAL output only ---
            # Apply np.expm1 (exp(x) - 1) which is the inverse of np.log1p (log(x+1))
            final_predicted_cases = np.round(np.maximum(0, np.expm1(log_prediction[0]) * SCALING_FACTOR))

            # Store the final, human-readable forecast for this step
            forecast_week = last_known_week + pd.Timedelta(weeks=step)
            all_forecasts.append({
                'kabupaten': kabupaten_name,
                'forecast_week': forecast_week, # Keep as datetime object for now
                'predicted_cases': int(final_predicted_cases)
            })

            # --- Update Features for the NEXT step in the loop ---
            # Shift all lag features one step into the future
            for col in ['total_cases', 'precipitation_sum', 'temperature_2m_mean']:
                for lag in range(4, 1, -1):
                    current_features[f'{col}_lag_{lag}'] = current_features[f'{col}_lag_{lag-1}']
            
            # --- FIX 2 (INPUT): Feed the LOG-TRANSFORMED prediction back into the loop ---
            # This ensures the data scale remains consistent with the training data.
            current_features['total_cases_lag_1'] = log_prediction[0]
            
            # For this simplified model, we assume the last known weather persists through its lags.
            # A real-time system would update these with actual weather forecasts.
            current_features['precipitation_sum_lag_1'] = last_known_row['precipitation_sum'].iloc[0]
            current_features['temperature_2m_mean_lag_1'] = last_known_row['temperature_2m_mean'].iloc[0]
            
            # Update time-based features for the new week
            new_date = last_known_week + pd.Timedelta(weeks=step)
            current_features['week_of_year'] = new_date.isocalendar().week
            current_features['month'] = new_date.month

            # --- FIX 3: Recalculate rolling means with the CORRECTLY SCALED lag features ---
            current_features['total_cases_rolling_mean_4wk'] = (current_features['total_cases_lag_1'] + current_features['total_cases_lag_2'] + current_features['total_cases_lag_3'] + current_features['total_cases_lag_4']) / 4.0
            current_features['temp_rolling_mean_4wk'] = (current_features['temperature_2m_mean_lag_1'] + current_features['temperature_2m_mean_lag_2'] + current_features['temperature_2m_mean_lag_3'] + current_features['temperature_2m_mean_lag_4']) / 4.0

    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv(OUTPUT_FORECAST_FILE, index=False)

    end_time = time.time()
    print(f"\n\n--- Multi-Week Forecast Complete ---")
    print(f"✅ Success! Created '{OUTPUT_FORECAST_FILE}' with a {FORECAST_HORIZON_WEEKS}-week horizon.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    generate_multi_step_forecasts()