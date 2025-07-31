# make_predictions_final.py
import pandas as pd
import joblib
import os
import traceback

def generate_future_features(historical_df, prediction_date_str):
    """
    Constructs the feature set for a future month using a robust mapping method
    that is immune to duplicate labels.
    """
    print(f"--- Engineering features for {prediction_date_str}... ---")
    prediction_date = pd.to_datetime(prediction_date_str)
    
    # --- 1. Get unique, most recent data for each district to build our future DataFrame ---
    # This also serves as our primary source for lag 1 data.
    df_lag1 = historical_df.sort_values('Date').drop_duplicates('Kabupaten_Standard', keep='last')
    
    # Create a clean list of all districts we need to predict for
    all_districts = df_lag1[['Kabupaten_Standard', 'Population']].copy()
    
    # --- 2. Create mapping dictionaries from historical data ---
    # This is a more robust way than setting a potentially duplicated index.
    
    # Lag 1 Mappers
    cases_lag1_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=1))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['Cases']
    inc_rate_lag1_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=1))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['Incidence_Rate_Lag1']
    oni_lag1_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=1))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['ONI_Value']
    precip_lag1_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=1))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['precipitation_sum']
    temp_lag1_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=1))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['temperature_2m_mean']
    humid_lag1_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=1))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['relative_humidity_2m_mean']

    # Lag 2 Mappers
    cases_lag2_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=2))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['Cases']
    oni_lag2_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=2))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['ONI_Value']
    precip_lag2_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=2))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['precipitation_sum']
    temp_lag2_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=2))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['temperature_2m_mean']
    humid_lag2_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=2))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['relative_humidity_2m_mean']

    # Lag 3 Mappers (for rolling windows)
    cases_lag3_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=3))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['Cases']
    precip_lag3_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=3))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['precipitation_sum']
    temp_lag3_map = historical_df[historical_df['Date'] == (prediction_date - pd.DateOffset(months=3))].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['temperature_2m_mean']

    # Historical Mappers (for weather assumption)
    hist_date = prediction_date - pd.DateOffset(years=1)
    hist_oni_map = historical_df[historical_df['Date'] == hist_date].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['ONI_Value']
    hist_precip_map = historical_df[historical_df['Date'] == hist_date].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['precipitation_sum']
    hist_temp_map = historical_df[historical_df['Date'] == hist_date].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['temperature_2m_mean']
    hist_humid_map = historical_df[historical_df['Date'] == hist_date].drop_duplicates('Kabupaten_Standard').set_index('Kabupaten_Standard')['relative_humidity_2m_mean']

    # --- 3. Build the future DataFrame using the mappers ---
    X_future = all_districts.copy()
    X_future['Month'] = prediction_date.month
    X_future['Year'] = prediction_date.year

    # Map all features
    X_future['Cases_Lag1'] = X_future['Kabupaten_Standard'].map(cases_lag1_map)
    X_future['Cases_Lag2'] = X_future['Kabupaten_Standard'].map(cases_lag2_map)
    X_future['Incidence_Rate_Lag1'] = X_future['Kabupaten_Standard'].map(inc_rate_lag1_map)
    X_future['ONI_Value_Lag1'] = X_future['Kabupaten_Standard'].map(oni_lag1_map)
    X_future['ONI_Value_Lag2'] = X_future['Kabupaten_Standard'].map(oni_lag2_map)
    X_future['precipitation_sum_Lag1'] = X_future['Kabupaten_Standard'].map(precip_lag1_map)
    X_future['precipitation_sum_Lag2'] = X_future['Kabupaten_Standard'].map(precip_lag2_map)
    X_future['temperature_2m_mean_Lag1'] = X_future['Kabupaten_Standard'].map(temp_lag1_map)
    X_future['temperature_2m_mean_Lag2'] = X_future['Kabupaten_Standard'].map(temp_lag2_map)
    X_future['relative_humidity_2m_mean_Lag1'] = X_future['Kabupaten_Standard'].map(humid_lag1_map)
    X_future['relative_humidity_2m_mean_Lag2'] = X_future['Kabupaten_Standard'].map(humid_lag2_map)

    # Calculate rolling features
    cases_lag3 = X_future['Kabupaten_Standard'].map(cases_lag3_map)
    X_future['Cases_Roll_Mean_3M'] = (X_future['Cases_Lag1'] + X_future['Cases_Lag2'] + cases_lag3) / 3
    
    precip_lag3 = X_future['Kabupaten_Standard'].map(precip_lag3_map)
    X_future['Precipitation_Roll_Sum_3M'] = X_future['precipitation_sum_Lag1'] + X_future['precipitation_sum_Lag2'] + precip_lag3
    
    temp_lag3 = X_future['Kabupaten_Standard'].map(temp_lag3_map)
    X_future['Temp_Roll_Mean_3M'] = (X_future['temperature_2m_mean_Lag1'] + X_future['temperature_2m_mean_Lag2'] + temp_lag3) / 3

    # Apply historical assumption
    X_future['ONI_Value'] = X_future['Kabupaten_Standard'].map(hist_oni_map)
    X_future['precipitation_sum'] = X_future['Kabupaten_Standard'].map(hist_precip_map)
    X_future['temperature_2m_mean'] = X_future['Kabupaten_Standard'].map(hist_temp_map)
    X_future['relative_humidity_2m_mean'] = X_future['Kabupaten_Standard'].map(hist_humid_map)

    # Fill any NaNs that might arise from districts not existing in a specific historical month
    X_future = X_future.fillna(method='ffill').fillna(method='bfill')

    print("--- Feature engineering for future date complete. ---")
    return X_future


def main():
    try:
        output_dir = "Prediction"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model_path = os.path.join("Model_v2_Corrected", "best_dengue_model_corrected.joblib")
        columns_path = os.path.join("Model_v2_Corrected", "model_columns_corrected.joblib")
        data_path = "dengue_features_engineered.csv"
        
        print("--- Loading model, column list, and historical data... ---")
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        historical_df = pd.read_csv(data_path, parse_dates=['Date'])
        
        prediction_date = '2024-01-01'
        X_future_full = generate_future_features(historical_df, prediction_date)
        X_future = X_future_full[model_columns]

        print("--- Making predictions with the loaded model... ---")
        predictions = model.predict(X_future)
        
        print("--- Formatting output and saving to CSV... ---")
        results_df = X_future_full[['Kabupaten_Standard', 'Population']].copy()
        results_df['Date'] = pd.to_datetime(prediction_date)
        
        latest_info = historical_df.sort_values('Date').drop_duplicates('Kabupaten_Standard', keep='last')[['Kabupaten_Standard', 'Province', 'BPS_Code']]
        results_df = pd.merge(results_df, latest_info, on='Kabupaten_Standard', how='left')
        
        results_df['Predicted_Cases'] = predictions
        results_df['Predicted_Cases'] = results_df['Predicted_Cases'].clip(lower=0).round().astype(int)
        
        results_df['Cases_Lag1'] = X_future['Cases_Lag1'].round().astype(int)
        results_df['Cases_Roll_Mean_3M'] = X_future['Cases_Roll_Mean_3M'].round(1)
        
        output_cols = ['Date', 'Province', 'Kabupaten_Standard', 'BPS_Code', 'Population', 'Predicted_Cases', 'Cases_Lag1', 'Cases_Roll_Mean_3M']
        results_df = results_df[output_cols]
        
        output_filename = os.path.join(output_dir, "january_2024_predictions.csv")
        results_df.to_csv(output_filename, index=False)
        
        print(f"\n--- PREDICTION COMPLETE ---")
        print(f"Predictions for {pd.to_datetime(prediction_date).strftime('%B %Y')} saved to '{output_filename}'")
        print("\nFirst 5 predictions:")
        print(results_df.head().to_string())

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Please ensure the following exist:")
        print(f"- '{data_path}'")
        print(f"- '{model_path}'")
        print(f"- '{columns_path}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the prediction process.")
        traceback.print_exc()

if __name__ == "__main__":
    main()
