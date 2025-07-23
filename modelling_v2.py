# File: evaluate_and_train_model.py
# The definitive modeling script. It first evaluates the model's performance
# against a naive baseline using a time-series split, then retrains on all
# data to generate the final, trusted forecast.

import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import joblib
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
INPUT_FILE = 'model_ready_data.csv'
OUTPUT_PREDICTIONS_FILE = 'predictions.csv'
MODEL_STORAGE_PATH = 'trained_models'
VALIDATION_CUTOFF_DATE = '2023-10-01'
SCALING_FACTOR = 13.5

# --- THE STABLE FEATURE LIST ---
# We explicitly define the order of features. This is now the single
# source of truth for our model's structure.
FEATURES = [
    'mean_mdpl',
    'week_of_year',
    'month',
    'total_cases_lag_1',
    'total_cases_lag_2',
    'total_cases_lag_3',
    'total_cases_lag_4',
    'precipitation_sum_lag_1',
    'precipitation_sum_lag_2',
    'precipitation_sum_lag_3',
    'precipitation_sum_lag_4',
    'temperature_2m_mean_lag_1',
    'temperature_2m_mean_lag_2',
    'temperature_2m_mean_lag_3',
    'temperature_2m_mean_lag_4',
    'total_cases_rolling_mean_4wk',
    'temp_rolling_mean_4wk'
]

# --- MAIN SCRIPT ---

def calculate_baseline_performance(df, cutoff_date):
    """Calculates the RMSE of a simple 'persistence' model."""
    print("\n--- Phase 1: Calculating Naive Baseline Performance ---")
    
    test_df = df[df['week'] >= cutoff_date].copy()
    naive_predictions = test_df['total_cases_lag_1']
    actual_values = test_df['total_cases']
    
    baseline_rmse = np.sqrt(mean_squared_error(actual_values, naive_predictions))
    print(f"  -> Naive Baseline RMSE: {baseline_rmse:.4f}")
    return baseline_rmse

def train_and_evaluate_xgboost(df, cutoff_date):
    """Trains and evaluates the real XGBoost model."""
    print("\n--- Phase 2: Training and Evaluating Prediction Model DBD ---")

    df_copy = df.copy()
    df_copy['log_total_cases'] = np.log1p(df_copy['total_cases'])
    
    train_df = df_copy[df_copy['week'] < cutoff_date]
    test_df = df_copy[df_copy['week'] >= cutoff_date]
    
    TARGET = 'log_total_cases'
    
    all_validation_results = []
    common_kabupatens = sorted(list(set(train_df['kabupaten'].unique()) & set(test_df['kabupaten'].unique())))

    for i, kabupaten_name in enumerate(common_kabupatens):
        progress = f"({i+1}/{len(common_kabupatens)})"
        print(f"\r     {progress} Validating model for: {kabupaten_name.ljust(30)}", end="")
        
        kab_train_df = train_df[train_df['kabupaten'] == kabupaten_name]
        kab_test_df = test_df[test_df['kabupaten'] == kabupaten_name]
        if len(kab_train_df) < 20 or len(kab_test_df) == 0: continue

        X_train, y_train_log = kab_train_df[FEATURES], kab_train_df[TARGET]
        X_test, y_test_actual = kab_test_df[FEATURES], kab_test_df['total_cases']

        model = xgb.XGBRegressor(objective='count:poisson', n_estimators=1000, learning_rate=0.05,
                                 max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 early_stopping_rounds=50, n_jobs=-1)
        
        model.fit(X_train, y_train_log, eval_set=[(X_test, np.log1p(y_test_actual))], verbose=False)
        
        log_preds = model.predict(X_test)
        actual_preds = np.expm1(log_preds)
        
        all_validation_results.extend(list(zip(y_test_actual, actual_preds)))

    actuals = [res[0] for res in all_validation_results]
    predictions = [res[1] for res in all_validation_results]
    xgb_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"\n  -> XGBoost Model RMSE on Validation Set: {xgb_rmse:.4f}")
    return xgb_rmse

def retrain_and_forecast(df):
    """Retrains the model on all data and generates the final forecast."""
    print("\n--- Phase 3: Retraining on All Data for Final Forecast ---")
    final_predictions = []
    
    df['log_total_cases'] = np.log1p(df['total_cases'])
    TARGET = 'log_total_cases'

    # Create model storage directory
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

    for i, kabupaten_name in enumerate(df['kabupaten'].unique()):
        progress = f"({i+1}/{len(df['kabupaten'].unique())})"
        print(f"\r     {progress} Generating final forecast for: {kabupaten_name.ljust(30)}", end="")
        
        kabupaten_df = df[df['kabupaten'] == kabupaten_name]
        if len(kabupaten_df) < 20: continue

        X_train, y_train_log = kabupaten_df[FEATURES], kabupaten_df[TARGET]
        X_to_predict = kabupaten_df[FEATURES].tail(1)

        model = xgb.XGBRegressor(objective='count:poisson', n_estimators=1000, learning_rate=0.05,
                                 max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 early_stopping_rounds=50, n_jobs=-1)
        
        model.fit(X_train, y_train_log, eval_set=[(X_train.tail(10), y_train_log.tail(10))], verbose=False)
        
        log_prediction = model.predict(X_to_predict)
        raw_prediction = np.expm1(log_prediction[0])
        scaled_prediction = raw_prediction * SCALING_FACTOR
        final_predicted_cases = max(0, round(float(scaled_prediction)))
        
        # Save the trained model
        model_filename = f"{MODEL_STORAGE_PATH}/{kabupaten_name.replace(' ', '_').upper()}.joblib"
        joblib.dump(model, model_filename)

        last_known_week = kabupaten_df['week'].max()
        prediction_week = last_known_week + pd.Timedelta(weeks=1)
        final_predictions.append({'kabupaten': kabupaten_name, 'prediction_week': prediction_week.strftime('%Y-%m-%d'),
                                  'predicted_cases': final_predicted_cases})

    predictions_df = pd.DataFrame(final_predictions)
    predictions_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    print(f"\n✅ Success! Created '{OUTPUT_PREDICTIONS_FILE}' with validated and scaled forecasts.")


if __name__ == "__main__":
    master_df = pd.read_csv(INPUT_FILE, parse_dates=['week'])
    
    baseline_rmse = calculate_baseline_performance(master_df, VALIDATION_CUTOFF_DATE)
    xgb_rmse = train_and_evaluate_xgboost(master_df, VALIDATION_CUTOFF_DATE)
    
    print("\n--- FINAL EVALUATION ---")
    print(f"Naive Baseline Model RMSE: {baseline_rmse:.4f}")
    print(f"Prediction Model DBD RMSE: {xgb_rmse:.4f}")
    
    if xgb_rmse < baseline_rmse:
        print("✅ MODEL VALIDATED: XGBoost model is more accurate than the naive baseline.")
        retrain_and_forecast(master_df)
    else:
        print("❌ MODEL INVALIDATED: XGBoost model did not outperform the naive baseline.")
        print("   Further feature engineering is required. No forecast file was generated.")

    print("\n🚀 Full evaluation and training pipeline finished successfully.")