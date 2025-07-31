# modelling_v2.py
# Use Randomg Forest and XGBoost for modelling
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def run_modeling_pipeline(data_file="dengue_features_engineered.csv", output_dir="Model_v2_Corrected"):
    """
    Final, corrected modeling pipeline that:
    1. Removes the target-leaking feature ('Incidence_Rate').
    2. Compares RF and XGBoost with Time Series CV.
    3. Evaluates with R², MAE, and RMSE.
    4. Saves the best model and a comprehensive report.
    """
    try:
        # --- 1. Setup Environment ---
        print(f"--- Starting Final Corrected Modeling Pipeline ---")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, "Model_Comparison_Report_Corrected.txt")
        with open(report_path, "w") as report:
            report.write("--- Model Comparison and Evaluation Report (Corrected for Target Leak) ---\n\n")

            # --- 2. Load and Prepare Data ---
            print(f"Step 1: Loading data from '{data_file}'...")
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            report.write(f"Loaded {len(df)} records from {data_file}\n")
            
            y = df['Cases']
            
            features_to_drop = ['Cases', 'Date', 'Province', 'Kabupaten_Standard', 'BPS_Code', 'Incidence_Rate']
            X = df.drop(columns=features_to_drop)
            
            print(f"Features used for training: {len(X.columns)} features")

            # --- 3. Time-Based Split ---
            print("Step 2: Performing time-based train-test split...")
            unique_dates = np.sort(df['Date'].unique())
            split_point = int(len(unique_dates) * 0.7)
            split_date = unique_dates[split_point]
            
            train_indices = df[df['Date'] < split_date].index
            test_indices = df[df['Date'] >= split_date].index
            
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
            
            report.write(f"Data split on date: {pd.to_datetime(split_date).date()}\n")
            report.write(f"Training set size: {len(X_train)} records\n")
            report.write(f"Test set size: {len(X_test)} records\n\n")

            # --- 4. Time Series Cross-Validation Setup ---
            tscv = TimeSeriesSplit(n_splits=5)

            # --- 5. Model Training and Tuning ---
            print("Step 3: Training and tuning models...")
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42)
            }
            
            param_grid = {
                "RandomForest": {'n_estimators': [100, 200], 'max_depth': [15, 20], 'min_samples_leaf': [5]},
                "XGBoost": {'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.1]}
            }

            best_estimators = {}
            for name, model in models.items():
                print(f"--- Tuning {name}... ---")
                report.write(f"--- Results for {name} ---\n")
                
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                
                best_estimators[name] = grid_search.best_estimator_
                
                report.write(f"Best Parameters found: {grid_search.best_params_}\n")
                report.write(f"Best Cross-Validation MAE: {-grid_search.best_score_:.3f}\n\n")

            # --- 6. Final Evaluation on Hold-Out Test Set ---
            print("\nStep 4: Evaluating best models on the unseen test set...")
            report.write("--- Final Evaluation on Test Set ---\n")
            
            results = {}
            for name, model in best_estimators.items():
                print(f"Evaluating best {name}...")
                predictions = model.predict(X_test)
                
                # --- RMSE CALCULATION ADDED HERE ---
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse) # Take the square root of MSE
                r2 = r2_score(y_test, predictions)
                
                results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
                
                report.write(f"\nModel: {name}\n")
                report.write(f"  R-squared (R²): {r2:.3f}\n")
                report.write(f"  Mean Absolute Error (MAE): {mae:.3f}\n")
                report.write(f"  Root Mean Squared Error (RMSE): {rmse:.3f}\n")

            # --- 7. Determine the Best Model and Save ---
            best_model_name = min(results, key=lambda k: results[k]['MAE']) # Still choosing best based on MAE
            best_model = best_estimators[best_model_name]
            
            conclusion = (f"\n--- Conclusion ---\n"
                          f"The best performing model is '{best_model_name}' with a "
                          f"Mean Absolute Error of {results[best_model_name]['MAE']:.3f} on the final test set.\n"
                          f"This model is now validated and saved.\n")
            print(conclusion)
            report.write(conclusion)
            
            if hasattr(best_model, 'feature_importances_'):
                report.write(f"\nTop 10 Features for Best Model ({best_model_name}):\n")
                feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                report.write(feature_importance.head(10).to_string() + "\n")
            
            model_filename = os.path.join(output_dir, "best_dengue_model_corrected.joblib")
            columns_filename = os.path.join(output_dir, "model_columns_corrected.joblib")
            
            print(f"Saving the best model ({best_model_name}) to '{model_filename}'...")
            joblib.dump(best_model, model_filename)
            joblib.dump(X.columns.tolist(), columns_filename)

        print(f"\n--- PIPELINE COMPLETE ---")
        print(f"All outputs are in the '{output_dir}/' directory.")


    except Exception as e:
        print(f"\nAn unexpected error occurred.")
        traceback.print_exc()

if __name__ == "__main__":
    run_modeling_pipeline()

#--- Final Evaluation on Test Set ---

#Model: RandomForest
    # R-squared (R²): 0.866
    # Mean Absolute Error (MAE): 2.630
    # Root Mean Squared Error (RMSE): 17.536

#Model: XGBoost
    # R-squared (R²): 0.929
    # Mean Absolute Error (MAE): 2.062
    # Root Mean Squared Error (RMSE): 17.714
