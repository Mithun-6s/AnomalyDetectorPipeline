import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
import os
import warnings
import google.generativeai as genai
import shap # Import SHAP

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
FILE_PATH = "SOL PIONEER_ManualTabsExport_Fri Sep 26 2025.xlsx"
HEADER_ROW_INDEX = 5
MODEL_DIR = "models"
SEQ_LEN = 3
GEMINI_API_KEY = "" # Replace with your actual key

# --- Model Definitions ---
HYDRO_MODEL_DEFINITIONS = {
    'DraughtMean_Predictor': {
        'target': 'Draught Mean [M]',
        'features': ['Cargo Weight [MT]', 'Ballast Water [MT]', 'Trim [M]'],
    },
    'CargoWeight_Predictor': {
        'target': 'Cargo Weight [MT]',
        'features': ['Draught Mean [M]', 'Ballast Water [MT]', 'Trim [M]'],
    },
    'BallastWater_Predictor': {
        'target': 'Ballast Water [MT]',
        'features': ['Draught Mean [M]', 'Cargo Weight [MT]', 'Trim [M]'],
    }
}

class VesselDataProcessor:
    """Handles loading, cleaning, and feature engineering of vessel data."""

    def __init__(self, file_path, header_row):
        self.file_path = file_path
        self.header_row = header_row
        self.raw_df = self._load_and_clean_data()

    def _load_and_clean_data(self):
        """Loads and performs initial cleaning on the raw excel data."""
        df = pd.read_excel(self.file_path, header=self.header_row)
        df.columns = df.columns.str.strip()
        df.replace(["", " "], np.nan, inplace=True)
        return df

    def _impute_hydro_columns(self, df):
        """Imputes missing values for hydrostatic model training."""
        df_imputed = df.copy()
        unnamed_cols = [c for c in df_imputed.columns if isinstance(c, str) and c.startswith('Unnamed:')]
        df_imputed.drop(columns=unnamed_cols, inplace=True, errors='ignore')

        for col in df_imputed.columns:
            if df_imputed[col].dtype == 'object':
                df_imputed[col].fillna("Missing", inplace=True)
            elif pd.api.types.is_numeric_dtype(df_imputed[col]):
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val if pd.notna(median_val) else 0, inplace=True)
        return df_imputed

    def _calculate_derived_hydro_features(self, df):
        """Calculates derived features for hydrostatic models."""
        df_derived = df.copy()
        num_cols = ['Draught Aft [M]', 'Draught Fore [M]']
        for col in num_cols:
            df_derived[col] = pd.to_numeric(df_derived[col], errors='coerce')
        df_derived.dropna(subset=num_cols, inplace=True)

        df_derived['Trim [M]'] = df_derived['Draught Aft [M]'] - df_derived['Draught Fore [M]']
        df_derived['Draught Mean [M]'] = (df_derived['Draught Fore [M]'] + df_derived['Draught Aft [M]']) / 2
        return df_derived

    @staticmethod
    def _calculate_heading(lat1, lon1, lat2, lon2):
        """Computes heading in degrees between two coordinates."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        d_lon = lon2 - lon1
        x = np.sin(d_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        heading = np.degrees(np.arctan2(x, y))
        return (heading + 360) % 360

    def _prepare_eta_navigation_features(self, df):
        """Prepares features for ETA and Navigation models."""
        df_prepared = df.copy()
        key_num_cols = ['LAT', 'LON', 'Speed Logged [KN]', 'Remaining Distance to PS [NM]', 'Remaining Time to PS']
        for col in key_num_cols:
            if col in df_prepared.columns:
                df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
                df_prepared[col].fillna(df_prepared[col].median(), inplace=True)
            else:
                df_prepared[col] = 0

        # --- FIX 1: Change how prev speed column is created ---
        for col in ['LAT', 'LON']: # Only loop for LAT and LON
            df_prepared[f'{col}_prev'] = df_prepared[col].shift(SEQ_LEN).fillna(method='bfill')
        
        # Create 'Speed_prev' explicitly to match old models
        df_prepared['Speed_prev'] = df_prepared['Speed Logged [KN]'].shift(SEQ_LEN).fillna(method='bfill')
        # -----------------------------------------------------

        df_prepared['Heading'] = df_prepared.apply(
            lambda r: self._calculate_heading(r['LAT_prev'], r['LON_prev'], r['LAT'], r['LON']),
            axis=1
        )
        return df_prepared

    def get_processed_dataframes(self):
        """Returns all processed dataframes required for model training."""
        hydro_df = self._impute_hydro_columns(self.raw_df.copy())
        hydro_df_final = self._calculate_derived_hydro_features(hydro_df)
        eta_nav_df = self._prepare_eta_navigation_features(self.raw_df.copy())
        return hydro_df_final, eta_nav_df


class ModelManager:
    """Handles training, evaluation, saving, and loading of all models."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _train_evaluate_regression_model(self, df, target_col, feature_cols):
        """Trains and evaluates a RandomForestRegressor for a given target."""
        model_data = df[feature_cols + [target_col]].copy()
        for col in model_data.columns:
            model_data[col] = pd.to_numeric(model_data[col], errors='coerce')
        model_data.dropna(inplace=True)

        X = model_data[feature_cols]
        y = model_data[target_col]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        residuals = y - model.predict(X)
        threshold = 3 * residuals.std()
        return model, threshold, X # Return full X data for SHAP

    def train_all_models(self, hydro_df, eta_nav_df):
        """Orchestrates the training of all models."""
        print("\n--- Training Hydrostatic Models ---")
        for name, config in HYDRO_MODEL_DEFINITIONS.items():
            model, threshold, X_data = self._train_evaluate_regression_model(
                hydro_df, config['target'], config['features']
            )
            self._save_model(name, (model, threshold, X_data)) # Save X_data

        print("\n--- Training Navigation Models ---")
        # --- FIX 2: Use 'Speed_prev' in nav_features ---
        nav_features = ['LAT_prev', 'LON_prev', 'Speed_prev', 'Heading']
        # -----------------------------------------------
        for target in ['LAT', 'LON']:
            model, threshold, X_data = self._train_evaluate_regression_model(eta_nav_df, target, nav_features)
            self._save_model(f"{target}_Predictor", (model, threshold, X_data)) # Save X_data

        print("\n--- Training ETA Model ---")
        # --- FIX 3: Use 'Speed_prev' in eta_features ---
        eta_features = ['LAT', 'LON', 'LAT_prev', 'LON_prev', 'Speed_prev',
                        'Speed Logged [KN]', 'Remaining Distance to PS [NM]', 'Heading']
        # -----------------------------------------------
        model, threshold, X_data = self._train_evaluate_regression_model(
            eta_nav_df, 'Remaining Time to PS', eta_features
        )
        self._save_model("ETA_Predictor", (model, threshold, X_data)) # Save X_data
        print("\nAll models trained and saved successfully.")

    def _save_model(self, model_name, model_data):
        """Saves a model and its associated data to a file."""
        path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(model_data, path)
        print(f"Saved {model_name} to {path}")

    def load_all_models(self):
        """Loads all trained models from the specified directory."""
        loaded_models = {}
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".joblib"):
                model_name = filename.replace(".joblib", "")
                path = os.path.join(self.model_dir, filename)
                try:
                    # --- SHAP CHANGE: Load background data ---
                    model, threshold, X_data = joblib.load(path)
                    loaded_models[model_name] = {
                        'model': model, 
                        'threshold': threshold,
                        'background_data': X_data # Store background data
                    }
                    print(f"Loaded {model_name} from {path}")
                    # -----------------------------------------
                except Exception as e:
                    print(f"Error loading {filename}: {e}. Skipping.")
        return loaded_models


class AnomalyDetector:
    """Uses trained models and an LLM to detect and explain anomalies."""

    def __init__(self, models, api_key):
        self.models = models
        genai.configure(api_key=api_key)
        
        # --- SHAP CHANGE: Pre-calculate explainers ---
        self.explainers = {}
        print("\n--- Pre-calculating SHAP Explainers ---")
        for name, config in self.models.items():
            try:
                model = config['model']
                background = config['background_data']
                
                # Sample background data for performance if it's too large
                if len(background) > 100:
                    background = shap.sample(background, 100)
                
                # Use shap.Explainer for flexibility
                explainer = shap.Explainer(model, background)
                self.explainers[name] = explainer
                print(f"  - Explainer created for {name}")
            except Exception as e:
                print(f"  - ⚠️ Could not create SHAP explainer for {name}: {e}")
        # ---------------------------------------------

        self.llm = genai.GenerativeModel('gemini-2.5-flash') # Use a fast model

    def _prepare_single_report(self, raw_report_data):
        """Prepares a single raw report with necessary derived features."""
        report = pd.Series(raw_report_data).copy()
        
        # Ensure 'Speed_prev' is also numeric if it exists
        numeric_cols = list(report.index)
        if 'Speed_prev' not in numeric_cols:
             numeric_cols.append('Speed_prev')

        # --- FIX for AttributeError ---
        for col in numeric_cols:
             # Coerce all, fillna(0) for safety during prediction
            value = pd.to_numeric(report.get(col, 0), errors='coerce')
            if pd.isna(value):
                report[col] = 0.0 # Use float for consistency
            else:
                report[col] = value
        # ------------------------------


        report['Trim [M]'] = report.get('Draught Aft [M]', 0) - report.get('Draught Fore [M]', 0)
        report['Draught Mean [M]'] = (report.get('Draught Fore [M]', 0) + report.get('Draught Aft [M]', 0)) / 2
        report['Heading'] = VesselDataProcessor._calculate_heading(
            report.get('LAT_prev', 0), report.get('LON_prev', 0),
            report.get('LAT', 0), report.get('LON', 0)
        )
        return report

    # --- SHAP CHANGE: Add shap_explanation_str to prompt ---
    def _get_llm_explanation(self, target_col, actual, predicted, shap_explanation_str):
        """Generates a structured JSON explanation for an anomaly using an LLM."""
        prompt = f"""
        Analyze a vessel data anomaly.
        
        Data:
        - Feature: "{target_col}"
        - Reported Value: {actual:.2f}
        - Expected Value (from model): {predicted:.2f}
        
        Model Explanation (SHAP Feature Importance):
        This shows *why* the model predicted the value it did. A high positive/negative value means that feature pushed the prediction up/down.
        - {shap_explanation_str}
        
        Task:
        Generate a JSON object with keys "error", "summary", and "severity" ('error', 'warning', or 'info').
        - "error": Concatenate the feature name, reported value, and expected value into a technical summary.
        - "summary": Using the SHAP explanation, provide a *human-readable, actionable recommendation*. For example, if 'Cargo Weight' had a huge impact, suggest "Verify 'Cargo Weight' as it is strongly influencing the anomalous 'Draught Mean' prediction."
        - "severity": Classify as 'error', 'warning', or 'info'.
        """
        # -----------------------------------------------------
        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return {
                "error": "Value is outside the expected range.",
                "summary": "Verify sensor readings and manual data entry.",
                "severity": "warning"
            }

    def check_report(self, raw_report_data):
        """Runs a full anomaly check on a single report."""
        print("\n--- Running Multi-Model Anomaly Check ---")
        report = self._prepare_single_report(raw_report_data)
        anomalies = []
        anomaly_id = 1

        model_configs = {**HYDRO_MODEL_DEFINITIONS}
        model_configs.update({
            'LAT_Predictor': {'target': 'LAT', 'features': ['LAT_prev', 'LON_prev', 'Speed_prev', 'Heading']},
            'LON_Predictor': {'target': 'LON', 'features': ['LAT_prev', 'LON_prev', 'Speed_prev', 'Heading']},
            'ETA_Predictor': {'target': 'Remaining Time to PS', 'features': ['LAT', 'LON', 'LAT_prev', 'LON_prev', 'Speed_prev', 'Speed Logged [KN]', 'Remaining Distance to PS [NM]', 'Heading']}
        })
        
        for name, config_info in self.models.items():
            model_name_key = name.split('.')[0] if '.' in name else name
            if model_name_key not in model_configs:
                continue

            target = model_configs[model_name_key]['target']
            features = model_configs[model_name_key]['features']
            
            try:
                # Ensure all features are present in the report for prediction
                if not all(f in report.index for f in features):
                    print(f"  ⚠️ Skipping {name}: Missing features {list(set(features) - set(report.index))}")
                    continue
                    
                actual = report[target]
                input_df = pd.DataFrame([report[features]]).astype(float)
                predicted = config_info['model'].predict(input_df)[0]
                residual = actual - predicted
                
                print(f"Checking '{target}': Actual={actual:.2f}, Predicted={predicted:.2f}, Threshold=+/~{config_info['threshold']:.2f}")

                if abs(residual) > config_info['threshold']:
                    print(f"  🚩 Anomaly Detected in {target}!")

                    shap_explanation_str = "SHAP explanation not available."
                    if model_name_key in self.explainers:
                        try:
                            explainer = self.explainers[model_name_key]
                            shap_values = explainer(input_df) 
                            
                            shap_vals_for_row = shap_values.values[0]

                            feature_importance = dict(zip(features, shap_vals_for_row))
                            sorted_importance = sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
                            importance_str = ", ".join([f"'{f}': {v:.3f}" for f, v in sorted_importance])
                            shap_explanation_str = f"Feature contributions to prediction: [{importance_str}]"
                        
                        except Exception as e:
                            print(f"  - ⚠️ SHAP explanation failed for {model_name_key}: {e}")
                    
                    llm_output = self._get_llm_explanation(target, actual, predicted, shap_explanation_str)
                    anomaly_obj = {
                        "id": anomaly_id,
                        "field": target,
                        "error": llm_output.get("error"),
                        "summary": llm_output.get("summary"),
                        "severity": llm_output.get("severity", "warning").lower()
                    }
                    if anomaly_obj["severity"] == "error":
                        anomaly_obj["comment"] = "Please resolve this issue before submission."
                    anomalies.append(anomaly_obj)
                    anomaly_id += 1
            except Exception as e:
                print(f"  ⚠️ Error checking {name}: {e}")

        print("\n--- Consolidated Anomaly Report (JSON) ---")
        print(json.dumps(anomalies, indent=2))
        return anomalies

def main():
    """Main execution block for the vessel analytics pipeline."""
    # Ensure file exists before proceeding
    if not os.path.exists(FILE_PATH):
        print(f"Error: Data file not found at {FILE_PATH}")
        print("Please download the file and place it in the correct location.")
        return

    processor = VesselDataProcessor(FILE_PATH, HEADER_ROW_INDEX)
    hydro_data, eta_nav_data = processor.get_processed_dataframes()

    trainer = ModelManager(MODEL_DIR)
    trainer.train_all_models(hydro_data, eta_nav_data)

    loaded_models = trainer.load_all_models()
    
    if not loaded_models:
        print("No models were loaded. Aborting anomaly detection.")
        return

    detector = AnomalyDetector(loaded_models, GEMINI_API_KEY)

    sample_report = {
        'Draught Fore [M]': 9.0, 'Draught Aft [M]': 9.5,
        'Ballast Water [MT]': 2000, 'Cargo Weight [MT]': 60000,
        'LAT': 9.37, 'LON': 200.22, 
        'LAT_prev': 16.05, 'LON_prev': 96.17,
        'Speed Logged [KN]': 13.1, 
        'Speed_prev': 12.5, # <-- Use 'Speed_prev' to match
        'Remaining Distance to PS [NM]': 220.5,
        'Remaining Time to PS': 610 # Anomalous ETA
    }
    
    detector.check_report(sample_report)

if __name__ == "__main__":
    main()

