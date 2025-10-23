import pandas as pd
import numpy as np
import re
from litai import LLM
import shap

# Import the new classes and necessary configs from the refactored pipeline
from full_pipeline import ModelManager, AnomalyDetector, MODEL_DIR, GEMINI_API_KEY

class InferenceApp:
    """Orchestrates the model inference process."""

    def __init__(self, model_dir, api_key):
        """Initializes the app, loading models and the detector."""
        print("Initializing InferenceApp...")
        self.model_dir = model_dir
        self.api_key = api_key
        self.model_manager = ModelManager(self.model_dir)
        self.models = self._load_models()
        
        if self.models:
            self.detector = AnomalyDetector(self.models, self.api_key)
        else:
            self.detector = None
            print("Error: AnomalyDetector not initialized due to missing models.")

    def _load_models(self):
        """Loads all trained models using the ModelManager."""
        print(f"\n--- Loading models from '{self.model_dir}' ---")
        models = self.model_manager.load_all_models()
        if not models:
            print("Error: No models were loaded. Please run full_pipeline.py first.")
            return None
        print(f"Successfully loaded {len(models)} models.")
        return models

    def _generate_sample_data(self):
        """Generates a single sample data point for inference."""
        data = {
            'Timestamp': pd.to_datetime('2024-01-01 10:00:00'),
            'Draught Fore [M]': 9.0,    # Anomalous
            'Draught Aft [M]': 9.5,     # Anomalous
            'Speed [kn]': 2.0,
            'RPM': 110,
            'Navigation': 'Auto',
            'ETA': pd.to_datetime('2024-01-05 08:00:00'),
            'Hydro Pressure (bar)': 95.0,
            'Fuel Consumption (L/H)': 88.0,
            'LAT': 9.37000,             # Anomalous
            'LON': 200.22000,           # Anomalous
            'LAT_prev': 16.05000 - 0.1,
            'LON_prev': 96.17000 - 0.1,
            'Speed Logged [KN]': 2.0,   # Matched to 'Speed [kn]'
            
            # --- Providing both keys to resolve conflicts ---
            'Speed Logged [KN]_prev': 11.5, # Key as defined in full_pipeline.py
            'Speed_prev': 11.5,             # Fallback key for older/conflicting models
            
            'Remaining Distance to PS [NM]': 250.0,
            'Remaining Time to PS': 1500, # Anomalous
            'Ballast Water [MT]': 2000,
            'Cargo Weight [MT]': 60000  # Anomalous (from full_pipeline sample)
        }
        
        report_dict = {k: v for k, v in data.items() if k not in ['Timestamp', 'ETA']}
        return report_dict

    def run_inference(self):
        """Generates sample data and runs the anomaly check."""
        if not self.detector:
            print("Inference aborted: Detector not initialized.")
            return

        raw_new_data_report = self._generate_sample_data()
        print("\nRaw incoming data for inference:")
        for k, v in raw_new_data_report.items():
            print(f"  {k}: {v}")

        # Use the detector's check_report method
        final_report = self.detector.check_report(raw_new_data_report)

        print("\nInference complete. Check the consolidated anomaly report above.")

def main():
    """Main execution function."""
    print("Starting multi-model inference pipeline.")
    
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("Warning: GEMINI_API_KEY is not set in full_pipeline.py. LLM explanations may fail.")

    app = InferenceApp(MODEL_DIR, GEMINI_API_KEY)
    app.run_inference()

if __name__ == "__main__":
    main()

