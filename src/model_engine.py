"""
Model Engine Module for Project ACAT.
Encapsulates LightGBM clinical inference, threshold calibration,
and native observation-level TreeSHAP attributions.
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

class CardiacRiskEngine:
    def __init__(self, model_path="src/model.joblib", metadata_path="src/model_metadata.json"):
        self.model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata["feature_columns"]
        self.threshold = self.metadata["optimal_threshold"]
        self.explainer = shap.TreeExplainer(self.model)

    def predict_risk(self, patient_dict: dict) -> dict:
        """
        Calculates calibrated risk probability and top TreeSHAP drivers without hardcoded overrides.
        """
        input_df = pd.DataFrame([patient_dict])[self.feature_cols]
        
        # Native ML model inference
        prob = float(self.model.predict_proba(input_df)[:, 1][0])
        is_critical = prob >= self.threshold
        
        # Observation-level TreeSHAP calculation
        shap_vals = self.explainer(input_df)
        attributions = dict(zip(self.feature_cols, shap_vals.values[0]))
        sorted_drivers = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_drivers = [f"{k} (SHAP impact: {v:+.4f})" for k, v in sorted_drivers[:3]]
        
        return {
            "probability": round(prob, 4),
            "decision_threshold": self.threshold,
            "risk_tier": "Critical Risk (High)" if is_critical else "Standard Risk (Low/Moderate)",
            "top_drivers": top_drivers,
            "is_critical": is_critical
        }
