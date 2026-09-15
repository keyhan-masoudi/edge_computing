import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings

# Suppress warnings for clean terminal output
warnings.filterwarnings("ignore")

from data_generator import generate_dataset
from feature_extractor import build_feature_table

FEATURE_COLS = [
    "rms", "peak_amplitude", "crest_factor", "energy",
    "zero_crossing_rate", "kurtosis_val", "skewness_val",
    "signal_entropy", "dominant_frequency", "spectral_centroid",
    "band_power_ratio", "burst_count",
]

# Map fault labels to states as required by the Edge Decision Engine
STATE_MAP = {
    "normal": "normal",
    "friction": "warning",
    "impact": "critical",
    "leakage": "warning",
    "crack_growth": "critical",
    "cavitation": "critical",
    "unknown_anomaly": "unknown",
}

class AEStateDetector:
    """
    Lightweight model using a single Small Random Forest.
    Fulfills PDF constraints: simple, edge-friendly, and generates all 4 required outputs.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Explicitly "small" Random Forest (few trees, low depth) for Edge Computing
        self.rf = RandomForestClassifier(
            n_estimators=30,  
            max_depth=5,      
            random_state=42,
        )
        self.is_fitted = False
        self.classes_ = []

    def _map_label_to_state(self, label: str) -> str:
        return STATE_MAP.get(label, "unknown")

    def fit(self, feature_df: pd.DataFrame):
        """Train the model on the feature table."""
        X = feature_df[FEATURE_COLS].values
        y_labels = feature_df["fault_label"].values

        X_scaled = self.scaler.fit_transform(X)

        # Train a single Random Forest on all data
        self.rf.fit(X_scaled, y_labels)
        self.classes_ = list(self.rf.classes_)
        self.is_fitted = True
        print("[Model] AEStateDetector (Small Random Forest) fitted successfully.")

    def predict_single(self, features: dict) -> dict:
        """
        Predict state for a single sample.
        Returns exact 4 fields required by PDF: 
        predicted_state, confidence, model_uncertainty, fault_score
        """
        x = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]])
        x_scaled = self.scaler.transform(x)

        if not self.is_fitted:
            return {
                "predicted_state": "unknown",
                "confidence": 0.0,
                "model_uncertainty": 1.0,
                "fault_score": 1.0,
            }

        # Retrieve class probabilities
        proba = self.rf.predict_proba(x_scaled)[0]
        max_idx = np.argmax(proba)
        rf_pred_label = self.classes_[max_idx]
        
        # 1. Confidence: Probability of the predicted class
        confidence = float(proba[max_idx])

        # 2. Uncertainty: Inverse of confidence
        model_uncertainty = round(1.0 - confidence, 4)

        # 3. Fault Score: Probability that the machine is NOT normal
        if "normal" in self.classes_:
            normal_idx = self.classes_.index("normal")
            fault_score = round(1.0 - float(proba[normal_idx]), 4)
        else:
            fault_score = 1.0

        # 4. Predicted State + Hard Rule for "unknown"
        # Rule: If model lacks confidence, force 'unknown' state (As required by PDF)
        if confidence < 0.45:
            predicted_state = "unknown"
        else:
            predicted_state = self._map_label_to_state(rf_pred_label)

        return {
            "predicted_state": predicted_state,
            "confidence": round(confidence, 4),
            "model_uncertainty": model_uncertainty,
            "fault_score": fault_score,
        }

    def predict_batch(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Execute predictions across an entire feature dataframe."""
        results = []
        for _, row in feature_df.iterrows():
            feats = {c: row[c] for c in FEATURE_COLS}
            pred = self.predict_single(feats)
            results.append(pred)
        return pd.DataFrame(results)

def evaluate_model(feature_df: pd.DataFrame) -> None:
    """Quick cross-validation sanity check on the model."""
    df_known = feature_df.copy()
    
    if len(df_known) < 10:
        print("[Warning] Not enough samples for stable CV.")
        return
        
    X = df_known[FEATURE_COLS].values
    y = df_known["fault_label"].apply(lambda l: STATE_MAP.get(l, "unknown")).values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(rf, X_s, y, cv=cv, scoring="accuracy")
        print(f"[Model CV] Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    except ValueError as e:
        print(f"[Model CV Warning] {e}. Normal for small datasets.")

if __name__ == "__main__":
    # Local test block
    samples = generate_dataset(80)
    ft = build_feature_table(samples)
    
    detector = AEStateDetector()
    detector.fit(ft)
    preds = detector.predict_batch(ft)
    
    print("\n[Success] Prediction distribution:")
    print(preds["predicted_state"].value_counts())
    
    print("\n[Outputs Verification (First 3 samples)]")
    print(preds.head(3).to_string())
    
    print("\n[Evaluation]")
    evaluate_model(ft)