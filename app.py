"""
SilentCare AI - Flask Application
Main web server providing API endpoints for the dashboard.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from data_generator import (
    FEATURES, FEATURE_LABELS, DEPARTMENTS,
    generate_timeline, engineer_features_single
)

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Load models ────────────────────────────────────────────────────────────────
def load_models():
    models = {}
    try:
        models["rfr"] = joblib.load(os.path.join(MODELS_DIR, "rfr.pkl"))
        models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        models["feat_cols"] = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        models["iso"] = joblib.load(os.path.join(MODELS_DIR, "iso.pkl"))
        models["top_features"] = joblib.load(os.path.join(MODELS_DIR, "top_features.pkl"))
        for h in ["24h", "72h", "168h", "336h"]:
            models[f"rfc_{h}"] = joblib.load(os.path.join(MODELS_DIR, f"rfc_{h}.pkl"))
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")
    return models


MODELS = load_models()


# ── Helper: prepare features from raw timeline df ──────────────────────────────
def prepare_X(df: pd.DataFrame):
    feat_cols = MODELS.get("feat_cols", FEATURES)
    scaler = MODELS.get("scaler")
    X = df[feat_cols].fillna(0).values
    if scaler is not None:
        X = scaler.transform(X)
    return X


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard/<int:seed>")
def dashboard_data(seed: int):
    """Generate full dashboard payload for a given seed (timeline)."""
    df_raw = generate_timeline(seed=seed, n_days=180)
    df_eng = engineer_features_single(df_raw)

    feat_cols = MODELS.get("feat_cols", FEATURES)
    scaler = MODELS.get("scaler")

    # Align columns
    for col in feat_cols:
        if col not in df_eng.columns:
            df_eng[col] = 0.0

    X = df_eng[feat_cols].fillna(0).values
    if scaler is not None:
        X = scaler.transform(X)

    days = list(range(180))

    # 1. Health Score (predicted by RF Regressor)
    rfr = MODELS.get("rfr")
    if rfr:
        health_scores = rfr.predict(X).tolist()
    else:
        health_scores = df_raw["health_score"].tolist()

    # Clamp to 0-100
    health_scores = [max(0, min(100, h)) for h in health_scores]

    # Current state (last day)
    current_health = round(health_scores[-1], 1)
    current_state = (
        "Critical" if current_health < 40 else
        "Warning" if current_health < 60 else
        "Moderate" if current_health < 75 else
        "Good"
    )

    # 2. Indicator Trends (last 60 days for display)
    indicator_data = {}
    for feat in ["waiting_time", "staff_workload", "bed_occupancy"]:
        vals = df_raw[feat].tolist()
        indicator_data[feat] = {
            "label": FEATURE_LABELS[feat],
            "values": vals,
        }

    # 3. Future Risk (classification)
    risk = {}
    horizon_map = {"24h": "24 h", "72h": "72 h", "168h": "7 d", "336h": "14 d"}
    last_X = X[-1:].reshape(1, -1)
    for h_key, h_label in horizon_map.items():
        rfc = MODELS.get(f"rfc_{h_key}")
        if rfc:
            proba = rfc.predict_proba(last_X)[0]
            classes = list(rfc.classes_)
            # If classifier only saw class 0 (no degradation), risk is 0
            prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
        else:
            prob = 0.5
        risk[h_key] = {"label": h_label, "probability": round(prob * 100, 1)}

    # 4. Anomaly Detection
    iso = MODELS.get("iso")
    if iso:
        anomaly_scores = iso.score_samples(X)
        anomaly_flags = (iso.predict(X) == -1).tolist()
        anomaly_indices = [i for i, v in enumerate(anomaly_flags) if v]
    else:
        anomaly_flags = [False] * 180
        anomaly_indices = []

    # 5. Feature Importance
    top_features = MODELS.get("top_features", [])
    feature_importance = [
        {"feature": FEATURE_LABELS.get(f, f), "importance": round(float(imp) * 100, 2)}
        for f, imp in top_features[:10]
    ]

    # 6. Department Status
    dept_status = {}
    last_row = df_raw.iloc[-1]
    for dept, feats in DEPARTMENTS.items():
        vals = []
        for feat in feats:
            v = float(last_row.get(feat, 0.5))
            # Normalize: equipment_availability and nurse_patient_ratio are inverted
            if feat in ["equipment_availability", "nurse_patient_ratio"]:
                stress = 1 - v
            else:
                stress = v
            vals.append(stress)
        avg_stress = np.mean(vals)
        workload = round(avg_stress * 100, 1)
        status = (
            "Critical" if workload > 70 else
            "Warning" if workload > 50 else
            "Normal"
        )
        dept_status[dept] = {"workload": workload, "status": status}

    # 7. Diagnostics
    all_feature_values = {
        feat: round(float(df_raw[feat].iloc[-1]), 3) for feat in FEATURES
    }
    # Find top stressors
    stressor_scores = {}
    for feat in FEATURES:
        v = float(df_raw[feat].iloc[-1])
        if feat in ["equipment_availability", "nurse_patient_ratio"]:
            stressor_scores[feat] = 1 - v
        else:
            stressor_scores[feat] = v
    top_stressors = sorted(stressor_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    diagnostics = [
        {"feature": FEATURE_LABELS.get(f, f), "value": round(v * 100, 1)}
        for f, v in top_stressors
    ]

    return jsonify({
        "seed": seed,
        "days": days,
        "health_scores": health_scores,
        "current_health": current_health,
        "current_state": current_state,
        "indicator_data": indicator_data,
        "risk": risk,
        "anomaly_indices": anomaly_indices,
        "anomaly_count": len(anomaly_indices),
        "feature_importance": feature_importance,
        "dept_status": dept_status,
        "diagnostics": diagnostics,
        "all_feature_values": all_feature_values,
    })


@app.route("/api/seeds")
def available_seeds():
    """Return available test seeds (40–49)."""
    return jsonify({"seeds": list(range(40, 50))})


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Trigger model retraining (for demo purposes)."""
    global MODELS
    try:
        from train import train_models
        train_models()
        MODELS = load_models()
        return jsonify({"status": "success", "message": "Models retrained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
