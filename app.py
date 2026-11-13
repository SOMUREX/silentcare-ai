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
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from data_generator import (
    FEATURES, FEATURE_LABELS, DEPARTMENTS,
    generate_timeline, engineer_features_single,
    load_or_build_dataset
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


# ── Helper: build single-row feature vector from raw dict ─────────────────────
def vectorize_inputs(raw_dict):
    """Convert raw input values into full 43 engineered feature vector."""
    feat_cols = MODELS.get("feat_cols", FEATURES)
    scaler = MODELS.get("scaler")

    # Start with base features
    row = {}
    for f in FEATURES:
        # Default to 0.5 if not provided, scale from percentage if > 1
        val = float(raw_dict.get(f, 0.5))
        if val > 1.0:
            val = val / 100.0
        row[f] = max(0.0, min(1.0, val))

    # Rolling & trends (for single point simulation, roll=val, trend=0)
    for f in FEATURES:
        row[f"{f}_roll7"] = row[f]
        row[f"{f}_trend"] = float(raw_dict.get(f"{f}_trend", 0.0))

    # Interaction terms
    row["wait_x_workload"] = row["waiting_time"] * row["staff_workload"]
    row["icu_x_nurse"] = row["icu_occupancy"] * (1.0 - row["nurse_patient_ratio"])
    row["equip_stress"] = row["equipment_downtime"] * (1.0 - row["equipment_availability"])
    row["pharmacy_complaint"] = row["pharmacy_delay"] * row["patient_complaints"]

    # Align columns exactly to model training order
    vec = np.array([[row.get(col, 0.0) for col in feat_cols]], dtype=float)
    if scaler is not None:
        vec = scaler.transform(vec)
    return vec, row


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

    health_scores = [round(max(0, min(100, float(h))), 1) for h in health_scores]

    # Current state (last day)
    current_health = health_scores[-1]
    current_state = (
        "Critical" if current_health < 40 else
        "Warning" if current_health < 60 else
        "Moderate" if current_health < 75 else
        "Good"
    )

    # 2. Indicator Trends
    indicator_data = {}
    for feat in ["waiting_time", "staff_workload", "bed_occupancy"]:
        vals = [round(float(v), 3) for v in df_raw[feat].tolist()]
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
            prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
        else:
            prob = 0.5
        risk[h_key] = {"label": h_label, "probability": round(prob * 100, 1)}

    # 4. Anomaly Detection
    iso = MODELS.get("iso")
    if iso:
        anomaly_flags = (iso.predict(X) == -1).tolist()
        anomaly_indices = [i for i, v in enumerate(anomaly_flags) if v]
    else:
        anomaly_indices = []

    # 5. Feature Importance
    top_features = MODELS.get("top_features", [])
    feature_importance = [
        {"feature": FEATURE_LABELS.get(f, f.replace("_", " ").title()), "importance": round(float(imp) * 100, 2)}
        for f, imp in top_features[:10]
    ]

    # 6. Department Status
    dept_status = {}
    last_row = df_raw.iloc[-1]
    for dept, feats in DEPARTMENTS.items():
        vals = []
        for feat in feats:
            v = float(last_row.get(feat, 0.5))
            if feat in ["equipment_availability", "nurse_patient_ratio"]:
                stress = 1 - v
            else:
                stress = v
            vals.append(stress)
        avg_stress = np.mean(vals)
        workload = round(float(avg_stress) * 100, 1)
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


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Real-time 'What-If' Stress Simulation endpoint."""
    data = request.get_json() or {}
    vec, normalized_inputs = vectorize_inputs(data)

    # 1. Health Score
    rfr = MODELS.get("rfr")
    pred_health = float(rfr.predict(vec)[0]) if rfr else 50.0
    pred_health = round(max(0.0, min(100.0, pred_health)), 1)

    state = (
        "Critical" if pred_health < 40 else
        "Warning" if pred_health < 60 else
        "Moderate" if pred_health < 75 else
        "Good"
    )

    # 2. Risk Predictions across horizons
    risk = {}
    horizon_map = {"24h": "24 h", "72h": "72 h", "168h": "7 d", "336h": "14 d"}
    for h_key, h_label in horizon_map.items():
        rfc = MODELS.get(f"rfc_{h_key}")
        if rfc:
            proba = rfc.predict_proba(vec)[0]
            classes = list(rfc.classes_)
            prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
        else:
            prob = 0.5
        risk[h_key] = {"label": h_label, "probability": round(prob * 100, 1)}

    # 3. Anomaly Flag
    iso = MODELS.get("iso")
    is_anomaly = bool(iso.predict(vec)[0] == -1) if iso else False

    # 4. Department Status
    dept_status = {}
    for dept, feats in DEPARTMENTS.items():
        stresses = []
        for feat in feats:
            v = float(normalized_inputs.get(feat, 0.5))
            if feat in ["equipment_availability", "nurse_patient_ratio"]:
                stresses.append(1.0 - v)
            else:
                stresses.append(v)
        avg_s = np.mean(stresses)
        workload = round(float(avg_s) * 100, 1)
        st = "Critical" if workload > 70 else "Warning" if workload > 50 else "Normal"
        dept_status[dept] = {"workload": workload, "status": st}

    return jsonify({
        "health_score": pred_health,
        "state": state,
        "risk": risk,
        "is_anomaly": is_anomaly,
        "dept_status": dept_status,
        "inputs": {k: round(v * 100, 1) for k, v in normalized_inputs.items() if k in FEATURES}
    })


@app.route("/api/seeds")
def available_seeds():
    """Return available test seeds (40–49)."""
    return jsonify({"seeds": list(range(40, 50))})


@app.route("/api/data_sources")
def data_sources():
    """Return information on the integrated real datasets."""
    return jsonify({
        "sources": [
            {"name": "MIMIC-IV Clinical (v2.2)", "records": "275 admissions, 140 ICU stays, 15K pharmacy orders", "features": "Bed & ICU Occupancy, Transfers, Pharmacy Delays"},
            {"name": "MIMIC-IV Emergency Dept", "records": "222 ED stays with triage vitals & acuity", "features": "Waiting Time, Emergency Volume"},
            {"name": "Hospital Resources Census", "records": "Multi-facility daily bed/ICU/PACU census", "features": "Occupancy Rates, Diversion Flags, Staffing Adequacy"},
            {"name": "Staffing Operations", "records": "13,523 shift logs across departments", "features": "Staff Workload, Nurse-to-Patient Ratio, Overtime"},
            {"name": "Medical Equipment Management", "records": "17,461 asset monitoring & maintenance records", "features": "Equipment Availability, Unplanned Downtime"},
            {"name": "Surgical OR Schedule", "records": "4,226 operating room surgical cases", "features": "Turnover Delays, Cancellation Rates"},
            {"name": "NHS England A&E Time Series", "records": "208 monthly time-series points (2010–2026)", "features": "Emergency Cases, >4h & >12h Waiting Dynamics"},
            {"name": "NHS England Diagnostics Backlog", "records": "Monthly national diagnostics queue tables", "features": "Diagnostic Turnaround Stress"},
            {"name": "NHS England Discharge Timeliness", "records": "Provider-level acute hospital discharge logs", "features": "Post-ready Discharge Delay Days"},
            {"name": "NHS Written Complaints Reports", "records": "305,264 annual patient grievance records", "features": "Patient Complaints Metric"},
            {"name": "NHS Organisation Overnight Census", "records": "Overnight bed census across England trusts", "features": "Baseline Capacities"},
            {"name": "Facility Master Topology", "records": "Level I–IV Trauma hospital operational specs", "features": "Capacity baselines"}
        ],
        "models": {
            "regressor": "Random Forest Regressor (250 trees, max depth 14, R² = 0.9914)",
            "classifiers": "4x Random Forest Classifiers (24h, 72h, 7d, 14d horizons)",
            "anomaly": "Isolation Forest (200 trees, 6% contamination)"
        }
    })


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
