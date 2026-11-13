"""
SilentCare AI - Model Training Module
Trains Random Forest Regressor, Random Forest Classifier, and Isolation Forest
on synthetic hospital operational data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

from data_generator import FEATURES, build_dataset, engineer_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
HORIZONS = [("24h", "24 h"), ("72h", "72 h"), ("168h", "7 d"), ("336h", "14 d")]


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all engineered feature column names."""
    base = FEATURES[:]
    rolls = [f"{f}_roll7" for f in FEATURES]
    trends = [f"{f}_trend" for f in FEATURES]
    interactions = ["wait_x_workload", "icu_x_nurse", "equip_stress", "pharmacy_complaint"]
    all_cols = base + rolls + trends + interactions
    return [c for c in all_cols if c in df.columns]


def train_models():
    """Train and save all ML models."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Generating training data (seeds 0-39)...")
    train_raw = build_dataset(range(40))

    print("Generating test data (seeds 40-49)...")
    test_raw = build_dataset(range(40, 50))

    print("Engineering features...")
    train_df = engineer_features(train_raw)
    test_df = engineer_features(test_raw)

    feat_cols = get_feature_cols(train_df)
    print(f"Feature count: {len(feat_cols)}")

    X_train = train_df[feat_cols].fillna(0).values
    y_train_reg = train_df["health_score"].values
    X_test = test_df[feat_cols].fillna(0).values
    y_test_reg = test_df["health_score"].values

    # Scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Random Forest Regressor ---
    print("Training Random Forest Regressor (250 trees, depth 14)...")
    rfr = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    rfr.fit(X_train_sc, y_train_reg)
    y_pred_reg = rfr.predict(X_test_sc)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"  Regressor: MAE={mae:.2f}, R2={r2:.4f}")
    joblib.dump(rfr, os.path.join(MODELS_DIR, "rfr.pkl"))

    # --- Random Forest Classifiers ---
    for col_suffix, label in HORIZONS:
        target_col = f"degraded_{col_suffix}"
        if target_col not in train_df.columns:
            print(f"  Warning: {target_col} not found, skipping")
            continue

        y_train_cls = train_df[target_col].values
        y_test_cls = test_df[target_col].values

        rfc = RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
        rfc.fit(X_train_sc, y_train_cls)
        y_pred_cls = rfc.predict(X_test_sc)
        print(f"  Classifier ({label}):")
        print(classification_report(y_test_cls, y_pred_cls, zero_division=0))
        joblib.dump(rfc, os.path.join(MODELS_DIR, f"rfc_{col_suffix}.pkl"))

    # --- Isolation Forest ---
    print("Training Isolation Forest (200 estimators, contamination=0.06)...")
    iso = IsolationForest(n_estimators=200, contamination=0.06, random_state=42, n_jobs=-1)
    iso.fit(X_train_sc)
    joblib.dump(iso, os.path.join(MODELS_DIR, "iso.pkl"))

    # Save scaler and feature list
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feat_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    # Feature importance
    importance = dict(zip(feat_cols, rfr.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    joblib.dump(top_features, os.path.join(MODELS_DIR, "top_features.pkl"))
    print("Top features:", [f[0] for f in top_features])
    print("\nAll models saved!")
    return feat_cols


if __name__ == "__main__":
    train_models()
