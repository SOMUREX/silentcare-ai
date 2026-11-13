"""
SilentCare AI - Synthetic Hospital Data Generator
Generates 180-day synthetic hospital operational time-series data
with 13 operational indicators.
"""

import numpy as np
import pandas as pd


FEATURES = [
    "waiting_time",
    "staff_workload",
    "bed_occupancy",
    "equipment_availability",
    "patient_transfers",
    "patient_complaints",
    "emergency_cases",
    "icu_occupancy",
    "diagnostic_turnaround",
    "discharge_delay",
    "nurse_patient_ratio",
    "equipment_downtime",
    "pharmacy_delay",
]

FEATURE_LABELS = {
    "waiting_time": "Waiting Time",
    "staff_workload": "Staff Workload",
    "bed_occupancy": "Bed Occupancy",
    "equipment_availability": "Equipment Availability",
    "patient_transfers": "Patient Transfers",
    "patient_complaints": "Patient Complaints",
    "emergency_cases": "Emergency Cases",
    "icu_occupancy": "ICU Occupancy",
    "diagnostic_turnaround": "Diagnostic Turnaround",
    "discharge_delay": "Discharge Delay",
    "nurse_patient_ratio": "Nurse-Patient Ratio",
    "equipment_downtime": "Equipment Downtime",
    "pharmacy_delay": "Pharmacy Delay",
}

DEPARTMENTS = {
    "Emergency": ["waiting_time", "emergency_cases", "patient_transfers"],
    "ICU": ["icu_occupancy", "nurse_patient_ratio", "discharge_delay"],
    "Diagnostics": ["diagnostic_turnaround", "equipment_availability", "equipment_downtime"],
    "Pharmacy": ["pharmacy_delay", "patient_complaints"],
    "General Ward": ["bed_occupancy", "staff_workload"],
}


def generate_timeline(seed: int = 0, n_days: int = 180) -> pd.DataFrame:
    """Generate a single synthetic hospital timeline."""
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)

    # Base seasonal pattern (weekly + gradual drift)
    weekly = np.sin(2 * np.pi * days / 7) * 0.08
    gradual_drift = np.linspace(0, 0.25, n_days)

    data = {}

    # --- Waiting Time ---
    base = 0.3 + gradual_drift * 0.5 + weekly * 0.1
    noise = rng.normal(0, 0.05, n_days)
    spikes = (rng.random(n_days) < 0.04) * rng.uniform(0.1, 0.3, n_days)
    data["waiting_time"] = np.clip(base + noise + spikes, 0, 1)

    # --- Staff Workload ---
    base = 0.5 + gradual_drift * 0.3 + weekly * 0.12
    noise = rng.normal(0, 0.04, n_days)
    data["staff_workload"] = np.clip(base + noise, 0, 1)

    # --- Bed Occupancy ---
    base = 0.6 + gradual_drift * 0.2 + weekly * 0.08
    noise = rng.normal(0, 0.04, n_days)
    data["bed_occupancy"] = np.clip(base + noise, 0, 1)

    # --- Equipment Availability ---
    base = 0.85 - gradual_drift * 0.3 - weekly * 0.05
    noise = rng.normal(0, 0.03, n_days)
    breakdowns = (rng.random(n_days) < 0.03) * rng.uniform(0.1, 0.25, n_days)
    data["equipment_availability"] = np.clip(base + noise - breakdowns, 0, 1)

    # --- Patient Transfers ---
    base = 0.25 + gradual_drift * 0.2 + weekly * 0.06
    noise = rng.normal(0, 0.04, n_days)
    data["patient_transfers"] = np.clip(base + noise, 0, 1)

    # --- Patient Complaints ---
    base = 0.15 + gradual_drift * 0.4 + weekly * 0.05
    noise = rng.normal(0, 0.03, n_days)
    data["patient_complaints"] = np.clip(base + noise, 0, 1)

    # --- Emergency Cases ---
    base = 0.4 + weekly * 0.15
    noise = rng.normal(0, 0.06, n_days)
    surges = (rng.random(n_days) < 0.05) * rng.uniform(0.15, 0.4, n_days)
    data["emergency_cases"] = np.clip(base + noise + surges, 0, 1)

    # --- ICU Occupancy ---
    base = 0.55 + gradual_drift * 0.25 + weekly * 0.07
    noise = rng.normal(0, 0.04, n_days)
    data["icu_occupancy"] = np.clip(base + noise, 0, 1)

    # --- Diagnostic Turnaround ---
    base = 0.3 + gradual_drift * 0.35 + weekly * 0.05
    noise = rng.normal(0, 0.04, n_days)
    data["diagnostic_turnaround"] = np.clip(base + noise, 0, 1)

    # --- Discharge Delay ---
    base = 0.2 + gradual_drift * 0.4 + weekly * 0.04
    noise = rng.normal(0, 0.03, n_days)
    data["discharge_delay"] = np.clip(base + noise, 0, 1)

    # --- Nurse-Patient Ratio ---
    base = 0.75 - gradual_drift * 0.3
    noise = rng.normal(0, 0.04, n_days)
    data["nurse_patient_ratio"] = np.clip(base + noise, 0, 1)

    # --- Equipment Downtime ---
    base = 0.1 + gradual_drift * 0.3 + weekly * 0.03
    noise = rng.normal(0, 0.03, n_days)
    events = (rng.random(n_days) < 0.04) * rng.uniform(0.1, 0.35, n_days)
    data["equipment_downtime"] = np.clip(base + noise + events, 0, 1)

    # --- Pharmacy Delay ---
    base = 0.2 + gradual_drift * 0.3 + weekly * 0.04
    noise = rng.normal(0, 0.03, n_days)
    data["pharmacy_delay"] = np.clip(base + noise, 0, 1)

    df = pd.DataFrame(data, index=days)
    df.index.name = "day"

    # Ground-truth health score (0=poor, 100=excellent)
    good_high = ["equipment_availability", "nurse_patient_ratio"]
    health_parts = []
    for feat in FEATURES:
        val = df[feat]
        if feat in good_high:
            health_parts.append(val)
        else:
            health_parts.append(1 - val)
    df["health_score"] = np.clip(np.mean(health_parts, axis=0) * 100, 0, 100)

    # Degradation labels
    for col_suffix, h_days in [("24h", 1), ("72h", 3), ("168h", 7), ("336h", 14)]:
        col = f"degraded_{col_suffix}"
        threshold = 50
        future_min = df["health_score"].rolling(window=h_days, min_periods=1).min().shift(-h_days)
        df[col] = (future_min < threshold).fillna(0).astype(int)

    df["seed"] = seed
    return df


def engineer_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling averages, trends, and interaction features for a single timeline."""
    result = df.copy()

    for feat in FEATURES:
        result[f"{feat}_roll7"] = result[feat].rolling(7, min_periods=1).mean()
        result[f"{feat}_trend"] = result[feat] - result[feat].shift(6).fillna(result[feat])

    # Interaction features
    result["wait_x_workload"] = result["waiting_time"] * result["staff_workload"]
    result["icu_x_nurse"] = result["icu_occupancy"] * (1 - result["nurse_patient_ratio"])
    result["equip_stress"] = result["equipment_downtime"] * (1 - result["equipment_availability"])
    result["pharmacy_complaint"] = result["pharmacy_delay"] * result["patient_complaints"]

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling averages, trends, and interaction features for multi-seed dataset."""
    result = df.copy()

    for feat in FEATURES:
        result[f"{feat}_roll7"] = result.groupby("seed")[feat].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        result[f"{feat}_trend"] = result.groupby("seed")[feat].transform(
            lambda x: x - x.shift(6).fillna(x)
        )

    result["wait_x_workload"] = result["waiting_time"] * result["staff_workload"]
    result["icu_x_nurse"] = result["icu_occupancy"] * (1 - result["nurse_patient_ratio"])
    result["equip_stress"] = result["equipment_downtime"] * (1 - result["equipment_availability"])
    result["pharmacy_complaint"] = result["pharmacy_delay"] * result["patient_complaints"]

    return result


def build_dataset(seeds, n_days=180):
    """Build multi-timeline dataset."""
    frames = [generate_timeline(s, n_days) for s in seeds]
    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    df = build_dataset(range(5))
    print(df.head())
    print(df.shape)
    print(df["health_score"].describe())
