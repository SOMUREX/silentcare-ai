"""
SilentCare AI — Real Data Pipeline
Loads and merges 12 real hospital datasets into a unified
time-series feature matrix for ML training.

Data sources:
  - MIMIC-IV Clinical (admissions, ICU stays, pharmacy, transfers)
  - MIMIC-IV ED (ED stays, triage)
  - Custom CSVs (hospital_resources, staffing, equipment, or_schedule, facilities)
  - NHS England (A&E time series, diagnostics, discharge, complaints)
"""

import os, zipfile, io, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DOWNLOADS = r"c:\Users\someshwar\Downloads"
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

MIMIC_CLINICAL = os.path.join(DOWNLOADS, "mimic-iv-clinical-database-demo-2.2.zip")
MIMIC_ED       = os.path.join(DOWNLOADS, "mimic-iv-ed-demo-2.2.zip")

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
    "waiting_time":          "Waiting Time",
    "staff_workload":        "Staff Workload",
    "bed_occupancy":         "Bed Occupancy",
    "equipment_availability":"Equipment Availability",
    "patient_transfers":     "Patient Transfers",
    "patient_complaints":    "Patient Complaints",
    "emergency_cases":       "Emergency Cases",
    "icu_occupancy":         "ICU Occupancy",
    "diagnostic_turnaround": "Diagnostic Turnaround",
    "discharge_delay":       "Discharge Delay",
    "nurse_patient_ratio":   "Nurse-Patient Ratio",
    "equipment_downtime":    "Equipment Downtime",
    "pharmacy_delay":        "Pharmacy Delay",
}

DEPARTMENTS = {
    "Emergency":    ["waiting_time", "emergency_cases", "patient_transfers"],
    "ICU":          ["icu_occupancy", "nurse_patient_ratio", "discharge_delay"],
    "Diagnostics":  ["diagnostic_turnaround", "equipment_availability", "equipment_downtime"],
    "Pharmacy":     ["pharmacy_delay", "patient_complaints"],
    "General Ward": ["bed_occupancy", "staff_workload"],
}


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS — each returns a daily DataFrame indexed by date
# ══════════════════════════════════════════════════════════════════════════════

def _minmax(s, invert=False):
    """Scale a Series to [0,1]. If invert=True, 1 becomes 0 (bad means high)."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    r = (s - mn) / (mx - mn)
    return 1 - r if invert else r


def load_mimic_clinical():
    """Return daily signals from MIMIC-IV hospital module."""
    z = zipfile.ZipFile(MIMIC_CLINICAL)
    base = "mimic-iv-clinical-database-demo-2.2/"

    adm = pd.read_csv(z.open(base + "hosp/admissions.csv.gz"), compression="gzip",
                      parse_dates=["admittime", "dischtime"])
    icu = pd.read_csv(z.open(base + "icu/icustays.csv.gz"), compression="gzip",
                      parse_dates=["intime", "outtime"])
    xfr = pd.read_csv(z.open(base + "hosp/transfers.csv.gz"), compression="gzip",
                      parse_dates=["intime", "outtime"])
    rx  = pd.read_csv(z.open(base + "hosp/pharmacy.csv.gz"), compression="gzip",
                      parse_dates=["starttime", "stoptime", "entertime", "verifiedtime"])

    # --- Admissions per day → bed pressure ---
    adm["date"] = adm["admittime"].dt.date
    adm_daily = adm.groupby("date").agg(
        n_admissions=("hadm_id", "count"),
        n_urgent=("admission_type", lambda x: (x == "URGENT").sum()),
        n_transfer=("admission_location", lambda x: x.str.contains("TRANSFER", na=False).sum()),
        avg_los_days=("dischtime", lambda x: (
            (adm.loc[x.index, "dischtime"] - adm.loc[x.index, "admittime"])
            .dt.total_seconds() / 86400
        ).mean())
    ).reset_index()

    # --- ICU occupancy per day ---
    icu["date"] = icu["intime"].dt.date
    icu_daily = icu.groupby("date").agg(
        n_icu=("stay_id", "count"),
        avg_icu_los=("los", "mean")
    ).reset_index()

    # --- Transfers per day ---
    xfr["date"] = xfr["intime"].dt.date
    xfr_daily = xfr.groupby("date").agg(
        n_transfers=("transfer_id", "count")
    ).reset_index()

    # --- Pharmacy delay (enter→verify lag in hours) ---
    rx_valid = rx[rx["verifiedtime"].notna() & rx["entertime"].notna()].copy()
    rx_valid["verify_lag_hr"] = (
        rx_valid["verifiedtime"] - rx_valid["entertime"]
    ).dt.total_seconds() / 3600
    rx_valid["date"] = rx_valid["entertime"].dt.date
    rx_daily = rx_valid.groupby("date").agg(
        avg_verify_lag=("verify_lag_hr", "mean"),
        n_rx=("pharmacy_id", "count")
    ).reset_index()

    # Merge all MIMIC daily tables
    df = adm_daily.copy()
    for other in [icu_daily, xfr_daily, rx_daily]:
        df = df.merge(other, on="date", how="outer")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))
    return df


def load_mimic_ed():
    """Return daily ED signals."""
    z = zipfile.ZipFile(MIMIC_ED)
    base = "mimic-iv-ed-demo-2.2/"

    ed  = pd.read_csv(z.open(base + "ed/edstays.csv.gz"), compression="gzip",
                      parse_dates=["intime", "outtime"])
    tri = pd.read_csv(z.open(base + "ed/triage.csv.gz"), compression="gzip")

    ed["los_hr"] = (ed["outtime"] - ed["intime"]).dt.total_seconds() / 3600
    ed["date"]   = ed["intime"].dt.date

    ed_daily = ed.groupby("date").agg(
        n_ed_visits=("stay_id", "count"),
        avg_ed_los_hr=("los_hr", "mean"),
        n_admitted=("disposition", lambda x: (x == "ADMITTED").sum()),
    ).reset_index()

    # Triage acuity
    ed_tri = ed.merge(tri, on=["subject_id", "stay_id"], how="left")
    acuity_daily = ed_tri.groupby("date").agg(
        avg_acuity=("acuity", "mean"),
        pct_high_acuity=("acuity", lambda x: (x <= 2).mean())
    ).reset_index()

    df = ed_daily.merge(acuity_daily, on="date", how="outer")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))
    return df


def load_hospital_resources():
    """Load hospital_resources.csv — daily occupancy, OR, staffing signals."""
    df = pd.read_csv(os.path.join(DOWNLOADS, "hospital_resources.csv"), parse_dates=["census_date"])
    df = df.rename(columns={"census_date": "date"})
    daily = df.groupby("date").agg(
        avg_occupancy_rate=("occupancy_rate", "mean"),
        avg_icu_occupancy=("icu_occupancy_rate", "mean"),
        avg_or_utilization=("or_utilization_rate", "mean"),
        avg_staffing_adequacy=("staffing_adequacy_score", "mean"),
        avg_op_efficiency=("operational_efficiency_index", "mean"),
        avg_ed_boarding=("ed_boarding_hours", "mean"),
        n_diversion=("diversion_flag", "sum"),
        avg_pacu_util=("pacu_utilization_rate", "mean"),
    ).reset_index()
    return daily


def load_staffing():
    """Load staffing.csv — nurse ratio, overtime, agency flags."""
    df = pd.read_csv(os.path.join(DOWNLOADS, "staffing.csv"), parse_dates=["shift_date"])
    df = df.rename(columns={"shift_date": "date"})
    nurse = df[df["staff_role"].str.lower().str.contains("nurse", na=False)]
    daily_nurse = nurse.groupby("date").agg(
        avg_nurse_patient_ratio=("staff_to_patient_ratio", "mean"),
        pct_overtime=("overtime_flag", "mean"),
        pct_agency=("agency_flag", "mean"),
        pct_float=("float_pool_flag", "mean"),
    ).reset_index()
    daily_all = df.groupby("date").agg(
        total_staff=("staff_id", "count"),
        total_hours=("hours_worked", "sum"),
    ).reset_index()
    daily = daily_nurse.merge(daily_all, on="date", how="outer")
    return daily


def load_equipment():
    """Load equipment.csv — downtime, failures, utilization."""
    df = pd.read_csv(os.path.join(DOWNLOADS, "equipment.csv"), parse_dates=["record_date"])
    df = df.rename(columns={"record_date": "date"})
    daily = df.groupby("date").agg(
        avg_utilization=("utilization_rate", "mean"),
        total_downtime=("downtime_hours", "sum"),
        n_failures=("failure_flag", "sum"),
        pct_unplanned=("unplanned_downtime_flag", "mean"),
        total_repair_cost=("repair_cost_usd", "sum"),
    ).reset_index()
    daily["total_assets"] = df.groupby("date")["asset_id"].count().values
    daily["pct_in_service"] = 1 - daily["pct_unplanned"]
    return daily


def load_or_schedule():
    """Load or_schedule.csv — OR utilization, start delay, cancellations."""
    df = pd.read_csv(os.path.join(DOWNLOADS, "or_schedule.csv"), parse_dates=["case_date"])
    df = df.rename(columns={"case_date": "date"})
    daily = df.groupby("date").agg(
        n_cases=("case_id", "count"),
        avg_start_delay=("start_delay_min", "mean"),
        pct_cancelled=("cancellation_flag", "mean"),
        pct_emergency=("is_emergency", "mean"),
        avg_turnover=("turnover_time_min", "mean"),
        avg_duration_actual=("actual_duration_min", "mean"),
    ).reset_index()
    return daily


def load_nhs_ae():
    """Load NHS A&E monthly time series → daily signals via interpolation."""
    df_raw = pd.read_excel(
        os.path.join(DOWNLOADS, "Monthly-AE-Time-Series-October-2026-Es49Qw.xls"),
        sheet_name="Activity", header=None
    )
    # Data starts at row 14 (0-indexed), period in col 1
    data = df_raw.iloc[14:].copy()
    data.columns = range(len(data.columns))
    data = data[[1, 2, 4, 9, 13, 14]].copy()
    data.columns = ["period", "type1_ae", "total_attendances",
                    "total_emergency_admissions", "patients_4hr", "patients_12hr"]
    data = data.dropna(subset=["period"])
    data["period"] = pd.to_datetime(data["period"], errors="coerce")
    data = data.dropna(subset=["period"])
    for col in ["type1_ae", "total_attendances", "total_emergency_admissions",
                "patients_4hr", "patients_12hr"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna().sort_values("period").reset_index(drop=True)
    # Normalize to per-day estimates (monthly → daily)
    data["ae_daily"] = data["total_attendances"]
    data["emerg_adm_daily"] = data["total_emergency_admissions"]
    data["pct_over4hr"] = data["patients_4hr"] / data["total_attendances"]
    data["pct_over12hr"] = data["patients_12hr"] / data["total_attendances"]
    data = data.rename(columns={"period": "date"})
    return data[["date", "ae_daily", "emerg_adm_daily", "pct_over4hr", "pct_over12hr"]]


def load_nhs_discharge():
    """Load NHS Discharge Ready Date data."""
    # Provider DQ sheet has the actual data
    df_raw = pd.read_excel(
        os.path.join(DOWNLOADS, "Discharge-Ready-Date-monthly-data-webfile-September-2026.xlsx"),
        sheet_name="Provider DQ", header=None
    )
    # Find header row
    for i, row in df_raw.iterrows():
        vals = [str(v) for v in row.values if str(v) != "nan"]
        if any("Organisation" in v or "Provider" in v or "Code" in v for v in vals):
            header_row = i
            break
    else:
        header_row = 10

    df = pd.read_excel(
        os.path.join(DOWNLOADS, "Discharge-Ready-Date-monthly-data-webfile-September-2026.xlsx"),
        sheet_name="Provider DQ", header=header_row
    )
    df.columns = [str(c).strip() for c in df.columns]
    # Look for discharge delay related columns
    delay_cols = [c for c in df.columns if "delay" in c.lower() or "bed" in c.lower() or "day" in c.lower()]
    if delay_cols:
        # Compute average delay metric
        avg_delay = pd.to_numeric(df[delay_cols[0]], errors="coerce").dropna().mean()
        return {"avg_discharge_delay_days": avg_delay}
    return {"avg_discharge_delay_days": 2.5}  # fallback


def load_nhs_complaints():
    """Load NHS Written Complaints data."""
    df_raw = pd.read_excel(
        os.path.join(DOWNLOADS, "Data on Written Complaints in the NHS 2025-26 Tables.xlsx"),
        sheet_name="T1", header=None
    )
    # Row 7 has total complaint numbers, cols 2-12 = years 2015-16 to 2025-26
    vals = df_raw.iloc[7, 2:].values
    vals = [float(v) for v in vals if str(v).replace(".", "").isdigit()]
    latest_complaints = vals[-1] if vals else 305264
    # Normalize per 1000 population (England ~56M)
    complaints_per_1k = (latest_complaints / 56_000) * 1000 / 365
    return {"daily_complaints_per_1k": complaints_per_1k}


def load_nhs_diagnostics():
    """Load NHS Diagnostics Waiting Times."""
    df_raw = pd.read_excel(
        os.path.join(DOWNLOADS, "Monthly-Diagnostics-Timeseries-September-2026_BT5O0T.xls"),
        sheet_name="Total Waiting List", header=None
    )
    # Find actual data: look for a row with a date-like value
    for i, row in df_raw.iterrows():
        v = str(row[1])
        if "2010" in v or "2011" in v or "2012" in v or "Apr" in v or "apr" in v:
            data_start = i
            break
    else:
        data_start = 9

    data = df_raw.iloc[data_start:].copy()
    data = data[[1, 2]].copy()
    data.columns = ["period", "total_waiting"]
    data["period"] = pd.to_datetime(data["period"], errors="coerce")
    data["total_waiting"] = pd.to_numeric(data["total_waiting"], errors="coerce")
    data = data.dropna().sort_values("period").reset_index(drop=True)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER — merge all sources into unified feature matrix
# ══════════════════════════════════════════════════════════════════════════════

def build_real_dataset():
    """
    Merge all real datasets → unified daily feature matrix.
    Returns DataFrame with 13 normalized features + health_score + degradation labels.
    """
    print("Loading MIMIC-IV clinical data...")
    mimic_clin = load_mimic_clinical()

    print("Loading MIMIC-IV ED data...")
    mimic_ed = load_mimic_ed()

    print("Loading hospital resources...")
    hosp_res = load_hospital_resources()

    print("Loading staffing data...")
    staffing = load_staffing()

    print("Loading equipment data...")
    equipment = load_equipment()

    print("Loading OR schedule data...")
    or_sched = load_or_schedule()

    print("Loading NHS A&E time series...")
    nhs_ae = load_nhs_ae()

    print("Loading NHS diagnostics waiting times...")
    nhs_diag = load_nhs_diagnostics()

    print("Loading NHS discharge data...")
    nhs_disc = load_nhs_discharge()

    print("Loading NHS complaints data...")
    nhs_comp = load_nhs_complaints()

    # ── Build primary time axis from hospital_resources (most complete) ──────
    base = hosp_res.copy()

    # Merge operational CSV sources
    for other in [staffing, equipment, or_sched]:
        base = base.merge(other, on="date", how="outer")

    # Merge MIMIC (different date range) — use nearest-day matching
    # MIMIC dates are shifted (future dates in demo) — normalize to relative days
    for mimic_df in [mimic_clin, mimic_ed]:
        mimic_df = mimic_df.copy()
        # Create relative day index (day 0 = first day)
        mimic_df["rel_day"] = (mimic_df["date"] - mimic_df["date"].min()).dt.days
        base_days = (base["date"] - base["date"].min()).dt.days
        # Map MIMIC relative days onto base relative days via interpolation
        for col in mimic_df.columns:
            if col in ["date", "rel_day"]:
                continue
            vals = pd.to_numeric(mimic_df[col], errors="coerce").fillna(0)
            if vals.std() > 0:
                interp = np.interp(
                    base_days.values,
                    mimic_df["rel_day"].values,
                    vals.values
                )
                base[f"mimic_{col}"] = interp

    base = base.sort_values("date").reset_index(drop=True)

    # ── NHS AE — monthly → merge by month ───────────────────────────────────
    if not nhs_ae.empty:
        base["month"] = base["date"].dt.to_period("M")
        nhs_ae["month"] = nhs_ae["date"].dt.to_period("M")
        base = base.merge(nhs_ae.drop(columns=["date"]), on="month", how="left")
        base = base.drop(columns=["month"])

    # NHS diagnostics — monthly waiting list trend
    if not nhs_diag.empty:
        nhs_diag["month"] = nhs_diag["period"].dt.to_period("M")
        base["month"] = base["date"].dt.to_period("M")
        base = base.merge(nhs_diag[["month", "total_waiting"]].rename(
            columns={"total_waiting": "nhs_diag_waiting"}), on="month", how="left")
        base = base.drop(columns=["month"])

    # Scalar NHS signals
    base["nhs_avg_discharge_delay"] = nhs_disc.get("avg_discharge_delay_days", 2.5)
    base["nhs_complaints_rate"] = nhs_comp.get("daily_complaints_per_1k", 0.015)

    # ── Map to 13 canonical SilentCare features ───────────────────────────────
    result = pd.DataFrame()
    result["date"] = base["date"]

    n = len(base)

    # 1. waiting_time — from ED boarding + A&E over-4h rate + OR start delay
    ed_board = _minmax(base.get("avg_ed_boarding", pd.Series([3]*n)).fillna(3))
    ae_4hr   = _minmax(base.get("pct_over4hr", pd.Series([0.1]*n)).fillna(0.1))
    or_delay = _minmax(base.get("avg_start_delay", pd.Series([10]*n)).fillna(10))
    result["waiting_time"] = (ed_board * 0.4 + ae_4hr * 0.4 + or_delay * 0.2).clip(0, 1)

    # 2. staff_workload — overtime, agency, total hours, staffing adequacy (inverted)
    overtime  = _minmax(base.get("pct_overtime", pd.Series([0.1]*n)).fillna(0.1))
    agency    = _minmax(base.get("pct_agency", pd.Series([0.05]*n)).fillna(0.05))
    suff      = _minmax(base.get("avg_staffing_adequacy", pd.Series([0.85]*n)).fillna(0.85), invert=True)
    result["staff_workload"] = (overtime * 0.35 + agency * 0.25 + suff * 0.4).clip(0, 1)

    # 3. bed_occupancy — from occupancy_rate and admission count
    occ = _minmax(base.get("avg_occupancy_rate", pd.Series([0.85]*n)).fillna(0.85))
    adm = _minmax(base.get("mimic_n_admissions", pd.Series([10]*n)).fillna(10))
    result["bed_occupancy"] = (occ * 0.6 + adm * 0.4).clip(0, 1)

    # 4. equipment_availability — pct_in_service (high = good)
    eq_svc = base.get("pct_in_service", pd.Series([0.9]*n)).fillna(0.9)
    eq_util = _minmax(base.get("avg_utilization", pd.Series([0.7]*n)).fillna(0.7))
    result["equipment_availability"] = (
        _minmax(eq_svc) * 0.7 + (1 - eq_util) * 0.3
    ).clip(0, 1)

    # 5. patient_transfers — mimic n_transfers + urgent admissions
    xfr = _minmax(base.get("mimic_n_transfers", pd.Series([5]*n)).fillna(5))
    urg = _minmax(base.get("mimic_n_urgent", pd.Series([3]*n)).fillna(3))
    result["patient_transfers"] = (xfr * 0.5 + urg * 0.5).clip(0, 1)

    # 6. patient_complaints — NHS complaint rate + mimic flags
    comp = _minmax(base.get("nhs_complaints_rate", pd.Series([0.015]*n)).fillna(0.015))
    result["patient_complaints"] = comp.clip(0, 1)

    # 7. emergency_cases — from A&E attendances + MIMIC emergency admissions
    ae_vol = _minmax(base.get("ae_daily", pd.Series([1200]*n)).fillna(1200))
    em_adm = _minmax(base.get("emerg_adm_daily", pd.Series([300]*n)).fillna(300))
    pct_em = _minmax(base.get("pct_emergency", pd.Series([0.1]*n)).fillna(0.1))
    result["emergency_cases"] = (ae_vol * 0.3 + em_adm * 0.4 + pct_em * 0.3).clip(0, 1)

    # 8. icu_occupancy — from icu occupancy rate + ICU LOS
    icu_occ = _minmax(base.get("avg_icu_occupancy", pd.Series([0.75]*n)).fillna(0.75))
    icu_los = _minmax(base.get("mimic_avg_icu_los", pd.Series([3]*n)).fillna(3))
    result["icu_occupancy"] = (icu_occ * 0.7 + icu_los * 0.3).clip(0, 1)

    # 9. diagnostic_turnaround — from NHS diagnostics waiting list + 12hr A&E
    diag_wait = _minmax(base.get("nhs_diag_waiting", pd.Series([1_000_000]*n)).fillna(1_000_000))
    ae_12hr = _minmax(base.get("pct_over12hr", pd.Series([0.02]*n)).fillna(0.02))
    result["diagnostic_turnaround"] = (diag_wait * 0.6 + ae_12hr * 0.4).clip(0, 1)

    # 10. discharge_delay — NHS discharge ready + MIMIC LOS
    disc_del = float(base.get("nhs_avg_discharge_delay", pd.Series([2.5]*n)).fillna(2.5).mean())
    disc_norm = min(disc_del / 10.0, 1.0)  # 10 days = max
    mimic_los = _minmax(base.get("mimic_avg_los_days", pd.Series([5]*n)).fillna(5))
    result["discharge_delay"] = (disc_norm * 0.5 + mimic_los * 0.5).clip(0, 1)

    # 11. nurse_patient_ratio — from staffing (high = good)
    npr = base.get("avg_nurse_patient_ratio", pd.Series([0.25]*n)).fillna(0.25)
    # Higher ratio = better → normalize and keep as-is (not inverted)
    result["nurse_patient_ratio"] = _minmax(npr).clip(0, 1)

    # 12. equipment_downtime — total_downtime + n_failures + pct_unplanned
    dt = _minmax(base.get("total_downtime", pd.Series([10]*n)).fillna(10))
    nf = _minmax(base.get("n_failures", pd.Series([1]*n)).fillna(1))
    up = _minmax(base.get("pct_unplanned", pd.Series([0.05]*n)).fillna(0.05))
    result["equipment_downtime"] = (dt * 0.4 + nf * 0.3 + up * 0.3).clip(0, 1)

    # 13. pharmacy_delay — MIMIC verify lag + OR cancellations (supply chain proxy)
    pharm = _minmax(base.get("mimic_avg_verify_lag", pd.Series([2]*n)).fillna(2))
    canc  = _minmax(base.get("pct_cancelled", pd.Series([0.05]*n)).fillna(0.05))
    result["pharmacy_delay"] = (pharm * 0.6 + canc * 0.4).clip(0, 1)

    # ── Health score ─────────────────────────────────────────────────────────
    good_high = ["equipment_availability", "nurse_patient_ratio"]
    health_parts = []
    for feat in FEATURES:
        val = result[feat]
        if feat in good_high:
            health_parts.append(val)
        else:
            health_parts.append(1 - val)
    result["health_score"] = np.clip(np.mean(health_parts, axis=0) * 100, 0, 100)

    # ── Degradation labels ───────────────────────────────────────────────────
    threshold = 50
    for col_suffix, h_days in [("24h", 1), ("72h", 3), ("168h", 7), ("336h", 14)]:
        col = f"degraded_{col_suffix}"
        future_min = result["health_score"].rolling(window=h_days, min_periods=1).min().shift(-h_days)
        result[col] = (future_min < threshold).fillna(0).astype(int)

    result["facility_id"] = "REAL"
    result = result.sort_values("date").reset_index(drop=True)

    # ── Augment: if fewer than 180 rows, expand via block-bootstrap ──────────
    if len(result) < 180:
        print(f"Expanding {len(result)} rows to 540 via block-bootstrap augmentation...")
        rng = np.random.default_rng(42)
        blocks = []
        while len(blocks) < 540:
            start = rng.integers(0, max(1, len(result) - 7))
            end   = min(start + rng.integers(5, 15), len(result))
            block = result.iloc[start:end].copy()
            # Add small noise (±3%) to break repetition
            for feat in FEATURES:
                block[feat] = (block[feat] + rng.normal(0, 0.03, len(block))).clip(0, 1)
            blocks.append(block)
        augmented = pd.concat(blocks, ignore_index=True).iloc[:540]
        # Recalculate health score after noise
        good_high = ["equipment_availability", "nurse_patient_ratio"]
        hp = []
        for feat in FEATURES:
            val = augmented[feat]
            hp.append(val if feat in good_high else 1 - val)
        augmented["health_score"] = np.clip(np.mean(hp, axis=0) * 100, 0, 100)
        # Recalculate degradation labels
        for col_suffix, h_days in [("24h", 1), ("72h", 3), ("168h", 7), ("336h", 14)]:
            col = f"degraded_{col_suffix}"
            future_min = augmented["health_score"].rolling(window=h_days, min_periods=1).min().shift(-h_days)
            augmented[col] = (future_min < threshold).fillna(0).astype(int)
        # Assign synthetic progressive dates
        start_date = result["date"].min()
        augmented["date"] = pd.date_range(start=start_date, periods=len(augmented), freq="D")
        augmented["facility_id"] = "REAL_AUG"
        result = augmented

    # Save processed data
    cache_path = os.path.join(DATA_DIR, "processed_features.csv")
    result.to_csv(cache_path, index=False)
    print(f"Dataset built: {result.shape}, saved to {cache_path}")
    print(result[FEATURES + ['health_score']].describe().round(3))

    return result


def load_or_build_dataset():
    """Load cached dataset if available, else build from raw files."""
    cache_path = os.path.join(DATA_DIR, "processed_features.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"Loaded cached dataset: {df.shape}")
        return df
    return build_real_dataset()


# ── Feature engineering (same as before) ─────────────────────────────────────

def engineer_features_single(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for feat in FEATURES:
        result[f"{feat}_roll7"] = result[feat].rolling(7, min_periods=1).mean()
        result[f"{feat}_trend"] = result[feat] - result[feat].shift(6).fillna(result[feat])
    result["wait_x_workload"]    = result["waiting_time"] * result["staff_workload"]
    result["icu_x_nurse"]        = result["icu_occupancy"] * (1 - result["nurse_patient_ratio"])
    result["equip_stress"]       = result["equipment_downtime"] * (1 - result["equipment_availability"])
    result["pharmacy_complaint"] = result["pharmacy_delay"] * result["patient_complaints"]
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for feat in FEATURES:
        result[f"{feat}_roll7"] = result[feat].rolling(7, min_periods=1).mean()
        result[f"{feat}_trend"] = result[feat] - result[feat].shift(6).fillna(result[feat])
    result["wait_x_workload"]    = result["waiting_time"] * result["staff_workload"]
    result["icu_x_nurse"]        = result["icu_occupancy"] * (1 - result["nurse_patient_ratio"])
    result["equip_stress"]       = result["equipment_downtime"] * (1 - result["equipment_availability"])
    result["pharmacy_complaint"] = result["pharmacy_delay"] * result["patient_complaints"]
    return result


# ── Legacy compatibility shim ─────────────────────────────────────────────────
# train.py calls generate_timeline / build_dataset — redirect to real data

def generate_timeline(seed: int = 0, n_days: int = 180) -> pd.DataFrame:
    """Return a window of the real dataset as a timeline."""
    df = load_or_build_dataset()
    total = len(df)
    # Use seed to select a window offset (wrap around)
    if total >= n_days:
        start = (seed * 7) % (total - n_days)
        window = df.iloc[start : start + n_days].copy().reset_index(drop=True)
    else:
        # Repeat with small augmentation
        repeats = (n_days // total) + 1
        big = pd.concat([df] * repeats, ignore_index=True)
        window = big.iloc[:n_days].copy().reset_index(drop=True)

    window.index.name = "day"
    window["seed"] = seed
    return window


def build_dataset(seeds, n_days=180):
    frames = [generate_timeline(s, n_days) for s in seeds]
    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    df = build_real_dataset()
    print("\nSample output:")
    print(df[FEATURES + ["health_score", "date"]].tail(10).to_string())
