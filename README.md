# SilentCare AI 🏥

**An intelligent machine learning framework for detecting latent operational degradation and predicting pre-failure states in hospital systems.**

> Department of Artificial Intelligence and Machine Learning  
> **Charan V · Aswin Kumar K · Someshwar J**  
> III AIML C · Academic Year 2026–2027

---

## 🔬 Project Overview

SilentCare AI monitors 13 cross-departmental hospital operational indicators simultaneously to detect latent operational degradation patterns before they manifest as critical failures. Unlike traditional threshold-based alerting which responds reactively to single-metric breaches, SilentCare AI provides multi-horizon degradation risk forecasting and unsupervised anomaly detection across interconnected clinical and operational domains.

| Component | Architecture / Algorithm | Purpose | Metric Achieved |
|-----------|--------------------------|---------|-----------------|
| **Operational Health Estimation** | Random Forest Regressor (250 trees, max depth 14) | Continuous Health Score (0–100) | **$R^2 = 0.9914$**, MAE = 0.17 |
| **Multi-Horizon Risk Prediction** | Random Forest Classifier (250 trees, balanced) | Degradation probability at 24h, 72h, 7d, 14d | **97%–100% Accuracy / F1-Score** |
| **Anomaly Detection** | Isolation Forest (200 trees, contamination 0.06) | Unsupervised multidimensional anomaly flags | Flags subtle co-occurring anomalies |
| **Feature Engineering** | 7-day rolling means, 6-day trend momentum, interaction cross-products | Capture degradation trajectory and inter-department stress | 43 engineered features |

---

## 📂 Real Hospital Datasets & Integration

The system ingests and unifies 12 real clinical and healthcare operational data sources:

1. **MIMIC-IV Clinical Database (v2.2)**: Inpatient admissions, ICU stays, transfers, and pharmacy verification timelines.
2. **MIMIC-IV Emergency Department (ED v2.2)**: Emergency room visits, triage acuity scores, and length of stay.
3. **Hospital Resources**: Census dates, bed occupancy rates, ICU capacity, PACU utilization, diversion flags.
4. **Staffing Operations**: Shift logs, staff-to-patient ratios, overtime flags, float pool, and agency reliance.
5. **Medical Equipment Management**: Asset maintenance logs, utilization, unplanned downtime hours, repair costs.
6. **Operating Room Schedules**: Surgical volumes, case turnover times, start delays, and cancellation rates.
7. **NHS England Monthly A&E Time Series (Oct 2026)**: Historical emergency volume and >4h / >12h waiting times.
8. **NHS England Monthly Diagnostics Timeseries (Sep 2026)**: Diagnostic waiting list sizes and backlog dynamics.
9. **NHS Discharge-Ready Date Webfile (Sep 2026)**: Timeliness of acute hospital discharges and delayed transfer days.
10. **NHS Written Complaints Tables (2025–26)**: Patient complaint trends across hospital and community services.
11. **NHS Organisations England Overnight (Q1 2026–27)**: Organization metadata.
12. **Facilities Master Record**: Multi-facility capacities, trauma tiers, and surgical suites.

---

## 📊 13 Monitored Operational Indicators

1. **Waiting Time** (ED boarding + A&E wait + OR start delays)
2. **Staff Workload** (Overtime, agency staffing, inverted adequacy)
3. **Bed Occupancy** (Inpatient census + daily admissions)
4. **Equipment Availability** (Active in-service proportion)
5. **Patient Transfers** (Inter-unit transfers and urgent admissions)
6. **Patient Complaints** (Official patient grievance rates)
7. **Emergency Cases** (A&E volume and urgent escalations)
8. **ICU Occupancy** (Intensive care census and ICU LOS)
9. **Diagnostic Turnaround** (Lab/imaging queues and backlog delays)
10. **Discharge Delay** (Post-ready discharge lag)
11. **Nurse-to-Patient Ratio** (Staffing safety threshold)
12. **Equipment Downtime** (Unplanned failure and maintenance duration)
13. **Pharmacy Delay** (Order-to-verification latency)

---

## 🚀 Quick Start (Running Locally)

```bash
# 1. Clone repository
git clone https://github.com/SOMUREX/silentcare-ai.git
cd silentcare-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch SilentCare AI (auto-trains on first launch)
python run.py
```

The interactive dashboard will be live at: **http://localhost:5000**

---

## 🏗️ Repository Structure

```
silentcare-ai/
├── data_generator.py      # Data pipeline: in-memory aggregation of MIMIC-IV, NHS & operational data
├── train.py               # Model training script (RF Regressor, RF Classifiers, Isolation Forest)
├── app.py                 # Flask web backend & REST API endpoints
├── run.py                 # Self-healing start script
├── requirements.txt       # Production dependencies
├── Procfile               # Cloud deployment (Render / Heroku)
├── render.yaml            # Render Blueprint spec
├── data/                  # Processed feature matrix
├── static/
│   ├── css/style.css      # Dark mode responsive healthcare dashboard styling
│   └── js/dashboard.js    # Chart.js visualizations & real-time telemetry updates
└── templates/
    └── index.html         # Live monitoring dashboard UI
```

---

## 🌐 Live Web Deployment

SilentCare AI is configured for instant 1-click cloud deployment on **Render**:
- **Build Command**: `pip install -r requirements.txt && python train.py`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`

---

## 👥 Contributors

- **Charan V** (RA2411026050149)
- **Aswin Kumar K** (RA2411026050186)
- **Someshwar J** (RA2411026050204)  
*Department of Artificial Intelligence and Machine Learning*
