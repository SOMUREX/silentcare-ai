# SilentCare AI 🏥

**An intelligent machine learning framework for detecting latent operational degradation and predicting pre-failure states in hospital systems.**

> Department of Artificial Intelligence and Machine Learning  
> Charan V · Aswin Kumar K · Someshwar J  
> Academic Year 2026–2027

---

## 🔬 What It Does

SilentCare AI monitors 13 hospital operational indicators simultaneously and detects hidden degradation patterns before they become critical — something traditional threshold-based monitoring misses.

| Component | Model | Purpose |
|-----------|-------|---------|
| Health Estimation | Random Forest Regressor | Operational health score (0–100) |
| Risk Prediction | Random Forest Classifier | Degradation probability at 24h, 72h, 7d, 14d |
| Anomaly Detection | Isolation Forest | Flag unusual indicator combinations |
| Feature Engineering | Rolling averages + trends | 7-obs rolling mean, trend, interactions |

## 📊 13 Operational Indicators

Waiting Time · Staff Workload · Bed Occupancy · Equipment Availability · Patient Transfers · Patient Complaints · Emergency Cases · ICU Occupancy · Diagnostic Turnaround · Discharge Delay · Nurse-Patient Ratio · Equipment Downtime · Pharmacy Delay

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models + start server
python run.py

# Or train separately, then start
python train.py
python app.py
```

The dashboard opens at **http://localhost:5000**

## 🏗️ Project Structure

```
silentcare-ai/
├── data_generator.py   # Synthetic hospital time-series generator
├── train.py            # ML model training (RF Regressor, RF Classifier, Isolation Forest)
├── app.py              # Flask web application + REST API
├── run.py              # One-click startup script
├── requirements.txt
├── Procfile            # Render/Heroku deployment
├── models/             # Saved model files (generated after training)
├── static/
│   ├── css/style.css   # Dark theme dashboard styles
│   └── js/dashboard.js # Chart.js visualizations + API client
└── templates/
    └── index.html      # Dashboard HTML
```

## 🌐 Live Demo

Deployed on **Render**: [SilentCare AI Dashboard](https://silentcare-ai.onrender.com)

## 📈 ML Model Details

- **Random Forest Regressor**: 250 estimators, max depth 14, min 3 samples/leaf
- **Random Forest Classifier**: 250 estimators, max depth 12, balanced_subsample weights
- **Isolation Forest**: 200 estimators, contamination 0.06
- **Dataset**: 180 simulated days × 40 training seeds + 10 test seeds

## ⚠️ Disclaimer

This system uses **synthetic data only**. No real hospital measurements or patient records are included. Not clinically validated.

## 📚 References

- Breiman (2001). Random Forests. *Machine Learning, 45*, 5–32.
- Liu et al. (2008). Isolation Forest. *IEEE ICDM*, 413–422.
- Pedregosa et al. (2011). Scikit-learn. *JMLR, 12*, 2825–2830.
