"""
SilentCare AI - Startup script
Trains models if not found, then starts the Flask server.
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def models_exist():
    required = ["rfr.pkl", "iso.pkl", "scaler.pkl", "feature_cols.pkl",
                "rfc_24h.pkl", "rfc_72h.pkl", "rfc_168h.pkl", "rfc_336h.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)

if __name__ == "__main__":
    if not models_exist():
        print("Models not found. Training now (this takes ~1-2 minutes)...")
        from train import train_models
        train_models()
        print("Training complete!")
    else:
        print("Models found. Skipping training.")

    from app import app
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting SilentCare AI on http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)

