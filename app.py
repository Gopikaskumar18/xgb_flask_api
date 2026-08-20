"""
Flask API for demand forecasting, with a /health endpoint and latency logging
for production monitoring.
"""
import time
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("demand-api")

app = Flask(__name__)

# ---- load artifacts (produced by train.py) ----
model = xgb.XGBRegressor()
model.load_model("final_xgb_model.json")
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("feature_medians.pkl", "rb") as f:
    medians = pickle.load(f)

FEATURE_COLS = list(medians.index)
CAT_COLS = list(label_encoders.keys())


def prepare(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).map(label_encoders[col]).fillna(-1).astype(int)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_COLS].fillna(medians)


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe for load balancers / App Runner health checks."""
    return jsonify({"status": "healthy", "model_loaded": True}), 200


@app.route("/", methods=["GET"])
def home():
    return "Demand Forecasting API. POST feature JSON to /predict."


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    try:
        data = request.get_json(force=True)
        if not data or "features" not in data:
            return jsonify({"error": "'features' key missing"}), 400
        X = prepare(data["features"])
        pred = max(0.0, float(model.predict(X)[0]))
        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(f"prediction={pred:.2f} latency_ms={latency_ms:.1f}")
        return jsonify({
            "predicted_demand": round(pred, 2),
            "predicted_units": int(round(pred)),
            "latency_ms": round(latency_ms, 1),
        })
    except Exception as e:
        log.exception("prediction failed")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
