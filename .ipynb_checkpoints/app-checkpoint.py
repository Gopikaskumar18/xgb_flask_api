"""
Flask REST API for the retail demand forecasting model.
Loads the trained XGBoost model plus the saved label encoders and feature
medians, so requests can send raw feature values and get a demand prediction.

Run:  python app.py   ->   POST http://127.0.0.1:5000/predict
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load artifacts saved by the modeling pipeline (section 10 of training)
# ---------------------------------------------------------------------------
model = xgb.XGBRegressor()
model.load_model("final_xgb_model.json")

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)      # {col: {category_string: int}}

with open("feature_medians.pkl", "rb") as f:
    medians = pickle.load(f)             # pd.Series: median per feature column

# The exact feature order the model was trained on
FEATURE_COLS = list(medians.index)

# Columns that were label-encoded during training
CAT_COLS = list(label_encoders.keys())


def prepare_features(raw: dict) -> pd.DataFrame:
    """Turn one raw feature dict into a single model-ready row."""
    df = pd.DataFrame([raw])

    # Encode categoricals using the SAME mappings learned at training time.
    # Unseen categories -> -1 (matches training behaviour).
    for col in CAT_COLS:
        if col in df.columns:
            mapping = label_encoders[col]
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Add any missing feature columns, then order them exactly as in training.
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_COLS]

    # Fill any missing/NaN values with the training medians (no leakage).
    df = df.fillna(medians)
    return df


@app.route("/", methods=["GET"])
def home():
    return "Demand Forecasting API is running. POST feature JSON to /predict."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "features" not in data:
            return jsonify({"error": "'features' key missing"}), 400

        X = prepare_features(data["features"])
        pred = float(model.predict(X)[0])

        # Demand can't be negative; round for a whole-unit forecast if wanted.
        pred = max(0.0, pred)
        return jsonify({
            "predicted_demand": round(pred, 2),
            "predicted_units":  int(round(pred)),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)