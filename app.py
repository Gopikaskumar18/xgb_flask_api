from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# -----------------------------
# Load trained model (CPU-friendly)
# -----------------------------
with open("final_xgb_model_cpu.json", "rb") as f:
    model = xgb.XGBRegressor()
    model.load_model("final_xgb_model_cpu.json")

# Load label encoders if needed
try:
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
except FileNotFoundError:
    label_encoders = None

# -----------------------------
# Root route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "XGBoost Model API is running. Use /predict with POST JSON data."

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting the features as dictionary under 'features'
        if "features" not in data:
            return jsonify({"error": "'features' key missing"}), 400

        features = data["features"]

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Apply label encoders if available
        if label_encoders:
            for col, le in label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])

        # Ensure numeric types only
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')

        # Predict
        prediction = model.predict(df)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
