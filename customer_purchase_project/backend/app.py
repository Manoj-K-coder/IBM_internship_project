# -*- coding: utf-8 -*-
"""
backend/app.py
---------------
Flask REST API that serves predictions from the trained
Purchase Amount model, plus dataset and model-info endpoints.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_shopping_behavior.csv")

app = Flask(__name__)

# Load model artifacts once at startup
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
metrics = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))

df = pd.read_csv(DATA_PATH)
df["Review Rating"] = df["Review Rating"].fillna(df["Review Rating"].mean())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Purchase Amount Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}

    missing = [f for f in FEATURES if f.lower().replace(" ", "_") not in
               {k.lower().replace(" ", "_"): v for k, v in payload.items()}]

    # Accept snake_case keys: age, previous_purchases, review_rating
    key_map = {
        "Age": "age",
        "Previous Purchases": "previous_purchases",
        "Review Rating": "review_rating",
    }

    try:
        row = [[float(payload[key_map[f]]) for f in FEATURES]]
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400

    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]

    return jsonify({
        "age": row[0][0],
        "previous_purchases": row[0][1],
        "review_rating": row[0][2],
        "predicted_purchase_amount": round(float(prediction), 2),
        "currency": "USD",
    })


@app.route("/dataset", methods=["GET"])
def dataset():
    return jsonify(df.to_dict(orient="records"))


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "features": FEATURES,
        "coefficients": metrics["coefficients"],
        "intercept": metrics["intercept"],
        "r2": metrics["r2"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_samples": metrics["n_samples"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
