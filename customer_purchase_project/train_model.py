# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Loads customer_shopping_behavior.csv, trains a Linear Regression model
to predict Purchase Amount (USD) from Age, Previous Purchases and
Review Rating, evaluates it, and saves the trained model + scaler
to /model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_shopping_behavior.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Review Rating has a few missing values -> impute with column mean
df["Review Rating"] = df["Review Rating"].fillna(df["Review Rating"].mean())

FEATURES = ["Age", "Previous Purchases", "Review Rating"]
TARGET = "Purchase Amount (USD)"

print(df[FEATURES + [TARGET]].describe())

X = df[FEATURES].values
y = df[TARGET].values

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train
model = LinearRegression()
model.fit(X_train_sc, y_train)

# Evaluate
y_pred = model.predict(X_test_sc)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n-- Model Evaluation --")
print(f"  MAE  : ${mae:,.2f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R2   : {r2:.4f}")

# Save model, scaler and feature list
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))
print("\nModel and scaler saved to /model.")

# Plots (saved to file; no GUI window needed)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Customer Purchase Amount Prediction - Model Diagnostics", fontsize=14)

# 1. Actual vs Predicted
axes[0].scatter(y_test, y_pred, color="#3b82d4", edgecolors="white", s=60, alpha=0.85)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Purchase Amount ($)")
axes[0].set_ylabel("Predicted Purchase Amount ($)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# 2. Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=15, color="#7c5cd8", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Residual ($)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "diagnostics.png")
plt.savefig(plot_path, dpi=120)
print(f"Diagnostics plot saved: {plot_path}")

# Persist metrics for the backend /model_info endpoint
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "intercept": float(model.intercept_),
    "coefficients": {f: float(c) for f, c in zip(FEATURES, model.coef_)},
    "n_samples": int(len(df)),
}
joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.pkl"))
