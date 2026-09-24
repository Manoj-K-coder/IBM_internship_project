# 🛍️ Customer Purchase Amount Prediction

A full-stack machine learning web application that predicts a customer's
purchase amount based on their age, previous purchase count, and review rating.

- **Backend**: Flask REST API
- **Frontend**: Streamlit
- **ML Model**: Linear Regression (scikit-learn)
- **Dataset**: `data/customer_shopping_behavior.csv` (3,900 customer records)

---

## Project Structure

```
customer_purchase_project/
├── data/
│   └── customer_shopping_behavior.csv   # Dataset
├── model/
│   ├── model.pkl                        # Trained model (generated)
│   ├── scaler.pkl                        # StandardScaler (generated)
│   ├── features.pkl                      # Feature list (generated)
│   ├── metrics.pkl                       # Evaluation metrics (generated)
│   └── diagnostics.png                   # Eval plots (generated)
├── backend/
│   └── app.py                            # Flask REST API
├── frontend/
│   └── ui.py                             # Streamlit UI
├── train_model.py                        # Model training script
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This saves `model/model.pkl`, `model/scaler.pkl`, `model/features.pkl`, and `model/metrics.pkl`.

### 3. Start the Flask backend
```bash
python backend/app.py
```
API runs at **http://localhost:5000**

### 4. Launch the Streamlit frontend
*(open a second terminal)*
```bash
streamlit run frontend/ui.py
```
UI opens at **http://localhost:8501**

---

## API Endpoints

| Method | Endpoint       | Description                          |
|--------|----------------|---------------------------------------|
| GET    | `/health`      | Health check                          |
| POST   | `/predict`     | Predict purchase amount               |
| GET    | `/dataset`     | Return full dataset as JSON           |
| GET    | `/model_info`  | Model coefficients + metrics          |

### POST `/predict` — example
```json
// Request
{ "age": 35, "previous_purchases": 20, "review_rating": 4.0 }

// Response
{
  "age": 35.0,
  "previous_purchases": 20.0,
  "review_rating": 4.0,
  "predicted_purchase_amount": 60.53,
  "currency": "USD"
}
```

---

## Frontend Pages

| Page                  | Description                                              |
|-----------------------|------------------------------------------------------------|
| **Predict Amount**    | Input age, purchases, rating; get instant prediction + gauge |
| **Dataset Explorer**  | Browse data, scatter/histogram/box/heatmap charts          |
| **Model Insights**    | R², MAE, RMSE, coefficients, feature importance chart       |

---

## Model Performance

| Metric | Value    |
|--------|----------|
| R²     | ≈ -0.005 |
| MAE    | ≈ $20.72 |
| RMSE   | ≈ $23.71 |

**Note:** Age, Previous Purchases and Review Rating show almost no linear
relationship with Purchase Amount in this dataset (R² ≈ 0), so the model's
predictions cluster tightly around the dataset mean regardless of input.
See the project report for details and suggested next steps.

---

## Tech Stack

| Layer     | Technology              |
|-----------|-------------------------|
| ML        | scikit-learn, numpy     |
| Backend   | Flask                   |
| Frontend  | Streamlit, Plotly       |
| Data      | pandas, matplotlib      |
