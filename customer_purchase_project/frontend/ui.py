# -*- coding: utf-8 -*-
"""
frontend/ui.py
---------------
Streamlit frontend for the Customer Purchase Amount Prediction app.
Talks to the Flask backend (http://localhost:5000).
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = "http://localhost:5000"

st.set_page_config(page_title="Purchase Amount Predictor", page_icon="🛍️", layout="wide")

page = st.sidebar.radio("Navigate", ["Predict Amount", "Dataset Explorer", "Model Insights"])

# ---------------------------------------------------------------- Predict
if page == "Predict Amount":
    st.title("🛍️ Predict Purchase Amount")
    st.write("Enter customer details to estimate how much they'll spend.")

    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age", min_value=18, max_value=100, value=35)
    previous_purchases = col2.number_input("Previous Purchases", min_value=0, max_value=100, value=20)
    review_rating = col3.slider("Review Rating", min_value=1.0, max_value=5.0, value=3.8, step=0.1)

    if st.button("Predict", type="primary"):
        try:
            resp = requests.post(f"{API_URL}/predict", json={
                "age": age,
                "previous_purchases": previous_purchases,
                "review_rating": review_rating,
            }, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            amount = result["predicted_purchase_amount"]

            st.metric("Predicted Purchase Amount", f"${amount:,.2f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=amount,
                title={"text": "Predicted Amount (USD)"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#3b82d4"}},
            ))
            st.plotly_chart(fig, use_container_width=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend API: {e}")

    st.divider()
    st.subheader("Quick Reference")
    quick_ref = pd.DataFrame({
        "Age": [25, 35, 45, 55, 65],
        "Previous Purchases": [5, 20, 25, 35, 45],
        "Review Rating": [3.0, 3.8, 4.0, 4.2, 4.5],
    })
    st.dataframe(quick_ref, use_container_width=True)

# ------------------------------------------------------------ Explorer
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    try:
        data = requests.get(f"{API_URL}/dataset", timeout=5).json()
        df = pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend API: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric("Avg Purchase", f"${df['Purchase Amount (USD)'].mean():.2f}")
    c3.metric("Avg Age", f"{df['Age'].mean():.1f}")
    c4.metric("Avg Rating", f"{df['Review Rating'].mean():.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Scatter", "Histogram", "Box Plot", "Correlation"])

    with tab1:
        fig = px.scatter(df, x="Age", y="Purchase Amount (USD)", color="Category",
                          title="Age vs Purchase Amount")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x="Purchase Amount (USD)", nbins=20,
                            title="Purchase Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.box(df, x="Category", y="Purchase Amount (USD)",
                     title="Purchase Amount by Category")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------- Insights
elif page == "Model Insights":
    st.title("📈 Model Insights")
    try:
        info = requests.get(f"{API_URL}/model_info", timeout=5).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend API: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{info['r2']:.4f}")
    c2.metric("MAE", f"${info['mae']:,.2f}")
    c3.metric("RMSE", f"${info['rmse']:,.2f}")

    st.subheader("Model Equation")
    eq = f"Purchase Amount = {info['intercept']:,.2f}"
    for feat, coef in info["coefficients"].items():
        eq += f" + ({coef:,.4f} × {feat}_scaled)"
    st.code(eq)

    st.subheader("Feature Coefficients")
    coef_df = pd.DataFrame({
        "Feature": list(info["coefficients"].keys()),
        "Coefficient": list(info["coefficients"].values()),
    })
    fig = px.bar(coef_df, x="Feature", y="Coefficient", title="Feature Coefficients")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "A low R² here indicates Age, Previous Purchases and Review Rating have "
        "little linear relationship with Purchase Amount in this dataset — "
        "spend appears to be driven by other factors not captured by these features."
    )
