import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# === Load model and data ===
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("simulated_transactions.csv")

model = load_model()
df = load_data()

# === Page Setup ===
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💸 Real-Time Fraud Detection Dashboard")

# === Sidebar Filters ===
st.sidebar.header("🔍 Filter Transactions")
channel = st.sidebar.selectbox("Select Channel", df["channel"].unique())
amount_range = st.sidebar.slider("Select Amount Range", float(df["amount"].min()), float(df["amount"].max()), (10.0, 10000.0))
filtered = df[(df["channel"] == channel) & (df["amount"].between(*amount_range))]

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Charts", "🤖 Predict Fraud"])

# === TAB 1: Overview ===
with tab1:
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(filtered))
    col2.metric("Fraud Cases", int(filtered["fraud_flag"].sum()))
    col3.metric("Fraud Rate", f"{100 * filtered['fraud_flag'].mean():.2f}%")
    st.dataframe(filtered.head(10))

# === TAB 2: Charts ===
with tab2:
    st.subheader("Fraud by User Type")
    fig_bar = px.histogram(filtered, x="user_type", color="fraud_flag", barmode="group", title="Fraud vs Non-Fraud by User Type")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Confusion Matrix (self-check)")
    cm = confusion_matrix(filtered["fraud_flag"], filtered["fraud_flag"])
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("ROC Curve (fake scores demo)")
    filtered["fraud_score"] = filtered["fraud_flag"] + 0.1 * (1 - filtered["fraud_flag"])
    fpr, tpr, _ = roc_curve(filtered["fraud_flag"], filtered["fraud_score"])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig_roc)

# === TAB 3: Real-Time Prediction ===
with tab3:
    st.subheader("Predict a New Transaction")

    with st.form("prediction_form"):
        amt = st.number_input("Amount", min_value=1.0, max_value=100000.0, value=250.0)
        channel = st.selectbox("Channel", df["channel"].unique())
        currency = st.selectbox("Currency", df["currency"].unique())
        location = st.selectbox("Location", df["location"].unique())
        user_type = st.selectbox("User Type", df["user_type"].unique())
        age = st.slider("Account Age (days)", 0, 2000, 180)

        submit = st.form_submit_button("🔍 Predict Fraud")

    if submit:
        input_df = pd.DataFrame([{
            "amount": amt,
            "channel": channel,
            "currency": currency,
            "location": location,
            "user_type": user_type,
            "account_age_days": age
        }])

        # One-hot encode to match training format
        input_encoded = pd.get_dummies(input_df)
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_columns]

        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        if pred == 1:
            st.error(f"🚨 Fraud Detected! Risk Score: {prob:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction. Risk Score: {prob:.2f}")

# === Footer ===
st.markdown("---")
st.markdown("Made with ❤️ by Rubiyah • Synthetic data only • [GitHub Repo](https://github.com/syaq1603/Project1_FraudDetection)")




