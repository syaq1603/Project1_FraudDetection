import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc

# === Load data ===
@st.cache_data
def load_data():
    return pd.read_csv("simulated_transactions.csv")

df = load_data()

# === Page Config ===
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# === Title and Description ===
st.title("💸 Fraud Detection Dashboard")
st.markdown("This dashboard visualizes synthetic financial transactions with fraud detection insights.")

# === Sidebar Filter ===
st.sidebar.header("🔍 Filter Options")
selected_channel = st.sidebar.selectbox("Select Transaction Channel", df["channel"].unique())
filtered_df = df[df["channel"] == selected_channel]

# === KPI Metrics ===
st.markdown("### 📊 Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(filtered_df))
with col2:
    st.metric("Fraud Cases", int(filtered_df["fraud_flag"].sum()))
with col3:
    st.metric("Fraud Rate", f"{100 * filtered_df['fraud_flag'].mean():.2f}%")

# === Bar Plot: Fraud by User Type ===
st.markdown("### 👤 Fraud Distribution by User Type")
fig_bar = px.histogram(filtered_df, x="user_type", color="fraud_flag",
                       barmode="group", title="Fraud vs Non-Fraud by User Type",
                       labels={"fraud_flag": "Fraud Flag", "user_type": "User Type"})
st.plotly_chart(fig_bar, use_container_width=True)

# === Confusion Matrix (self-compared demo) ===
st.markdown("### 🧱 Confusion Matrix")
cm = confusion_matrix(filtered_df["fraud_flag"], filtered_df["fraud_flag"])  # demo: self comparison
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

# === ROC Curve (fake scores for demo) ===
st.markdown("### 📈 ROC Curve")
filtered_df["fraud_score"] = filtered_df["fraud_flag"] + 0.1 * (1 - filtered_df["fraud_flag"])
fpr, tpr, _ = roc_curve(filtered_df["fraud_flag"], filtered_df["fraud_score"])
roc_auc = auc(fpr, tpr)

fig_roc, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0, 1], [0, 1], "k--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()
st.pyplot(fig_roc)

# === Footer ===
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit • Data is 100% simulated • [GitHub Repo](https://github.com/syaq1603/Project1_FraudDetection)")

