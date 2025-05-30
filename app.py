import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("simulated_transactions.csv")

st.title("💸 Fraud Detection Dashboard")
df = load_data()

# --- Show Dataset Preview ---
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# --- Show Class Balance ---
st.subheader("📊 Fraud vs Non-Fraud")
fraud_counts = df["fraud_flag"].value_counts()
st.bar_chart(fraud_counts)

# --- Confusion Matrix Heatmap ---
st.subheader("🧱 Confusion Matrix (Simulated Predictions)")
cm = confusion_matrix(df["fraud_flag"], df["fraud_flag"])  # demo: compare to itself
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"], ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# --- ROC Curve ---
st.subheader("📈 ROC Curve (Simulated Scores)")
# For demo, fake predictions
df["fraud_score"] = df["fraud_flag"] + (0.1 * (1 - df["fraud_flag"]))  # fake scores
fpr, tpr, _ = roc_curve(df["fraud_flag"], df["fraud_score"])
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], "k--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

# --- Footer ---
st.markdown("Made with ❤️ using Streamlit")
