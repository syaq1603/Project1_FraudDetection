# 💸 Financial Fraud Detection Simulator

This project simulates realistic financial transaction data—including fraud-related behaviors such as money laundering, scams, mules, and terrorist financing. It generates synthetic user profiles and transaction logs for use in machine learning, anomaly detection, and dashboard development.

## 📁 Project Structure

Project1_FraudDetection/
├── simulated_users.csv # Simulated user profiles
├── simulated_transactions.csv # Labeled transactions with fraud flags
├── fraud_simulation.ipynb # Main notebook (data generation + labeling + charts)

## 🔍 Features

- Generate 1000+ realistic user profiles using Faker
- Simulate 5000 financial transactions over 90 days
- Include labeled adversary behaviors:
  - Money laundering
  - Money mule activity
  - Scam accounts
  - Fraudsters (e.g., chargebacks)
  - Terrorist financing
- Automatically flag fraudulent activity with `fraud_flag`
- Visualizations of fraud vs. normal transaction behavior

## 📊 Use Cases

- Training machine learning models to detect financial fraud
- Benchmarking fraud detection pipelines
- Educating analysts with synthetic transaction flows

## 🚀 How to Use

1. Open the `fraud_simulation.ipynb` notebook in Google Colab
2. Run the cells to generate and label data
3. Download or analyze the `simulated_transactions.csv` for modeling
4. (Optional) Extend with Streamlit, sklearn, or anomaly detection

## 📌 Requirements

- Python 3.7+
- pandas
- faker
- matplotlib

Install with:

```bash
pip install pandas faker matplotlib

---
📜 License
This project is provided for educational and research purposes. All data is synthetic and does not represent any real individual or institution.

💡 Future Ideas
Add time-based fraud patterns

Train and evaluate a fraud classification model

Deploy a dashboard using Streamlit or Flask

Made with 💡 by syaq1603
