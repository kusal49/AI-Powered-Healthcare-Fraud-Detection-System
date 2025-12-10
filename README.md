# 🛡️ AI-Powered Healthcare Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![AWS](https://img.shields.io/badge/Cloud-AWS%20EC2-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## Project Overview
This project is an end-to-end **Machine Learning system** designed to assist insurance **Special Investigation Units (SIU)** in identifying fraudulent healthcare providers. 

Unlike traditional rule-based systems, this solution uses a **Random Forest classifier** to detect complex behavioral anomalies (e.g., patient churning, upcoding, and phantom billing) based on historical claims data.

The system is deployed as a **"Zero-Input" application**, meaning investigators simply enter a Provider ID, and the backend automatically aggregates thousands of claim records to generate a real-time risk score and explainable reason codes.

## Key Features
* **Zero-Input Feature Engineering:** Automated ETL pipeline that transforms raw transactional data into provider-level behavioral features (churn ratios, financial variance, etc.).
* **High-Recall Fraud Detection:** Random Forest model optimized for extreme class imbalance (1:10 fraud ratio) using cost-sensitive learning.
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to provide human-readable "Reason Codes" for every flag (e.g., *"Flagged due to 300% higher claim count than average"*).
* **Production-Ready Deployment:** Fully containerized with **Docker** and hosted on **AWS EC2** for global accessibility.

## Tech Stack
* **Language:** Python 3.11
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **Explainability:** SHAP
* **Dashboard/UI:** Streamlit
* **DevOps:** Docker, AWS EC2 (Ubuntu Linux)


## 🏗️ Project Architecture
```text
├── dashboard.py             # Streamlit Frontend (The Investigator Portal)
├── Dockerfile               # Container configuration for production
├── healthcare_fraud_rf.pkl  # Trained Random Forest Model
├── healthcare_scaler.pkl    # Standard Scaler for data normalization
├── provider_features.csv    # Pre-calculated Feature Store (Simulated Database)
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
