from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "healthcare_fraud_rf.pkl"
SCALER_PATH = "healthcare_scaler.pkl"
DATA_PATH = "provider_features.csv"

# --- 1. INITIALIZE APP & LOAD ARTIFACTS ---
app = FastAPI(title="SIU Fraud Hunter API", version="1.0")

print("Loading Model pipeline...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
reference_data = pd.read_csv(DATA_PATH)
print("System Ready.")

# Define the Input Format
class ProviderQuery(BaseModel):
    provider_id: str

# --- 2. HELPER FUNCTIONS ---
def get_provider_features(pid):
    """Looks up the pre-calculated features for a provider."""
    # Filter by Provider ID
    row = reference_data[reference_data['Provider'] == pid]
    
    if row.empty:
        return None
    
    # Drop the ID column so we have just the numbers
    features = row.drop('Provider', axis=1)
    return features

# --- 3. THE PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_fraud(query: ProviderQuery):
    
    # A. Lookup Data
    features_df = get_provider_features(query.provider_id)
    
    if features_df is None:
        raise HTTPException(status_code=404, detail=f"Provider {query.provider_id} not found in database.")
    
    # B. Scale Data (CRITICAL: Must match training scale)
    # The scaler expects a DataFrame/Array matching the training shape
    scaled_features = scaler.transform(features_df)
    
    # C. Predict
    # [:, 1] gives the probability of Class 1 (Fraud)
    fraud_prob = model.predict_proba(scaled_features)[0][1]
    is_fraud = fraud_prob > 0.5 # Threshold
    
    # D. Return Response
    return {
        "provider_id": query.provider_id,
        "fraud_probability": float(round(fraud_prob, 4)),
        "alert_triggered": bool(is_fraud),
        "risk_category": "CRITICAL" if fraud_prob > 0.8 else "HIGH" if fraud_prob > 0.5 else "LOW"
    }

# --- 4. HEALTH CHECK ---
@app.get("/")
def home():
    return {"status": "active", "model": "Random Forest v1"}