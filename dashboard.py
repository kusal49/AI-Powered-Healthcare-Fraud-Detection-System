import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SIU Fraud Hunter",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. LOAD ASSETS (Cached for speed) ---
@st.cache_resource
def load_artifacts():
    print("Loading Model Assets...")
    model = joblib.load('healthcare_fraud_rf.pkl')
    scaler = joblib.load('healthcare_scaler.pkl')
    # Load the feature lookup table (The CSV we saved earlier)
    data = pd.read_csv('provider_features.csv')
    return model, scaler, data

try:
    model, scaler, reference_data = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Artifacts not found! Make sure 'healthcare_fraud_rf.pkl', 'healthcare_scaler.pkl', and 'provider_features.csv' are in the same folder.")
    st.stop()

# --- 3. UI LAYOUT ---
st.title("🛡️ Healthcare Fraud Detection System")
st.markdown("### Special Investigation Unit (SIU) Portal")
st.markdown("---")

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    provider_id = st.text_input("Enter Provider ID:", placeholder="e.g., PRV51001")
with col2:
    st.write("") # Spacer
    analyze_btn = st.button("Analyze Risk", type="primary", use_container_width=True)

# --- 4. PREDICTION LOGIC ---
if analyze_btn and provider_id:
    
    # A. Lookup Provider
    # We strip whitespace just in case user added spaces
    pid = provider_id.strip()
    row = reference_data[reference_data['Provider'] == pid]
    
    if row.empty:
        st.error(f"❌ Provider ID '{pid}' not found in the historical database.")
    else:
        # B. Prepare Data
        # Drop the 'Provider' ID column (Model only wants numbers)
        features = row.drop('Provider', axis=1)
        
        # C. Scale Data (CRITICAL step)
        features_scaled = scaler.transform(features)
        
        # D. Predict
        probability = model.predict_proba(features_scaled)[0][1] # Probability of Class 1 (Fraud)
        prediction = model.predict(features_scaled)[0]
        
        # --- 5. DISPLAY RESULTS ---
        st.markdown("### Risk Assessment")
        
        # Dynamic Color Logic
        if probability > 0.8:
            risk_color = "red"
            risk_label = "CRITICAL RISK"
            emoji = "🚨"
        elif probability > 0.5:
            risk_color = "orange"
            risk_label = "HIGH RISK"
            emoji = "⚠️"
        else:
            risk_color = "green"
            risk_label = "SAFE"
            emoji = "✅"
            
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{probability:.1%}")
        m2.metric("Status", risk_label)
        m3.metric("Model Verdict", "Flagged" if prediction == 1 else "Cleared")
        
        # Visual Progress Bar
        st.progress(probability, text=f"Fraud Probability: {probability:.1%}")
        
        if probability > 0.5:
            st.error(f"{emoji} **RECOMMENDATION:** Open case file for immediate audit.")
        else:
            st.success(f"{emoji} **RECOMMENDATION:** No immediate action required.")
            
        # --- 6. CONTEXT (Show Key Stats) ---
        st.markdown("---")
        st.subheader("📊 Provider Snapshot")
        
        # Extract a few key features to show "Why"
        # Note: These names must match your CSV column names exactly.
        # We try to get common ones, handling potential missing columns safely.
        
        try:
            # Displaying raw values (easier for humans to read than scaled ones)
            c1, c2, c3 = st.columns(3)
            
            # Helper to get value safely
            def get_val(col_name):
                return row[col_name].values[0]

            with c1:
                st.info(f"**Claims Filed**\n\n{get_val('ClaimID_count'):,.0f}")
            with c2:
                # Check for one of our known financial columns
                if 'InscClaimAmtReimbursed_mean' in row.columns:
                     st.info(f"**Avg Claim Cost**\n\n${get_val('InscClaimAmtReimbursed_mean'):,.2f}")
                else:
                     st.info("**Avg Claim Cost**\n\nN/A")
            with c3:
                # Check for churn metric
                if 'Ratio_ClaimsPerPatient' in row.columns:
                    st.info(f"**Claims Per Patient**\n\n{get_val('Ratio_ClaimsPerPatient'):.2f}")
                else:
                    st.info("**Claims Per Patient**\n\nN/A")
                    
        except Exception as e:
            st.warning(f"Could not load snapshot details: {e}")

elif analyze_btn and not provider_id:
    st.warning("Please enter a Provider ID.")