import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. LOAD SAVED ARTIFACTS
# ==========================================
# (Make sure you run this script from the root BizFlow360 folder)
@st.cache_resource
def load_artifacts():
    model = joblib.load('edusei_ml/models/trained/best_model.joblib')
    scaler = joblib.load('edusei_ml/models/preprocessing/scaler.joblib')
    le_county = joblib.load('edusei_ml/models/preprocessing/le_county.joblib')
    le_sector = joblib.load('edusei_ml/models/preprocessing/le_sector.joblib')
    return model, scaler, le_county, le_sector

model, scaler, le_county, le_sector = load_artifacts()

# ==========================================
# 2. BUILD THE UI DASHBOARD
# ==========================================
st.set_page_config(page_title="BizFlow360 Playground", layout="centered")
st.title(" BizFlow360 Model Playground")
st.markdown("Test the XGBoost model with custom inputs before deploying to the final app.")

st.sidebar.header("Enter Business Details")

# Categorical Inputs
county = st.sidebar.selectbox("County", le_county.classes_)
sector = st.sidebar.selectbox("Sector", le_sector.classes_)

# Numerical Inputs
age_months = st.sidebar.number_input("Business Age (Months)", min_value=1, value=24)
employees = st.sidebar.number_input("Number of Employees", min_value=1, value=5)
revenue = st.sidebar.number_input("Monthly Revenue (KES)", min_value=0.0, value=150000.0)
expenses = st.sidebar.number_input("Monthly Expenses (KES)", min_value=0.0, value=120000.0)
assets = st.sidebar.number_input("Total Assets (KES)", min_value=0.0, value=500000.0)
liabilities = st.sidebar.number_input("Total Liabilities (KES)", min_value=0.0, value=200000.0)
loan = st.sidebar.number_input("Loan Amount (KES)", min_value=0.0, value=50000.0)
mpesa = st.sidebar.number_input("M-Pesa Monthly Volume (KES)", min_value=0.0, value=100000.0)

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if st.sidebar.button(" Predict Financial Distress"):
    
    # Encode categoricals exactly like we did in training
    county_encoded = le_county.transform([county])[0]
    sector_encoded = le_sector.transform([sector])[0]

    # Create a DataFrame with the exact feature names and order from training
    input_data = pd.DataFrame({
        'business_age_months': [age_months],
        'employees': [employees],
        'monthly_revenue_kes': [revenue],
        'monthly_expenses_kes': [expenses],
        'total_assets_kes': [assets],
        'total_liabilities_kes': [liabilities],
        'loan_amount_kes': [loan],
        'mpesa_volume_kes': [mpesa],
        'county_encoded': [county_encoded],
        'sector_encoded': [sector_encoded]
    })

    # Scale the numerical features
    input_scaled = scaler.transform(input_data)

    # Get the probability of distress (Class 1)
    probability = model.predict_proba(input_scaled)[0][1]
    risk_percentage = round(probability * 100, 2)

    # Determine Risk Category
    if risk_percentage < 40:
        category = " Low Risk"
        color = "green"
    elif risk_percentage < 70:
        category = "🟡 Medium Risk"
        color = "orange"
    else:
        category = "🔴 High Risk"
        color = "red"

    # ==========================================
    # 4. DISPLAY RESULTS
    # ==========================================
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    col1.metric("Distress Probability", f"{risk_percentage}%")
    col2.metric("Risk Category", category)

    # Simple Rule-Based Insights (Based on your SHAP analysis)
    st.markdown("### 💡 Key Risk Factors (Based on your inputs)")
    risk_factors = []
    
    if expenses > revenue:
        risk_factors.append("️ **Expenses exceed Revenue:** The business is operating at a loss.")
    if liabilities > assets:
        risk_factors.append("️ **High Debt:** Liabilities are greater than total assets.")
    if age_months < 12:
        risk_factors.append("️ **Very Young Business:** Less than 1 year of operation increases risk.")
    if mpesa < (revenue * 0.3):
        risk_factors.append("⚠️ **Low Digital Footprint:** Low M-Pesa volume relative to revenue.")
        
    if not risk_factors:
        st.success("No major red flags detected based on standard financial ratios!")
    else:
        for factor in risk_factors:
            st.warning(factor)