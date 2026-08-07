import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="BizFlow360 Playground", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & PREPROCESSORS (Bulletproof Paths)
# ==========================================
@st.cache_resource
def load_resources():
    base_dir = Path(__file__).parent
    
    model_path = base_dir / "ml_models" / "models" / "trained" / "best_model.joblib"
    scaler_path = base_dir / "ml_models" / "models" / "preprocessing" / "scaler.joblib"
    le_county_path = base_dir / "ml_models" / "models" / "preprocessing" / "le_county.joblib"
    le_sector_path = base_dir / "ml_models" / "models" / "preprocessing" / "le_sector.joblib"
    
    # Check if files exist
    for p in [model_path, scaler_path, le_county_path, le_sector_path]:
        if not p.exists():
            st.error(f"❌ Missing file: {p.name}. Please ensure you have trained and saved the models.")
            st.stop()
            
    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(le_county_path),
        joblib.load(le_sector_path)
    )

try:
    model, scaler, le_county, le_sector = load_resources()
except Exception as e:
    st.error(f"Failed to load model resources: {e}")
    st.stop()

# ==========================================
# 3. UI HEADER
# ==========================================
st.markdown('<div class="main-header">🚀 BizFlow360 Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Early Warning System for Kenyan MSMEs</div>', unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR: USER INPUTS (Section 3.12)
# ==========================================
st.sidebar.header("📝 Enter Business Details")

# Categorical Inputs
county = st.sidebar.selectbox("County", sorted(le_county.classes_))
sector = st.sidebar.selectbox("Business Sector", sorted(le_sector.classes_))

st.sidebar.subheader("Financial Metrics (KES)")
monthly_revenue = st.sidebar.number_input("Monthly Revenue", min_value=0, value=150000, step=10000, format="%d")
monthly_expenses = st.sidebar.number_input("Monthly Expenses", min_value=0, value=120000, step=10000, format="%d")
total_assets = st.sidebar.number_input("Total Assets", min_value=0, value=500000, step=50000, format="%d")
total_liabilities = st.sidebar.number_input("Total Liabilities", min_value=0, value=200000, step=50000, format="%d")
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=100000, step=10000, format="%d")
mpesa_volume = st.sidebar.number_input("M-Pesa Monthly Volume", min_value=0, value=100000, step=10000, format="%d")

st.sidebar.subheader("Operational Details")
business_age = st.sidebar.slider("Business Age (Months)", 1, 120, 24)
employees = st.sidebar.slider("Number of Employees", 1, 50, 5)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
def run_prediction():
    # Encode categoricals
    county_encoded = le_county.transform([county])[0]
    sector_encoded = le_sector.transform([sector])[0]
    
    # EXACT feature order used during training
    features = [
        business_age, employees, monthly_revenue, monthly_expenses,
        total_assets, total_liabilities, loan_amount, mpesa_volume,
        county_encoded, sector_encoded
    ]
    
    X_input = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    
    probability = model.predict_proba(X_scaled)[0][1]
    return probability, X_scaled

# ==========================================
# 6. MAIN DASHBOARD DISPLAY
# ==========================================
# Use a button to prevent SHAP from freezing the app on every slider tick
if st.sidebar.button("🔍 Run Prediction", type="primary"):
    with st.spinner("Analyzing financial health..."):
        probability, X_scaled = run_prediction()
        
        # Store in session state so it persists
        st.session_state['prob'] = probability
        st.session_state['X_scaled'] = X_scaled

# Display results if they exist in session state
if 'prob' in st.session_state:
    probability = st.session_state['prob']
    X_scaled = st.session_state['X_scaled']
    
    # --- Top Metrics ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Distress Risk Score", value=f"{probability*100:.1f}%")
    with col2:
        if probability < 0.4:
            risk_cat = "Low Risk 🟢"
            st.success(risk_cat)
        elif probability < 0.7:
            risk_cat = "Medium Risk 🟡"
            st.warning(risk_cat)
        else:
            risk_cat = "High Risk 🔴"
            st.error(risk_cat)
    with col3:
        st.metric(label="Model Engine", value="XGBoost")
        
    st.divider()
    
    # --- Key Risk Factors (SHAP) ---
    st.subheader("🔍 Key Risk Factors (SHAP Explanation)")
    st.markdown("Here is exactly why the model made this prediction:")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        feature_names = [
            'Business Age (Months)', 'Employees', 'Monthly Revenue', 'Monthly Expenses',
            'Total Assets', 'Total Liabilities', 'Loan Amount', 'M-Pesa Volume',
            'County (Encoded)', 'Sector (Encoded)'
        ]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_scaled[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title("Why is this business at risk?", fontsize=14)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate SHAP plot: {e}")
        
    st.divider()
    
    # --- Recommended Actions ---
    st.subheader("📋 Recommended Actions")
    
    if probability >= 0.7:
        st.error("️ **Immediate Action Required:** This business is showing strong signs of financial distress.")
        st.markdown("""
        * **Cut Operating Expenses:** Review and reduce non-essential monthly expenses immediately.
        * **Debt Restructuring:** Contact lenders to renegotiate loan terms or consolidate debt.
        * **Cash Flow Monitoring:** Implement strict daily/weekly cash flow tracking.
        """)
    elif probability >= 0.4:
        st.warning("⚠️ **Caution:** This business is at moderate risk. Preventative measures are advised.")
        st.markdown("""
        * **Build an Emergency Fund:** Aim to save 3-6 months of operating expenses.
        * **Diversify Revenue:** Explore new income streams to reduce reliance on a single source.
        * **Review M-Pesa Inflows:** Ensure all transactions are being captured and reconciled.
        """)
    else:
        st.success("✅ **Great Job!** This business appears financially stable.")
        st.markdown("""
        * **Maintain Current Strategy:** Continue current financial management practices.
        * **Invest in Growth:** Consider reinvesting profits into expanding the business or marketing.
        * **Regular Audits:** Schedule quarterly financial reviews to maintain this healthy status.
        """)

else:
    # Initial state before button is clicked
    st.info("👈 Please enter the business details in the sidebar and click **Run Prediction** to see the analysis.")