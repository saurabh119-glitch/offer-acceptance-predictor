import streamlit as st
import joblib
import json
import numpy as np

# Load model & metrics
model = joblib.load('offer_acceptance_model.pkl')
with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

st.set_page_config(page_title="Offer Acceptance Predictor", page_icon="💼")
st.title("💼 AI Offer Acceptance Predictor for Recruiters")
st.markdown("Predict if a candidate will accept your job offer — before you send it!")

# Sidebar: Model Transparency
with st.sidebar:
    st.header("🔍 About This Model")
    st.write(f"**Accuracy:** {metrics['Accuracy'] * 100:.0f}%")
    st.write(f"**F1 Score:** {metrics['F1_Score']:.2f}")
    st.markdown("""
    - Trained on 1,000+ synthetic profiles
    - Uses: experience, CTC gap, job search duration
    - Ideal for IT, Sales, HR roles in India
    """)
    st.info("💡 Tip: Lower CTC gap → higher acceptance!")

# Main form
col1, col2 = st.columns(2)
with col1:
    exp = st.slider("Years of Experience", 0, 15, 3)
    current_ctc = st.number_input("Current CTC (LPA)", 3, 30, 6)
    expected_ctc = st.number_input("Expected CTC (LPA)", 4, 40, 8)
with col2:
    days_search = st.slider("Days in Job Search", 1, 180, 30)
    apps_count = st.slider("Jobs Applied To", 1, 20, 5)
    role = st.selectbox("Role Type", ["IT", "Sales", "HR", "Finance", "Operations"])

# Encode role
role_map = {"IT": 0, "Sales": 1, "HR": 2, "Finance": 3, "Operations": 4}
role_enc = role_map[role]

# Predict
if st.button("🔮 Predict Acceptance"):
    features = [[exp, current_ctc, expected_ctc, days_search, apps_count, role_enc]]
    prob = model.predict_proba(features)[0][1]  # Probability of acceptance
    
    if prob > 0.7:
        st.success(f"✅ High Chance: **{prob:.0%}** likelihood to accept")
    elif prob > 0.4:
        st.warning(f"⚠️ Moderate Chance: **{prob:.0%}**")
    else:
        st.error(f"❌ Low Chance: **{prob:.0%}** — consider negotiation")
    
    st.info(f"💡 F1 Score: {metrics['F1_Score']:.2f} — balances precision & recall")

st.markdown("---")
st.caption("Made for Indian recruiters | [GitHub](https://github.com/yourname/offer-acceptance-predictor)")