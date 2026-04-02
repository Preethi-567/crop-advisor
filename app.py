

# ─── Streamlit App (save separately as app.py) ────────────────────────────────

# app.py — Streamlit UI for Crop Recommendation System
# Run: streamlit run app.py

import streamlit as st
import joblib
import sys
sys.path.insert(0, ".")
from predict import CropPredictor

st.set_page_config(page_title="🌾 Crop Advisor", layout="centered")

@st.cache_resource
def load_predictor():
    return CropPredictor.load("models/crop_model.pkl", "models/label_encoder.pkl")

predictor = load_predictor()

st.title("🌾 Smart Crop Recommendation System")
st.markdown("Enter your **soil and climate data** to get personalised crop recommendations.")

col1, col2 = st.columns(2)
with col1:
    N           = st.slider("Nitrogen (N)",        0,   140, 70)
    P           = st.slider("Phosphorus (P)",      5,   145, 50)
    K           = st.slider("Potassium (K)",       5,   205, 50)
    temperature = st.slider("Temperature (°C)",    8.0,  44.0, 25.0)
with col2:
    humidity    = st.slider("Humidity (%)",       14.0, 100.0, 60.0)
    ph          = st.slider("Soil pH",             3.5,  10.0,  6.5)
    rainfall    = st.slider("Rainfall (mm)",       20,   300,  100)

features = dict(N=N, P=P, K=K, temperature=temperature,
                humidity=humidity, ph=ph, rainfall=rainfall)

if st.button("🔍  Get Recommendations", type="primary"):
    recs = predictor.top_n_recommendations(features, n=3)
    st.subheader("Top 3 Recommendations")
    for r in recs:
        with st.expander(f"#{r['rank']}  {r['crop']}  —  {r['confidence']} confidence"):
            st.write(f"**Why:** {r['reason']}")
