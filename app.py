import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="🧠", layout="centered")

# Load trained model
model = tf.keras.models.load_model('model.h5')

with open('Label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('OneHotEncoder_geo.pkl', 'rb') as file:
    OneHotEncoder_geo = pickle.load(file)

with open('StandardScaler', 'rb') as file:
    scaler = pickle.load(file)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1rem 3rem 1rem; max-width: 680px; margin: auto; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,210,200,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 12px rgba(0,210,200,0.6));
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2c8, #7ee8fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero p {
    color: #8b949e;
    font-size: 0.9rem;
    margin: 0;
}

/* ── Section cards ── */
.section-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 0.5rem;
    margin-bottom: 1.4rem;
}
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #00d2c8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #c9d1d9 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #00d2c8 !important;
    box-shadow: 0 0 0 3px rgba(0,210,200,0.15) !important;
}

/* slider accent */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #00d2c8 !important;
}
div[data-testid="stSlider"] div[data-testid="stTickBar"] > div {
    background: linear-gradient(90deg, #00d2c8, #7ee8fa) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #00d2c8, #7ee8fa);
    color: #0d1117;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 0;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeSlideUp 0.5s ease forwards;
}
.result-stay {
    background: linear-gradient(135deg, #0d2818, #0f3d22);
    border: 1px solid #238636;
}
.result-churn {
    background: linear-gradient(135deg, #2d0f0f, #3d1515);
    border: 1px solid #da3633;
}
.result-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.6rem;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-stay .result-label  { color: #3fb950; }
.result-churn .result-label { color: #f85149; }
.result-sub {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 1.2rem;
}

/* probability bar */
.prob-bar-wrap {
    background: #21262d;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.4rem 0 0.3rem;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    animation: growBar 0.8s ease forwards;
    transform-origin: left;
}
.prob-bar-stay  { background: linear-gradient(90deg, #238636, #3fb950); }
.prob-bar-churn { background: linear-gradient(90deg, #da3633, #f85149); }
.prob-label {
    font-size: 0.78rem;
    color: #8b949e;
    display: flex;
    justify-content: space-between;
}
.prob-value {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.5rem 0 0.2rem;
}
.result-stay  .prob-value { color: #3fb950; }
.result-churn .prob-value { color: #f85149; }

/* pulse ring */
.pulse-ring {
    display: inline-block;
    width: 60px; height: 60px;
    border-radius: 50%;
    position: relative;
    margin-bottom: 0.8rem;
}
.pulse-ring::before, .pulse-ring::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    animation: pulse 1.8s ease-out infinite;
}
.pulse-stay  { background: rgba(63,185,80,0.2); }
.pulse-churn { background: rgba(248,81,73,0.2); }
.pulse-stay::before, .pulse-stay::after   { border: 2px solid #3fb950; }
.pulse-churn::before, .pulse-churn::after { border: 2px solid #f85149; }
.pulse-ring::after { animation-delay: 0.9s; }
.pulse-emoji {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.6rem;
    line-height: 1;
}

/* ── Keyframes ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes growBar {
    from { width: 0; }
}
@keyframes pulse {
    0%   { transform: scale(1);   opacity: 0.8; }
    100% { transform: scale(2.2); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🧠</span>
    <h1>Customer Churn Predictor</h1>
    <p>Enter customer details below to predict the likelihood of churn using a trained neural network.</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Personal Info ─────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
Geography = st.selectbox('Geography', OneHotEncoder_geo.categories_[0])
Gender    = st.selectbox('Gender', label_encoder_gender.classes_)
Age       = st.slider('Age', 18, 95, 35)
tenure    = st.slider('Tenure (years)', 0, 10, 3)
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Financial Info ────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">💳 Financial Information</div>', unsafe_allow_html=True)
CreditScore      = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
Balance          = st.number_input('Balance', min_value=0.0, value=0.0)
EstimatedSalary  = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
num_of_products  = st.slider('Number of Products', 1, 4, 1)
HasCrCard        = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────────────────────
predict = st.button('🔍 Predict Churn')

if predict:
    input_data = pd.DataFrame({
        'CreditScore':     [CreditScore],
        'Gender':          [label_encoder_gender.transform([Gender])[0]],
        'Age':             [Age],
        'Tenure':          [tenure],
        'Balance':         [Balance],
        'NumOfProducts':   [num_of_products],
        'HasCrCard':       [HasCrCard],
        'IsActiveMember':  [is_active_member],
        'EstimatedSalary': [EstimatedSalary]
    })

    geo_encoded = OneHotEncoder_geo.transform([[Geography]]).toarray()
    geo_df      = pd.DataFrame(geo_encoded, columns=OneHotEncoder_geo.get_feature_names_out(['Geography']))
    input_data  = pd.concat([input_data, geo_df], axis=1)

    scaled_input     = scaler.transform(input_data)
    prediction_prob  = model.predict(scaled_input)[0][0]
    pct              = round(float(prediction_prob) * 100, 1)
    is_churn         = prediction_prob > 0.5

    card_cls  = "result-churn" if is_churn else "result-stay"
    pulse_cls = "pulse-churn"  if is_churn else "pulse-stay"
    bar_cls   = "prob-bar-churn" if is_churn else "prob-bar-stay"
    emoji     = "⚠️" if is_churn else "✅"
    label     = "Likely to CHURN" if is_churn else "Likely to STAY"
    sub       = "This customer has a high risk of leaving." if is_churn else "This customer is likely to remain."

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="pulse-ring {pulse_cls}">
            <span class="pulse-emoji">{emoji}</span>
        </div>
        <div class="result-label">{label}</div>
        <div class="result-sub">{sub}</div>
        <div class="prob-value">{pct}%</div>
        <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.5rem;">Churn Probability</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar-fill {bar_cls}" style="width:{pct}%"></div>
        </div>
        <div class="prob-label"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>
    """, unsafe_allow_html=True)
