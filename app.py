import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

# Add src to path to import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_preprocessing import load_data, engineer_features

st.set_page_config(page_title="IoT Predictive Maintenance", layout="wide")

@st.cache_resource
def load_model():
    # Update this line to load the XGBoost model instead
    return joblib.load('models/xgb_model.joblib')

@st.cache_data
def load_and_process_test_data():
    df = load_data('data/raw/test_FD001.txt')
    df = engineer_features(df)
    return df

st.title("⚙️ IoT Predictive Maintenance Dashboard")
st.markdown("Predicting Jet Engine Remaining Useful Life (RUL) using telemetry data and LightGBM.")

try:
    model = load_model()
    test_df = load_and_process_test_data()
except FileNotFoundError:
    st.error("Model or data files not found. Please run train_model.py first and ensure data is in data/raw/.")
    st.stop()

# Sidebar: Select Engine
engine_ids = test_df['unit_number'].unique()
selected_engine = st.sidebar.selectbox("Select Engine ID (Fleet)", engine_ids)

# Filter data for selected engine
engine_data = test_df[test_df['unit_number'] == selected_engine].copy()
features = [c for c in engine_data.columns if c not in ['unit_number', 'time_cycles', 'RUL']]

# Make predictions
engine_data['Predicted_RUL'] = model.predict(engine_data[features])

# --- DASHBOARD UI ---
current_cycle = engine_data['time_cycles'].max()
current_rul = int(engine_data['Predicted_RUL'].iloc[-1])

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Current Operational Cycle", value=current_cycle)
with col2:
    # Color code the RUL warning
    if current_rul < 30:
        st.error(f"Predicted Remaining Cycles: {current_rul} ⚠️ CRITICAL")
    elif current_rul < 60:
        st.warning(f"Predicted Remaining Cycles: {current_rul} ⚠️ WARNING")
    else:
        st.success(f"Predicted Remaining Cycles: {current_rul} ✅ HEALTHY")

st.divider()

# Plot Degradation Curve
st.subheader("Engine Health Degradation Curve")
st.line_chart(engine_data.set_index('time_cycles')['Predicted_RUL'])

st.divider()

# SHAP Explainability
st.subheader("Sensor Importance (SHAP Values)")
st.markdown("This shows which sensor readings are driving the machine's current health score down.")

# Calculate SHAP values for the last cycle of the engine
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(engine_data[features].iloc[-1:])

fig, ax = plt.subplots(figsize=(10, 4))
shap.summary_plot(shap_values, engine_data[features].iloc[-1:], plot_type="bar", show=False)
st.pyplot(fig)