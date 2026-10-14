from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
from data_preprocessing import engineer_features, load_data  # noqa: E402


MODEL_PATH = ROOT_DIR / "models" / "xgb_model.joblib"
PROCESSED_TEST_PATH = ROOT_DIR / "data" / "processed" / "test_FD001_processed.parquet"
RAW_TEST_PATH = ROOT_DIR / "data" / "raw" / "test_FD001.txt"

st.set_page_config(page_title="IoT Predictive Maintenance", page_icon="⚙️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    # Prefer the committed, preprocessed dataset so deployment does not depend on
    # the raw NASA files, which are intentionally excluded from Git.
    if PROCESSED_TEST_PATH.exists():
        df = pd.read_parquet(PROCESSED_TEST_PATH)
        df = normalize_input_columns(df)
    elif RAW_TEST_PATH.exists():
        df = load_data(RAW_TEST_PATH)
    else:
        raise FileNotFoundError(
            f"Test data not found. Expected {PROCESSED_TEST_PATH} or {RAW_TEST_PATH}."
        )

    # Apply the same feature engineering used for training when loading the
    # stored raw-sensor parquet; leave already engineered data as-is.
    if not any(column.endswith("_roll_mean") for column in df.columns):
        df = engineer_features(df)
    return df


def normalize_input_columns(df):
    """Map common telemetry headers to the names used by the training pipeline."""
    df = df.rename(columns={column: str(column).strip().lower() for column in df.columns})
    aliases = {
        "engine_id": "unit_number",
        "unit_id": "unit_number",
        "engine": "unit_number",
        "unit": "unit_number",
        "cycle": "time_cycles",
        "time_in_cycles": "time_cycles",
    }
    rename = {
        alias: canonical for alias, canonical in aliases.items()
        if alias in df.columns and canonical not in df.columns
    }
    for index in range(1, 22):
        alias, canonical = f"sensor_{index}", f"s_{index}"
        if alias in df.columns and canonical not in df.columns:
            rename[alias] = canonical
    return df.rename(columns=rename)


def prepare_uploaded_data(upload, model):
    """Read, validate, and engineer features for a user-provided telemetry CSV."""
    df = pd.read_csv(upload)
    df = normalize_input_columns(df)
    if "RUL" in df.columns:
        df = df.drop(columns=["RUL"])

    expected_features = model_features(model)
    required_sensors = {
        feature.split("_roll_")[0]
        for feature in expected_features
        if feature.startswith("s_")
    }
    required_columns = {"unit_number", "time_cycles", *required_sensors}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
            + ". Include engine ID, cycle, and the sensor columns shown in the CSV template."
        )
    if df.empty:
        raise ValueError("The uploaded CSV has no telemetry rows.")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(required_columns)].isna().any().any():
        raise ValueError("Required ID, cycle, and sensor columns must contain numeric values.")
    if (df["time_cycles"] < 0).any():
        raise ValueError("Cycle values must be zero or greater.")

    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)
    if df.duplicated(["unit_number", "time_cycles"]).any():
        raise ValueError("Each engine must have at most one row per cycle.")
    df = engineer_features(df)
    missing_features = sorted(set(expected_features).difference(df.columns))
    if missing_features:
        raise ValueError(
            "Could not create the model's expected features: " + ", ".join(missing_features)
        )
    return df


def model_features(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = model.get_booster().feature_names
    return list(names) if names is not None and len(names) else []


st.title("⚙️ IoT Predictive Maintenance Dashboard")
st.markdown("Predicting jet engine remaining useful life (RUL) using telemetry data and XGBoost.")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. From the project root, run `python src/train_model.py`.")
    st.stop()

try:
    model = load_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

data_source = st.radio("Prediction data", ["Bundled sample", "Upload telemetry CSV"], horizontal=True)
if data_source == "Upload telemetry CSV":
    template_columns = ["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"] + [
        f"s_{index}" for index in range(1, 22)
    ]
    st.download_button(
        "Download CSV template",
        data=pd.DataFrame(columns=template_columns).to_csv(index=False),
        file_name="telemetry_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Upload CSV telemetry with one row per engine cycle. Include engine ID, cycle, and sensor readings; "
        "upload chronological history for each engine so rolling features use its prior readings. "
        "Headers may use `sensor_1` or `s_1` naming."
    )
    uploaded_file = st.file_uploader("Upload telemetry CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Choose a CSV file to generate predictions.")
        st.stop()
    try:
        test_df = prepare_uploaded_data(uploaded_file, model)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not use this CSV: {exc}")
        st.stop()
else:
    try:
        test_df = load_test_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

required_columns = {"unit_number", "time_cycles"}
if not required_columns.issubset(test_df.columns):
    st.error("The test dataset must include `unit_number` and `time_cycles` columns.")
    st.stop()

engine_ids = sorted(test_df["unit_number"].unique())
selected_engine = st.sidebar.selectbox("Select Engine ID (Fleet)", engine_ids)
test_df = test_df.sort_values(["unit_number", "time_cycles"]).copy()
features = model_features(model)
if not features:
    features = [column for column in test_df.columns if column not in {"unit_number", "time_cycles", "RUL"}]

missing_features = sorted(set(features).difference(test_df.columns))
if missing_features:
    st.error("Input data is missing model features: " + ", ".join(missing_features))
    st.stop()

test_df["Predicted_RUL"] = model.predict(test_df[features])
engine_data = test_df[test_df["unit_number"] == selected_engine].copy()
current_cycle = int(engine_data["time_cycles"].max())
current_rul = max(0, int(round(engine_data["Predicted_RUL"].iloc[-1])))

col1, col2 = st.columns(2)
col1.metric(label="Current Operational Cycle", value=current_cycle)
if current_rul < 30:
    col2.error(f"Predicted Remaining Cycles: {current_rul} ⚠️ CRITICAL")
elif current_rul < 60:
    col2.warning(f"Predicted Remaining Cycles: {current_rul} ⚠️ WARNING")
else:
    col2.success(f"Predicted Remaining Cycles: {current_rul} ✅ HEALTHY")

st.divider()
st.subheader("Engine Health Degradation Curve")
st.line_chart(engine_data.set_index("time_cycles")["Predicted_RUL"])
st.subheader("Latest predictions by engine")
latest_predictions = (
    test_df.sort_values("time_cycles")
    .groupby("unit_number", as_index=False)
    .tail(1)[["unit_number", "time_cycles", "Predicted_RUL"]]
    .sort_values("unit_number")
    .rename(columns={"unit_number": "Engine ID", "time_cycles": "Latest Cycle", "Predicted_RUL": "Predicted RUL"})
)
latest_predictions["Predicted RUL"] = latest_predictions["Predicted RUL"].clip(lower=0).round().astype(int)
st.dataframe(latest_predictions, use_container_width=True, hide_index=True)
st.download_button(
    "Download predictions as CSV",
    latest_predictions.to_csv(index=False),
    file_name="rul_predictions.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Sensor Importance (SHAP Values)")
st.markdown("Sensor readings contributing to the prediction for the selected engine's latest cycle.")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = load_explainer(model)
latest_features = engine_data[features].iloc[-1:]
shap_values = explainer.shap_values(latest_features)
shap.summary_plot(shap_values, latest_features, plot_type="bar", show=False, plot_size=None)
st.pyplot(plt.gcf())
plt.close(plt.gcf())
