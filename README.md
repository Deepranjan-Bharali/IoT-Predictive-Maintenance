# IoT Predictive Maintenance

A Streamlit dashboard that estimates remaining useful life (RUL) for NASA CMAPSS FD001 engine telemetry with an XGBoost regressor. It includes an engine health curve, SHAP sensor importance, CSV upload for new telemetry, and downloadable predictions.

## Run locally

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository includes the preprocessed FD001 test set used by the dashboard. `data/raw/` is excluded from Git except for `train_FD001.txt`, which is needed to reproduce the model. Run this from the project root to train or retrain it:

```powershell
python src/train_model.py
```

This creates `models/xgb_model.joblib`. Keep that file in the deployment commit because the app loads it at startup. Then run:

```powershell
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub with `models/xgb_model.joblib` and `data/processed/test_FD001_processed.parquet` included.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Streamlit installs dependencies from `requirements.txt`; `.streamlit/config.toml` contains the app's basic theme and server settings.

The app reads paths relative to its own location, so it does not depend on the deployment process's working directory. The raw dataset is only needed when training a model, not to run the dashboard.

## Predict on new telemetry

Choose **Upload telemetry CSV** in the dashboard and upload one row per engine cycle. Use **Download CSV template** for the expected headers. The CSV must include an engine ID (`unit_number`), cycle (`time_cycles`), and the sensor columns used by the trained model; sensor headers can use either `s_1` or `sensor_1` naming. Settings are optional. Include chronological history for each engine, since rolling sensor features are calculated from prior rows. The dashboard shows the latest prediction for every uploaded engine and lets you download the results as CSV.
