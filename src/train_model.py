import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from data_preprocessing import process_training_data

# Ensure models directory exists
os.makedirs('../models', exist_ok=True)

print("Loading and preprocessing data...")
train_df = process_training_data('../data/raw/train_FD001.txt')

print("Splitting data into Train and Validation sets...")
val_engines = train_df['unit_number'].unique()[-20:] 
val_df = train_df[train_df['unit_number'].isin(val_engines)]
train_subset = train_df[~train_df['unit_number'].isin(val_engines)]

features = [c for c in train_df.columns if c not in ['unit_number', 'time_cycles', 'RUL']]

X_train, y_train = train_subset[features], train_subset['RUL']
X_val, y_val = val_df[features], val_df['RUL']

print("Training XGBoost Regressor...")
# XGBoost handles early stopping within the model instantiation in recent versions
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42,
    early_stopping_rounds=20
)

model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=False
)

# Evaluate
preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"Validation RMSE: {rmse:.2f}")

# Save the model
model_path = '../models/xgb_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved successfully to {model_path}")