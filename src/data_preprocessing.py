import pandas as pd
import numpy as np

# Standard NASA CMAPSS column names
COLUMNS = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
          [f's_{i}' for i in range(1, 22)]

# Sensors that stay flat in FD001
DROP_COLS = ['setting_1', 'setting_2', 'setting_3', 
             's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

def load_data(filepath):
    """Loads the raw text file into a DataFrame."""
    return pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)

def add_rul(df, max_rul=125):
    """Calculates Remaining Useful Life and clips it to reduce noise."""
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)
    
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    
    # Target clipping
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df

def engineer_features(df, window_size=30):
    """Generates rolling mean and standard deviation for active sensors."""
    df = df.drop(columns=DROP_COLS, errors='ignore')
    sensor_cols = [c for c in df.columns if 's_' in c]
    
    for col in sensor_cols:
        df[f'{col}_roll_mean'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        df[f'{col}_roll_std'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).std().fillna(0)
        )
    return df

def process_training_data(filepath):
    """Full pipeline for training data."""
    df = load_data(filepath)
    df = add_rul(df)
    df = engineer_features(df)
    return df