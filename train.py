"""Train the final bike demand model and save it to models/.

Run from the project root:
    python train.py

Input:  data/processed/train.csv and data/processed/test.csv
        (created by the Data_Preparation notebook)
Output: models/bike_demand_model.joblib and models/model_config.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'data' / 'processed' / 'train.csv'
TEST_PATH = ROOT / 'data' / 'processed' / 'test.csv'
MODEL_PATH = ROOT / 'models' / 'bike_demand_model.joblib'
CONFIG_PATH = ROOT / 'models' / 'model_config.json'

# Order of the features is the order of the model input
FEATURES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'hum', 'windspeed']
TARGET = 'cnt'

# Parameters chosen in the Modeling notebook (section 5)
PARAMS = {
    'loss': 'poisson',
    'early_stopping': False,
    'random_state': 42,
    'learning_rate': 0.06405753233449049,
    'max_iter': 575,
    'max_leaf_nodes': 30,
    'min_samples_leaf': 24,
    'l2_regularization': 5.2477466025838915,
}


def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=['dteday'])
    test = pd.read_csv(TEST_PATH, parse_dates=['dteday'])
    return pd.concat([train, test], ignore_index=True)


def check_data(df):
    assert len(df) == 17379, f'Expected 17379 rows, got {len(df)}'
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    assert not missing_columns, f'Missing columns: {missing_columns}'
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0, 'Missing values found'
    assert set(df['weathersit'].unique()) <= {1, 2, 3}, 'weathersit must be 1-3'
    assert (df[TARGET] >= 0).all(), 'Poisson loss needs a non-negative target'


def train_model(df):
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(df[FEATURES], df[TARGET])
    return model


def save_model(model, df):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    config = {
        'features': FEATURES,
        'params': PARAMS,
        'target': TARGET,
        'train_period': [str(df['dteday'].min().date()), str(df['dteday'].max().date())],
        'n_rows': int(len(df)),
        'sklearn_version': sklearn.__version__,
        'note': 'weathersit 4 is merged into 3 before prediction',
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding='utf-8')


def main():
    df = load_data()
    check_data(df)
    print(f'Rows: {len(df)}, features: {len(FEATURES)}')

    model = train_model(df)
    save_model(model, df)

    # Reload the saved model and compare its predictions with the trained one
    loaded = joblib.load(MODEL_PATH)
    same = np.allclose(loaded.predict(df[FEATURES]), model.predict(df[FEATURES]))
    print(f'Saved {MODEL_PATH.name} and {CONFIG_PATH.name}; reload check: {same}')
    print(f'scikit-learn version: {sklearn.__version__}')


if __name__ == '__main__':
    main()
