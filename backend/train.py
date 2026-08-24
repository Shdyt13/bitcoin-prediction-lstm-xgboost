"""
Training script for the Bitcoin price forecasting models.

Trains three models (XGBoost, LSTM, and a Hybrid LSTM+XGBoost residual
corrector) for three forecast horizons (1, 3, and 7 days), evaluates each
on a held-out test split, and persists the trained models, scaler, and
evaluation metrics to the `models/` directory.

Run directly:
    python train.py
"""

import os
import random
import json

import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocessing import clean_bitcoin_data
from utils.windowing import create_sequences

# Fix random seeds for reproducible training runs.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 30
FORECAST_HORIZONS = [1, 3, 7]
MODELS_DIR = 'models'
DATA_PATH = 'data/Data Bitcoin Bersih.csv'
TRAIN_SPLIT_RATIO = 0.8
CLOSE_COL_INDEX = 1  # index of the "Close" column inside the cleaned feature array

# Shared hyperparameter search space for both the standalone and hybrid XGBoost models.
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

os.makedirs(MODELS_DIR, exist_ok=True)


def build_lstm_model():
    """Create and compile the LSTM architecture shared by the LSTM and Hybrid models."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 2)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def tune_xgboost(X_train, y_train):
    """Run randomized hyperparameter search for an XGBoost regressor and return the best model."""
    random_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=SEED),
        param_distributions=XGB_PARAM_GRID,
        n_iter=15, cv=3, scoring='neg_mean_absolute_error',
        random_state=SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_


def inverse_transform_close(scaler, scaled_values, close_col_index=CLOSE_COL_INDEX):
    """
    Undo MinMax scaling for a 1D array of scaled "Close" values.

    The scaler was fit on a 2-column array (Open, Close), so we rebuild a
    dummy array with the same shape before inverse-transforming, then pull
    out just the Close column.
    """
    dummy = np.zeros((len(scaled_values), 2))
    dummy[:, close_col_index] = scaled_values
    return scaler.inverse_transform(dummy)[:, close_col_index]


def evaluate_predictions(y_true, y_pred):
    """Compute RMSE, MAE, and MAPE (%) plus the raw values for later plotting."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "evaluasi_detail": {
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist()
        }
    }


def train_models():
    print("Memulai proses training model (XGBoost, LSTM, & Hybrid)...")

    if not os.path.exists(DATA_PATH):
        print(f"Error: File dataset tidak ditemukan di {DATA_PATH}")
        return

    # 1. Load and clean the raw dataset.
    print("Membaca dan membersihkan dataset...")
    _, df_bersih, _, _ = clean_bitcoin_data(DATA_PATH)
    total_data = len(df_bersih)

    # 2. Chronological train/test split (no shuffling, to avoid look-ahead bias).
    split_idx = int(total_data * TRAIN_SPLIT_RATIO)
    train_data_raw = df_bersih.iloc[:split_idx].values
    test_data_raw = df_bersih.iloc[split_idx:].values

    # 3. Fit the scaler on training data only, then persist it for inference.
    print("Melakukan scaling dan menyimpan scaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data_raw)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    scaled_train = scaler.transform(train_data_raw)

    eval_xgb = {}
    eval_lstm = {}
    eval_hybrid = {}
    info_data_dict = {}

    for horizon in FORECAST_HORIZONS:
        print(f"\n--- Memproses Target Horizon: {horizon} Hari ---")

        # Build training windows.
        X_train, y_train = create_sequences(scaled_train, WINDOW_SIZE, horizon)

        # Stitch the tail of the training data onto the test data so the first
        # test window has enough history, then scale using the training-fitted scaler.
        past_data_needed = train_data_raw[-(WINDOW_SIZE + horizon - 1):]
        test_data_combined = np.vstack((past_data_needed, test_data_raw))
        scaled_test = scaler.transform(test_data_combined)

        X_test, y_test = create_sequences(scaled_test, WINDOW_SIZE, horizon)

        if horizon == 1:
            info_data_dict = {
                "total_baris": total_data,
                "data_training_mentah": len(train_data_raw),
                "data_testing_mentah": len(test_data_raw),
                "sekuens_training": len(X_train),
                "sekuens_testing": len(X_test)
            }

        # Ground-truth test values in original price scale, reused for every model below.
        y_test_asli = inverse_transform_close(scaler, y_test)

        X_train_xgb = X_train.reshape(X_train.shape[0], -1)
        X_test_xgb = X_test.reshape(X_test.shape[0], -1)

        # ------------------------------------------------------------------
        # XGBoost
        # ------------------------------------------------------------------
        print(f"Melatih XGBoost untuk horizon {horizon}...")
        best_xgb = tune_xgboost(X_train_xgb, y_train)
        best_xgb.save_model(os.path.join(MODELS_DIR, f'xgb_model_h{horizon}.json'))

        y_pred_xgb_scaled = best_xgb.predict(X_test_xgb)
        y_pred_xgb_asli = inverse_transform_close(scaler, y_pred_xgb_scaled)
        eval_xgb[str(horizon)] = evaluate_predictions(y_test_asli, y_pred_xgb_asli)

        # ------------------------------------------------------------------
        # LSTM
        # ------------------------------------------------------------------
        print(f"Melatih LSTM untuk horizon {horizon}...")
        tf.random.set_seed(SEED)
        model_lstm = build_lstm_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_lstm.fit(
            X_train, y_train,
            batch_size=32, epochs=50, validation_split=0.1,
            callbacks=[early_stopping], verbose=0
        )
        model_lstm.save(os.path.join(MODELS_DIR, f'lstm_model_h{horizon}.keras'))

        y_pred_lstm_train_scaled = model_lstm.predict(X_train, verbose=0).flatten()
        y_pred_lstm_test_scaled = model_lstm.predict(X_test, verbose=0).flatten()
        y_pred_lstm_asli = inverse_transform_close(scaler, y_pred_lstm_test_scaled)
        eval_lstm[str(horizon)] = evaluate_predictions(y_test_asli, y_pred_lstm_asli)

        # ------------------------------------------------------------------
        # Hybrid (LSTM + XGBoost residual correction)
        #
        # XGBoost is trained on the raw windowed features to predict the
        # LSTM's residual error (y_train - LSTM prediction), not the price
        # itself. At inference time, the final prediction is the LSTM
        # forecast plus the predicted residual. This matches the approach
        # used in api.py.
        # ------------------------------------------------------------------
        print(f"Melatih Hybrid (Residual Correction) untuk horizon {horizon}...")

        residual_train = y_train - y_pred_lstm_train_scaled

        best_hybrid = tune_xgboost(X_train_xgb, residual_train)
        best_hybrid.save_model(os.path.join(MODELS_DIR, f'hybrid_model_h{horizon}.json'))

        y_pred_residual_test = best_hybrid.predict(X_test_xgb)
        y_pred_hybrid_scaled = y_pred_lstm_test_scaled + y_pred_residual_test
        y_pred_hybrid_asli = inverse_transform_close(scaler, y_pred_hybrid_scaled)
        eval_hybrid[str(horizon)] = evaluate_predictions(y_test_asli, y_pred_hybrid_asli)

    # Persist evaluation metrics for each model, one JSON file per algorithm.
    print("\nMenyimpan semua hasil evaluasi...")
    with open(os.path.join(MODELS_DIR, 'metrics_xgboost.json'), 'w') as f:
        json.dump({"evaluasi": eval_xgb, "info_data": info_data_dict}, f)
    with open(os.path.join(MODELS_DIR, 'metrics_lstm.json'), 'w') as f:
        json.dump({"evaluasi": eval_lstm, "info_data": info_data_dict}, f)
    with open(os.path.join(MODELS_DIR, 'metrics_hybrid.json'), 'w') as f:
        json.dump({"evaluasi": eval_hybrid, "info_data": info_data_dict}, f)

    print("Semua model (XGBoost, LSTM, Hybrid) berhasil dilatih dan disimpan.")


if __name__ == '__main__':
    train_models()