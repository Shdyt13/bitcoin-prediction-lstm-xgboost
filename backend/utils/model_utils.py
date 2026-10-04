# ---------------------------------------------------------
# Copyright (c) Shdyt13
# ---------------------------------------------------------
"""
Shared model definitions and training/evaluation helpers.
Digunakan oleh api.py (online prediction) dan train.py (offline training).
"""

import os
import random

import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Fix random seeds for reproducible runs.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 30
FORECAST_HORIZONS = [1, 3, 7]
CLOSE_COL_INDEX = 1  # Index of the "Close" column
VALID_ALGORITHMS = {"xgboost", "lstm", "hybrid"}

XGB_SEARCH_N_JOBS = max(1, (os.cpu_count() or 2) - 1)


# ==========================================
# === IMPLEMENTASI LSTM ===
# ==========================================
def build_lstm_model():
    """Membangun arsitektur LSTM."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 2)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_lstm(X_train, y_train, epochs, patience):
    """Melatih model LSTM dengan validasi split 10% dan early stopping."""
    tf.random.set_seed(SEED)
    model = build_lstm_model()
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        batch_size=32, epochs=epochs, validation_split=0.1,
        callbacks=[early_stopping], verbose=0,
    )
    return model

def run_lstm(X_train, y_train, X_test, future_input, epochs, patience):
    """
    Fungsi utama untuk menjalankan model LSTM tunggal.
    Melakukan proses Training Model dan memberikan prediksi.
    """
    future_3d = future_input.reshape(1, WINDOW_SIZE, 2)

    model = train_lstm(X_train, y_train, epochs, patience)
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    future_pred = float(model.predict(future_3d, verbose=0)[0][0])

    return model, y_pred_test, future_pred


# ==========================================
# === IMPLEMENTASI XGBOOST ===
# ==========================================
def tune_xgboost(X_train, y_train, param_grid, n_iter):
    """Melakukan Hyperparameter Tuning XGBoost menggunakan RandomizedSearchCV."""
    n_iter = min(n_iter, _grid_size(param_grid))
    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=SEED),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=min(3, len(X_train)),
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=XGB_SEARCH_N_JOBS,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def run_xgboost(X_train, y_train, X_test, future_input, param_grid, n_iter):
    """
    Fungsi utama untuk menjalankan model XGBoost tunggal.
    Melakukan proses Training Model dan memberikan prediksi.
    """
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    future_2d = future_input.reshape(1, -1)

    model = tune_xgboost(X_train_2d, y_train, param_grid, n_iter)
    y_pred_test = model.predict(X_test_2d)
    future_pred = float(model.predict(future_2d)[0])

    return model, y_pred_test, future_pred


# ==========================================
# === IMPLEMENTASI HYBRID LSTM-XGBOOST ===
# ==========================================
def run_hybrid(X_train, y_train, X_test, future_input, param_grid, n_iter, epochs, patience):
    """
    Fungsi utama untuk melatih model Hybrid LSTM + XGBoost (Residual Correction).
    LSTM dilatih sebagai base-learner, kemudian XGBoost memprediksi sisa kesalahan (residual error).
    """
    future_3d = future_input.reshape(1, WINDOW_SIZE, 2)
    future_2d = future_input.reshape(1, -1)

    # 1. Tahap Pemodelan Base-Learner (LSTM)
    lstm_model = train_lstm(X_train, y_train, epochs, patience)
    y_pred_lstm_train = lstm_model.predict(X_train, verbose=0).flatten()
    y_pred_lstm_test = lstm_model.predict(X_test, verbose=0).flatten()
    future_lstm_pred = float(lstm_model.predict(future_3d, verbose=0)[0][0])

    # 2. Kalkulasi Residu
    residual_train = y_train - y_pred_lstm_train

    # 3. Tahap Meta-Learner (XGBoost) memprediksi Residual
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    residual_model = tune_xgboost(X_train_2d, residual_train, param_grid, n_iter)
    
    # 4. Prediksi Akhir
    residual_test_pred = residual_model.predict(X_test_2d)
    y_pred_test = y_pred_lstm_test + residual_test_pred

    future_residual_pred = float(residual_model.predict(future_2d)[0])
    future_pred = future_lstm_pred + future_residual_pred

    return (lstm_model, residual_model), y_pred_test, future_pred


def _grid_size(param_grid):
    """Total kombinasi parameter unik."""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size

def inverse_transform_close(scaler, scaled_values, close_col_index=CLOSE_COL_INDEX):
    """Kembalikan nilai hasil prediksi ke dalam skala harga asli (Denormalisasi)."""
    dummy = np.zeros((len(scaled_values), 2))
    dummy[:, close_col_index] = scaled_values
    return scaler.inverse_transform(dummy)[:, close_col_index]

def evaluate_predictions(y_true, y_pred):
    """Menghitung metrik evaluasi model (RMSE, MAE, MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.any(y_true == 0):
        mape = float("nan")
    else:
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape,
        "evaluasi_detail": {
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist(),
        },
    }