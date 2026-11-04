# ---------------------------------------------------------
# Copyright (c) Shdyt13
# ---------------------------------------------------------
"""
Shared model definitions and training/evaluation helpers.
Digunakan oleh api.py (online prediction) dan train.py (offline training).
"""

import os
import random

# Harus diset SEBELUM tensorflow diimpor agar deterministik antar build/run.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
try:
    # Satu thread + op deterministik: hasil identik pada mesin/build berbeda.
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
except Exception as exc:  # pragma: no cover
    print(f"Peringatan: gagal mengaktifkan determinisme TensorFlow: {exc}")

WINDOW_SIZE = 30
FORECAST_HORIZONS = [1, 3, 7]
CLOSE_COL_INDEX = 1  # Index of the "Close" column
VALID_ALGORITHMS = {"xgboost", "lstm", "hybrid"}

XGB_SEARCH_N_JOBS = 1


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
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    loss_list = [float(v) for v in history.history.get("loss", [])]
    val_loss_list = [float(v) for v in history.history.get("val_loss", [])]
    best_epoch_idx = int(np.argmin(val_loss_list)) if val_loss_list else 0

    model._training_history = {
        "max_epochs": int(epochs),
        "epochs_trained": len(loss_list),
        "best_epoch": best_epoch_idx + 1,
        "patience": int(patience),
        "batch_size": 32,
        "validation_split": 0.1,
        "optimizer": "adam",
        "loss_function": "mean_squared_error",
        "train_samples": int(round(len(X_train) * 0.9)),
        "val_samples": int(len(X_train) - int(round(len(X_train) * 0.9))),
        "best_train_loss": float(loss_list[best_epoch_idx]) if loss_list else None,
        "best_val_loss": float(val_loss_list[best_epoch_idx]) if val_loss_list else None,
        "final_train_loss": float(loss_list[-1]) if loss_list else None,
        "final_val_loss": float(val_loss_list[-1]) if val_loss_list else None,
        "stopped_early": len(loss_list) < int(epochs),
        "epoch_history": [
            {
                "epoch": idx + 1,
                "train_loss": float(loss_list[idx]),
                "val_loss": float(val_loss_list[idx]) if idx < len(val_loss_list) else None,
                "is_best_epoch": (idx == best_epoch_idx),
            }
            for idx in range(len(loss_list))
        ],
    }
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
    cv_folds = min(3, len(X_train))
    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=SEED, n_jobs=1),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=XGB_SEARCH_N_JOBS,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    cv_res = search.cv_results_
    candidates = []
    num_candidates = len(cv_res.get("params", []))
    for i in range(num_candidates):
        params_i = cv_res["params"][i]
        fold_scores = []
        for f_idx in range(cv_folds):
            key = f"split{f_idx}_test_score"
            if key in cv_res:
                fold_scores.append(float(-cv_res[key][i]))
        candidates.append({
            "iteration": i + 1,
            "rank": int(cv_res["rank_test_score"][i]),
            "n_estimators": int(params_i.get("n_estimators", 0)),
            "learning_rate": float(params_i.get("learning_rate", 0.0)),
            "max_depth": int(params_i.get("max_depth", 0)),
            "subsample": float(params_i.get("subsample", 0.0)),
            "colsample_bytree": float(params_i.get("colsample_bytree", 0.0)),
            "fold_1_mae": fold_scores[0] if len(fold_scores) > 0 else None,
            "fold_2_mae": fold_scores[1] if len(fold_scores) > 1 else None,
            "fold_3_mae": fold_scores[2] if len(fold_scores) > 2 else None,
            "mean_cv_mae": float(-cv_res["mean_test_score"][i]),
            "std_cv_mae": float(cv_res["std_test_score"][i]),
            "mean_fit_time_sec": float(cv_res["mean_fit_time"][i]),
        })

    best_params = search.best_params_
    best_model._tuning_report = {
        "search_method": "RandomizedSearchCV",
        "n_iter": int(n_iter),
        "cv_folds": int(cv_folds),
        "scoring": "neg_mean_absolute_error",
        "best_cv_mae": float(-search.best_score_),
        "best_params": {
            "n_estimators": int(best_params.get("n_estimators", 0)),
            "learning_rate": float(best_params.get("learning_rate", 0.0)),
            "max_depth": int(best_params.get("max_depth", 0)),
            "subsample": float(best_params.get("subsample", 0.0)),
            "colsample_bytree": float(best_params.get("colsample_bytree", 0.0)),
        },
        "candidates": candidates,
    }
    return best_model


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

    residual_model._residual_stats = {
        "train_residual_mean_scaled": float(np.mean(residual_train)),
        "train_residual_std_scaled": float(np.std(residual_train)),
        "train_residual_mae_scaled": float(np.mean(np.abs(residual_train))),
        "train_residual_min_scaled": float(np.min(residual_train)),
        "train_residual_max_scaled": float(np.max(residual_train)),
        "test_residual_mean_scaled": float(np.mean(residual_test_pred)),
        "test_residual_mae_scaled": float(np.mean(np.abs(residual_test_pred))),
        "future_lstm_base_scaled": float(future_lstm_pred),
        "future_xgb_residual_scaled": float(future_residual_pred),
        "future_hybrid_final_scaled": float(future_pred),
    }

    return (lstm_model, residual_model), y_pred_test, future_pred


def extract_training_metadata(algorithm, model_obj):
    """Ekstrak metadata proses training & hyperparameter tuning dari objek model."""
    if algorithm == "xgboost":
        return {
            "xgboost_tuning": getattr(model_obj, "_tuning_report", None),
        }
    if algorithm == "lstm":
        return {
            "lstm_training": getattr(model_obj, "_training_history", None),
        }
    if algorithm == "hybrid" and isinstance(model_obj, tuple) and len(model_obj) == 2:
        lstm_base, residual_xgb = model_obj
        return {
            "lstm_training": getattr(lstm_base, "_training_history", None),
            "xgboost_tuning": getattr(residual_xgb, "_tuning_report", None),
            "residual_stats": getattr(residual_xgb, "_residual_stats", None),
        }
    return {}


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
