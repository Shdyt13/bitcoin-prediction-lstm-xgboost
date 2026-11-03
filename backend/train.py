# ---------------------------------------------------------
# Copyright (c) Shdyt13
# ---------------------------------------------------------
"""
Offline training script for the Bitcoin price forecasting models.
Diselaraskan 100% dengan alur pemrosesan pada halaman web (backend/api.py).
"""

import argparse
import functools
import json
import logging
import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.model_utils import (
    CLOSE_COL_INDEX,
    FORECAST_HORIZONS,
    WINDOW_SIZE,
    evaluate_predictions,
    extract_training_metadata,
    inverse_transform_close,
    run_hybrid,
    run_lstm,
    run_xgboost,
)
from utils.preprocessing import DataFormatError, clean_bitcoin_data
from utils.windowing import create_sequences

MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_TRAIN_CSV = os.path.join(BASE_DIR, "data", "Data Training 2020 - 2025.csv")
DEFAULT_TEST_CSV = os.path.join(BASE_DIR, "data", "Data Testing 2026.csv")

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
XGB_N_ITER = 5
LSTM_EPOCHS = 30
LSTM_PATIENCE = 3

ALGORITHM_RUNNERS = {
    "xgboost": functools.partial(run_xgboost, param_grid=XGB_PARAM_GRID, n_iter=XGB_N_ITER),
    "lstm": functools.partial(run_lstm, epochs=LSTM_EPOCHS, patience=LSTM_PATIENCE),
    "hybrid": functools.partial(
        run_hybrid,
        param_grid=XGB_PARAM_GRID,
        n_iter=XGB_N_ITER,
        epochs=LSTM_EPOCHS,
        patience=LSTM_PATIENCE,
    ),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bitcoin forecasting models.")
    parser.add_argument(
        "--train", default=DEFAULT_TRAIN_CSV, help="Path to the training CSV."
    )
    parser.add_argument(
        "--test", default=DEFAULT_TEST_CSV, help="Path to the testing CSV."
    )
    parser.add_argument(
        "--models-dir", default=MODELS_DIR, help="Directory to write trained models/metrics into."
    )
    return parser.parse_args()


def prepare_test_sequences(train_data, test_data, scaler, horizon):
    """Build windowed test sequences using the tail of the training data as context."""
    context_size = WINDOW_SIZE + horizon - 1
    if len(train_data) < context_size:
        raise DataFormatError(
            f"Training data has only {len(train_data)} rows, but at least "
            f"{context_size} are needed for horizon={horizon}."
        )
    past_data = train_data[-context_size:]

    combined_data = np.vstack((past_data, test_data))
    scaled_data = scaler.transform(combined_data)

    return create_sequences(scaled_data, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX)


def _save_json_atomic(payload, path):
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _save_trained_model(algorithm, model_obj, horizon, models_dir):
    """Simpan objek model hasil pelatihan ke direktori models."""
    if algorithm == "xgboost":
        model_obj.save_model(os.path.join(models_dir, f"xgb_model_h{horizon}.json"))
    elif algorithm == "lstm":
        model_obj.save(os.path.join(models_dir, f"lstm_model_h{horizon}.keras"))
    elif algorithm == "hybrid":
        lstm_base, hybrid_xgb = model_obj
        lstm_base.save(os.path.join(models_dir, f"hybrid_lstm_model_h{horizon}.keras"))
        hybrid_xgb.save_model(os.path.join(models_dir, f"hybrid_model_h{horizon}.json"))


def train_models(train_path, test_path, models_dir=MODELS_DIR):
    logger.info("Memulai proses training model (XGBoost, LSTM, & Hybrid)...")

    if not os.path.exists(train_path):
        logger.error("File training tidak ditemukan di %s", train_path)
        return False
    if not os.path.exists(test_path):
        logger.error("File testing tidak ditemukan di %s", test_path)
        return False

    os.makedirs(models_dir, exist_ok=True)

    # ==========================================
    # === PRA-PEMROSESAN DATA ===
    # ==========================================
    logger.info("Membaca dan membersihkan dataset...")
    try:
        train_df, train_clean, train_date_col, _ = clean_bitcoin_data(train_path)
        test_df, test_clean, date_column, close_column = clean_bitcoin_data(test_path)
    except DataFormatError as exc:
        logger.error("Gagal membersihkan data: %s", exc)
        return False

    train_data = train_clean.values
    test_data = test_clean.values

    # Skala data (Normalisasi Min-Max) menggunakan data latih
    logger.info("Melakukan scaling (Min-Max) dan menyimpan scaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    scaled_train = scaler.transform(train_data)
    last_test_date = test_df[date_column].iloc[-1]
    test_dates_all = test_df[date_column].dt.strftime("%d %b %Y").tolist()

    # Menyiapkan data 30 hari terakhir untuk keperluan visualisasi grafik (sama dengan api.py)
    latest_data = test_df.tail(30)
    history_dates = latest_data[date_column].dt.strftime("%d %b %Y").tolist()
    history_prices = latest_data[close_column].tolist()

    # ==========================================
    # === TRAINING MODEL & PREDIKSI (Sama dengan Web API) ===
    # ==========================================
    for algorithm, run_algorithm in ALGORITHM_RUNNERS.items():
        logger.info("=== Menjalankan Algoritma: %s ===", algorithm.upper())
        predictions = {}
        evaluations = {}
        analisis_training = {}
        horizons_sequence_info = {}
        future_dates = []
        future_prices = []
        info_data = {}

        for horizon in FORECAST_HORIZONS:
            logger.info("--- Memproses %s | Target Horizon: %d Hari ---", algorithm.upper(), horizon)

            try:
                # Membentuk sekuens Sliding Window
                X_train, y_train = create_sequences(
                    scaled_train, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX
                )
                X_test, y_test = prepare_test_sequences(train_data, test_data, scaler, horizon)
            except (ValueError, DataFormatError) as exc:
                logger.error("Melewati horizon %d: %s", horizon, exc)
                continue

            horizons_sequence_info[str(horizon)] = {
                "horizon_days": int(horizon),
                "window_size": int(WINDOW_SIZE),
                "context_stitching_rows": int(WINDOW_SIZE + horizon - 1),
                "sekuens_training": int(len(X_train)),
                "sekuens_testing": int(len(X_test)),
            }

            if horizon == 1:
                info_data = {
                    "total_baris": len(train_data) + len(test_data),
                    "data_training_mentah": len(train_data),
                    "data_testing_mentah": len(test_data),
                    "sekuens_training": len(X_train),
                    "sekuens_testing": len(X_test),
                    "window_size": int(WINDOW_SIZE),
                    "train_start_date": train_df[train_date_col].iloc[0].strftime("%d %b %Y"),
                    "train_end_date": train_df[train_date_col].iloc[-1].strftime("%d %b %Y"),
                    "test_start_date": test_df[date_column].iloc[0].strftime("%d %b %Y"),
                    "test_end_date": test_df[date_column].iloc[-1].strftime("%d %b %Y"),
                    "scaler_min_open": float(scaler.data_min_[0]),
                    "scaler_max_open": float(scaler.data_max_[0]),
                    "scaler_min_close": float(scaler.data_min_[1]),
                    "scaler_max_close": float(scaler.data_max_[1]),
                    "test_dates": test_dates_all,
                }

            # Data historis 30 hari terakhir untuk proyeksi ke masa depan (Blind Forecasting)
            future_input_raw = test_data[-WINDOW_SIZE:]
            future_input = scaler.transform(future_input_raw)

            try:
                # Menjalankan Algoritma Terpilih (XGBoost, LSTM, atau Hybrid)
                model_obj, y_pred_test_scaled, future_prediction_scaled = run_algorithm(
                    X_train, y_train, X_test, future_input
                )
                analisis_training[str(horizon)] = extract_training_metadata(algorithm, model_obj)
                _save_trained_model(algorithm, model_obj, horizon, models_dir)

                # Denormalisasi untuk mendapatkan hasil pada rentang harga aktual
                y_test_actual = inverse_transform_close(scaler, y_test)
                y_pred_test_actual = inverse_transform_close(scaler, y_pred_test_scaled)
                evaluations[str(horizon)] = evaluate_predictions(y_test_actual, y_pred_test_actual)

                future_prediction = float(
                    inverse_transform_close(scaler, np.array([future_prediction_scaled]))[0]
                )
                predictions[str(horizon)] = future_prediction

                future_dates.append(
                    (last_test_date + pd.Timedelta(days=horizon)).strftime("%d %b %Y")
                )
                future_prices.append(future_prediction)
            except Exception:
                logger.exception("Training %s gagal pada horizon %d", algorithm, horizon)
                continue

        info_data["horizons_sequence_info"] = horizons_sequence_info

        if evaluations:
            _save_json_atomic(
                {
                    "status": "success",
                    "hasil_prediksi": predictions,
                    "evaluasi": evaluations,
                    "analisis_training": analisis_training,
                    "info_data": info_data,
                    "grafik": {
                        "history_dates": history_dates,
                        "history_prices": history_prices,
                        "future_dates": future_dates,
                        "future_prices": future_prices,
                    },
                },
                os.path.join(models_dir, f"metrics_{algorithm}.json"),
            )

    logger.info("Semua model berhasil dilatih dan disimpan di '%s'.", models_dir)
    return True


if __name__ == "__main__":
    args = parse_args()
    success = train_models(args.train, args.test, args.models_dir)
    sys.exit(0 if success else 1)
