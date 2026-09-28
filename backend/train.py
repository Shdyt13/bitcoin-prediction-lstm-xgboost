"""
Offline training script for the Bitcoin price forecasting models.

Trains three models (XGBoost, LSTM, and a Hybrid LSTM+XGBoost residual
corrector) for three forecast horizons (1, 3, and 7 days) using a
manually split training and testing dataset (two separate CSV files),
evaluates each on the test data, and persists the trained models,
scaler, and evaluation metrics to the `models/` directory.

Usage:
    python train.py --train data/train.csv --test data/test.csv
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.model_utils import (
    CLOSE_COL_INDEX,
    FORECAST_HORIZONS,
    WINDOW_SIZE,
    evaluate_predictions,
    inverse_transform_close,
    run_hybrid,
    run_lstm,
    run_xgboost,
)
from utils.preprocessing import DataFormatError, clean_bitcoin_data
from utils.windowing import create_sequences

MODELS_DIR = "models"

# Wider search space/training budget than api.py, since this runs offline.
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}
XGB_N_ITER = 15
LSTM_EPOCHS = 50
LSTM_PATIENCE = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bitcoin forecasting models.")
    parser.add_argument("--train", default="data/train.csv", help="Path to the training CSV.")
    parser.add_argument("--test", default="data/test.csv", help="Path to the testing CSV.")
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
    """Write JSON to `path` via a temp file + rename, so a crash mid-write can't corrupt it."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def train_models(train_path, test_path, models_dir=MODELS_DIR):
    logger.info("Memulai proses training model (XGBoost, LSTM, & Hybrid)...")

    if not os.path.exists(train_path):
        logger.error("File training tidak ditemukan di %s", train_path)
        return False
    if not os.path.exists(test_path):
        logger.error("File testing tidak ditemukan di %s", test_path)
        return False

    os.makedirs(models_dir, exist_ok=True)

    # 1. Load and clean both manually split datasets.
    logger.info("Membaca dan membersihkan dataset...")
    try:
        _, train_clean, _, _ = clean_bitcoin_data(train_path)
        _, test_clean, _, _ = clean_bitcoin_data(test_path)
    except DataFormatError as exc:
        logger.error("Gagal membersihkan data: %s", exc)
        return False

    train_data = train_clean.values
    test_data = test_clean.values

    # 2. Fit the scaler on training data only, then persist it for inference.
    logger.info("Melakukan scaling dan menyimpan scaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    scaled_train = scaler.transform(train_data)

    eval_xgb, eval_lstm, eval_hybrid = {}, {}, {}
    info_data = {}

    for horizon in FORECAST_HORIZONS:
        logger.info("--- Memproses Target Horizon: %d Hari ---", horizon)

        try:
            X_train, y_train = create_sequences(
                scaled_train, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX
            )
            X_test, y_test = prepare_test_sequences(train_data, test_data, scaler, horizon)
        except (ValueError, DataFormatError) as exc:
            logger.error("Melewati horizon %d: %s", horizon, exc)
            continue

        if horizon == 1:
            info_data = {
                "total_baris": len(train_data) + len(test_data),
                "data_training_mentah": len(train_data),
                "data_testing_mentah": len(test_data),
                "sekuens_training": len(X_train),
                "sekuens_testing": len(X_test),
            }

        y_test_actual = inverse_transform_close(scaler, y_test)
        future_input = X_test[-1]  # last window; future value itself is unused during training

        try:
            # XGBoost
            logger.info("Melatih XGBoost untuk horizon %d...", horizon)
            xgb_model, y_pred_xgb, _ = run_xgboost(
                X_train, y_train, X_test, future_input, XGB_PARAM_GRID, XGB_N_ITER
            )
            xgb_model.save_model(os.path.join(models_dir, f"xgb_model_h{horizon}.json"))
            eval_xgb[str(horizon)] = evaluate_predictions(
                y_test_actual, inverse_transform_close(scaler, y_pred_xgb)
            )

            # LSTM
            logger.info("Melatih LSTM untuk horizon %d...", horizon)
            lstm_model, y_pred_lstm, _ = run_lstm(
                X_train, y_train, X_test, future_input, LSTM_EPOCHS, LSTM_PATIENCE
            )
            lstm_model.save(os.path.join(models_dir, f"lstm_model_h{horizon}.keras"))
            eval_lstm[str(horizon)] = evaluate_predictions(
                y_test_actual, inverse_transform_close(scaler, y_pred_lstm)
            )

            # Hybrid (LSTM + XGBoost residual correction)
            logger.info("Melatih Hybrid (Residual Correction) untuk horizon %d...", horizon)
            (_, hybrid_xgb), y_pred_hybrid, _ = run_hybrid(
                X_train, y_train, X_test, future_input,
                XGB_PARAM_GRID, XGB_N_ITER, LSTM_EPOCHS, LSTM_PATIENCE,
            )
            hybrid_xgb.save_model(os.path.join(models_dir, f"hybrid_model_h{horizon}.json"))
            eval_hybrid[str(horizon)] = evaluate_predictions(
                y_test_actual, inverse_transform_close(scaler, y_pred_hybrid)
            )
        except Exception:
            logger.exception("Training gagal pada horizon %d", horizon)
            continue

    # Persist evaluation metrics, one JSON file per algorithm.
    logger.info("Menyimpan semua hasil evaluasi...")
    for name, evaluasi in (("xgboost", eval_xgb), ("lstm", eval_lstm), ("hybrid", eval_hybrid)):
        if not evaluasi:
            logger.warning("Tidak ada hasil evaluasi untuk '%s', file metrik dilewati.", name)
            continue
        _save_json_atomic(
            {"evaluasi": evaluasi, "info_data": info_data},
            os.path.join(models_dir, f"metrics_{name}.json"),
        )

    logger.info("Semua model berhasil dilatih dan disimpan di '%s'.", models_dir)
    return True


if __name__ == "__main__":
    args = parse_args()
    success = train_models(args.train, args.test, args.models_dir)
    sys.exit(0 if success else 1)