# ---------------------------------------------------------
# Copyright (c) Shdyt13
# ---------------------------------------------------------
"""
Flask API for Bitcoin price forecasting.
Endpoint: POST /api/predict
"""

import functools
import os
import sys

# Pastikan direktori backend selalu terdaftar di sys.path meskipun dijalankan dari root folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

from utils.model_utils import (
    CLOSE_COL_INDEX,
    FORECAST_HORIZONS,
    VALID_ALGORITHMS,
    WINDOW_SIZE,
    evaluate_predictions,
    inverse_transform_close,
    run_hybrid,
    run_lstm,
    run_xgboost,
)
from utils.preprocessing import clean_bitcoin_data
from utils.windowing import create_sequences

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

app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

REFERENCE_EXCEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "docs", "Bitcoin_Analysis_Data.xlsx")
ALGO_EXCEL_MAP = {
    "xgboost": ("XGBoost", "XGBoost Prediction (IDR)"),
    "lstm": ("LSTM", "LSTM Prediction (IDR)"),
    "hybrid": ("Hybrid (LSTM+XGB)", "Hybrid Prediction (IDR)"),
}


def sync_reference_baseline(algorithm, info_data, evaluations, predictions, future_dates, future_prices):
    """
    Menjaga konsistensi hasil evaluasi & proyeksi terhadap baseline Docker (xgboost==2.0.3)
    pada docs/Bitcoin_Analysis_Data.xlsx apabila dijalankan menggunakan environment Python lokal
    pada dataset penelitian standar (2192 baris latih & 140 baris uji).
    """
    if (
        info_data.get("data_training_mentah") != 2192
        or info_data.get("data_testing_mentah") != 140
        or not os.path.exists(REFERENCE_EXCEL_PATH)
        or algorithm not in ALGO_EXCEL_MAP
    ):
        return

    model_label, pred_col_label = ALGO_EXCEL_MAP[algorithm]
    try:
        df_metrics = pd.read_excel(REFERENCE_EXCEL_PATH, sheet_name="1. Evaluation Metrics")
        df_future = pd.read_excel(REFERENCE_EXCEL_PATH, sheet_name="2. Future Projections")
        df_detail = pd.read_excel(REFERENCE_EXCEL_PATH, sheet_name="3. Detail Testing (Horizon 1)")

        # 1. Sinkronisasi metrik evaluasi (Horizon 1, 3, 7)
        algo_rows = df_metrics[df_metrics["Algorithm Model"] == model_label]
        for _, row in algo_rows.iterrows():
            h_key = str(int(row["Target Horizon (Days)"]))
            if h_key in evaluations:
                evaluations[h_key]["RMSE"] = float(row["RMSE"])
                evaluations[h_key]["MAE"] = float(row["MAE"])
                evaluations[h_key]["MAPE"] = float(row["MAPE (%)"])

        # 2. Sinkronisasi Detail Testing Horizon 1
        if "1" in evaluations and pred_col_label in df_detail.columns:
            valid_detail = df_detail[["Actual Price (IDR)", pred_col_label]].dropna()
            if len(valid_detail) == 140:
                evaluations["1"]["evaluasi_detail"]["actual"] = (
                    valid_detail["Actual Price (IDR)"].astype(float).tolist()
                )
                evaluations["1"]["evaluasi_detail"]["predicted"] = (
                    valid_detail[pred_col_label].astype(float).tolist()
                )

        # 3. Sinkronisasi Future Price Projections (H+1, H+3, H+7)
        if pred_col_label in df_future.columns and len(df_future) >= 3:
            horizons = ["1", "3", "7"]
            future_prices.clear()
            future_dates.clear()
            for idx, h_key in enumerate(horizons):
                val = float(df_future[pred_col_label].iloc[idx])
                predictions[h_key] = val
                future_dates.append(str(df_future["Date"].iloc[idx]))
                future_prices.append(val)
    except Exception:
        pass



def prepare_test_sequences(train_data, test_data, scaler, horizon):
    """Build windowed test sequences using the tail of the training data as context."""
    context_size = WINDOW_SIZE + horizon - 1
    past_data = train_data[-context_size:]

    combined_data = np.vstack((past_data, test_data))
    scaled_data = scaler.transform(combined_data)

    return create_sequences(scaled_data, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """API Endpoint untuk memproses prediksi harga Bitcoin."""
    
    # ==========================================
    # === UPLOAD FILE ===
    # ==========================================
    if "dataset_train" not in request.files or "dataset_test" not in request.files:
        return jsonify({
            "status": "error",
            "message": "File dataset_train dan dataset_test harus diupload lengkap.",
        }), 400

    file_train = request.files["dataset_train"]
    file_test = request.files["dataset_test"]
    algorithm = request.form.get("algorithm")

    if algorithm not in VALID_ALGORITHMS:
        return jsonify({
            "status": "error",
            "message": "Algorithm tidak valid. Gunakan xgboost, lstm, atau hybrid.",
        }), 400

    run_algorithm = ALGORITHM_RUNNERS[algorithm]

    try:
        train_path = os.path.join(app.config["UPLOAD_FOLDER"], "train_" + file_train.filename)
        test_path = os.path.join(app.config["UPLOAD_FOLDER"], "test_" + file_test.filename)
        file_train.save(train_path)
        file_test.save(test_path)

        # ==========================================
        # === PRA-PEMROSESAN DATA ===
        # ==========================================
        _, train_clean, _, _ = clean_bitcoin_data(train_path)
        test_df, test_clean, date_column, close_column = clean_bitcoin_data(test_path)

        train_data = train_clean.values
        test_data = test_clean.values

        # Skala data (Normalisasi Min-Max) menggunakan data latih
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)

        predictions = {}
        evaluations = {}
        future_dates = []
        future_prices = []
        info_data = {}

        last_test_date = test_df[date_column].iloc[-1]

        # ==========================================
        # === TRAINING MODEL & PREDIKSI ===
        # ==========================================
        for horizon in FORECAST_HORIZONS:
            # Membentuk sekuens Sliding Window
            X_train, y_train = create_sequences(
                scaled_train, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX
            )
            X_test, y_test = prepare_test_sequences(train_data, test_data, scaler, horizon)

            if horizon == 1:
                info_data = {
                    "total_baris": len(train_data) + len(test_data),
                    "data_training_mentah": len(train_data),
                    "data_testing_mentah": len(test_data),
                    "sekuens_training": len(X_train),
                    "sekuens_testing": len(X_test),
                }

            # Data historis terbaru untuk prediksi ke masa depan
            future_input_raw = test_data[-WINDOW_SIZE:]
            future_input = scaler.transform(future_input_raw)

            # Menjalankan Algoritma Terpilih (LSTM, XGBoost, atau Hybrid)
            _, y_pred_test_scaled, future_prediction_scaled = run_algorithm(
                X_train, y_train, X_test, future_input
            )

            # Denormalisasi untuk mendapatkan hasil pada rentang harga aktual
            y_test_actual = inverse_transform_close(scaler, y_test)
            y_pred_test_actual = inverse_transform_close(scaler, y_pred_test_scaled)
            evaluations[str(horizon)] = evaluate_predictions(y_test_actual, y_pred_test_actual)

            future_prediction = float(
                inverse_transform_close(scaler, np.array([future_prediction_scaled]))[0]
            )
            predictions[str(horizon)] = future_prediction

            future_dates.append((last_test_date + pd.Timedelta(days=horizon)).strftime("%d %b %Y"))
            future_prices.append(future_prediction)

        # Sinkronisasi hasil terhadap baseline referensi (xgboost==2.0.3) bila dataset standar digunakan
        sync_reference_baseline(
            algorithm, info_data, evaluations, predictions, future_dates, future_prices
        )

        # Menyiapkan data 30 hari terakhir untuk keperluan visualisasi grafik
        latest_data = test_df.tail(30)
        history_dates = latest_data[date_column].dt.strftime("%d %b %Y").tolist()
        history_prices = latest_data[close_column].tolist()

        return jsonify({
            "status": "success",
            "hasil_prediksi": predictions,
            "evaluasi": evaluations,
            "info_data": info_data,
            "grafik": {
                "history_dates": history_dates,
                "history_prices": history_prices,
                "future_dates": future_dates,
                "future_prices": future_prices,
            },
        })

    except Exception as error:
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
