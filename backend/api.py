"""
Flask API for Bitcoin price forecasting.

Exposes a single endpoint, POST /api/predict, which accepts a training
and a testing CSV upload plus an `algorithm` field ('xgboost', 'lstm',
or 'hybrid'), trains the requested model(s) for three forecast horizons
(1, 3, and 7 days), evaluates them on the uploaded test split, and
returns predictions, evaluation metrics, and chart-ready data as JSON.

The hybrid algorithm is an LSTM + XGBoost residual corrector: XGBoost is
trained on the raw windowed features to predict the LSTM's residual
error, and the final prediction is the LSTM forecast plus the predicted
residual. This matches the approach used in train.py.

Run directly:
    python api.py
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from utils.preprocessing import clean_bitcoin_data
from utils.windowing import create_sequences

# Fix random seeds for reproducible runs.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 30
FORECAST_HORIZONS = [1, 3, 7]
CLOSE_COL_INDEX = 1  # index of the "Close" column inside the cleaned feature array
VALID_ALGORITHMS = {"xgboost", "lstm", "hybrid"}

# Shared hyperparameter search space for both the standalone and hybrid XGBoost models.
XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def build_lstm_model():
    """Create and compile the LSTM architecture shared by the LSTM and Hybrid models."""
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 2)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(X_train, y_train):
    """Train the LSTM model with early stopping."""
    tf.random.set_seed(SEED)

    model = build_lstm_model()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=30,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    return model


def tune_xgboost(X_train, y_train):
    """Run randomized hyperparameter search for an XGBoost regressor and return the best model."""
    random_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=SEED),
        param_distributions=XGB_PARAM_GRID,
        n_iter=5,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=-1,
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
            "predicted": y_pred.tolist(),
        },
    }


def prepare_test_sequences(train_data, test_data, scaler, horizon):
    """
    Build windowed test sequences using the tail of the training data as
    historical context, so the first test window has enough history.
    """
    context_size = WINDOW_SIZE + horizon - 1
    past_data = train_data[-context_size:]

    combined_data = np.vstack((past_data, test_data))
    scaled_data = scaler.transform(combined_data)

    return create_sequences(
        scaled_data, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX
    )


def run_xgboost(X_train, y_train, X_test, future_input):
    """Train a standalone XGBoost model and return (test predictions, future prediction)."""
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    future_input_2d = future_input.reshape(1, -1)

    model = tune_xgboost(X_train_2d, y_train)

    y_pred_test = model.predict(X_test_2d)
    future_prediction = float(model.predict(future_input_2d)[0])

    return y_pred_test, future_prediction


def run_lstm(X_train, y_train, X_test, future_input):
    """Train a standalone LSTM model and return (test predictions, future prediction)."""
    future_input_3d = future_input.reshape(1, WINDOW_SIZE, 2)

    model = train_lstm(X_train, y_train)

    y_pred_test = model.predict(X_test, verbose=0).flatten()
    future_prediction = float(model.predict(future_input_3d, verbose=0)[0][0])

    return y_pred_test, future_prediction


def run_hybrid(X_train, y_train, X_test, future_input):
    """
    Train an LSTM + XGBoost residual corrector and return (test predictions,
    future prediction).

    XGBoost is trained on the raw windowed features to predict the LSTM's
    residual error (y_train - LSTM prediction), not the price itself. The
    final prediction is the LSTM forecast plus the predicted residual.
    """
    future_input_3d = future_input.reshape(1, WINDOW_SIZE, 2)
    future_input_2d = future_input.reshape(1, -1)

    lstm_model = train_lstm(X_train, y_train)

    y_pred_lstm_train = lstm_model.predict(X_train, verbose=0).flatten()
    y_pred_lstm_test = lstm_model.predict(X_test, verbose=0).flatten()
    future_lstm_prediction = float(
        lstm_model.predict(future_input_3d, verbose=0)[0][0]
    )

    residual_train = y_train - y_pred_lstm_train

    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    residual_model = tune_xgboost(X_train_2d, residual_train)

    residual_test_prediction = residual_model.predict(X_test_2d)
    y_pred_test = y_pred_lstm_test + residual_test_prediction

    future_residual_prediction = float(
        residual_model.predict(future_input_2d)[0]
    )
    future_prediction = future_lstm_prediction + future_residual_prediction

    return y_pred_test, future_prediction


ALGORITHM_RUNNERS = {
    "xgboost": run_xgboost,
    "lstm": run_lstm,
    "hybrid": run_hybrid,
}


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """Run model training, evaluation, and future forecasting for the requested algorithm."""
    if "dataset_train" not in request.files or "dataset_test" not in request.files:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": (
                        "File dataset_train dan dataset_test harus diupload lengkap."
                    ),
                }
            ),
            400,
        )

    file_train = request.files["dataset_train"]
    file_test = request.files["dataset_test"]
    algorithm = request.form.get("algorithm")

    if algorithm not in VALID_ALGORITHMS:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Algorithm tidak valid. Gunakan xgboost, lstm, atau hybrid.",
                }
            ),
            400,
        )

    run_algorithm = ALGORITHM_RUNNERS[algorithm]

    try:
        # Save uploaded datasets.
        train_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "train_" + file_train.filename
        )
        test_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "test_" + file_test.filename
        )
        file_train.save(train_path)
        file_test.save(test_path)

        # Load and preprocess datasets.
        _, train_clean, _, _ = clean_bitcoin_data(train_path)
        test_df, test_clean, date_column, close_column = clean_bitcoin_data(test_path)

        train_data = train_clean.values
        test_data = test_clean.values

        # Fit scaler using training data only.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)

        predictions = {}
        evaluations = {}
        future_dates = []
        future_prices = []
        info_data = {}

        last_test_date = test_df[date_column].iloc[-1]

        for horizon in FORECAST_HORIZONS:
            X_train, y_train = create_sequences(
                scaled_train, WINDOW_SIZE, horizon, target_col_index=CLOSE_COL_INDEX
            )
            X_test, y_test = prepare_test_sequences(
                train_data, test_data, scaler, horizon
            )

            if horizon == 1:
                info_data = {
                    "total_baris": len(train_data) + len(test_data),
                    "data_training_mentah": len(train_data),
                    "data_testing_mentah": len(test_data),
                    "sekuens_training": len(X_train),
                    "sekuens_testing": len(X_test),
                }

            # Prepare the latest window for future forecasting.
            future_input_raw = test_data[-WINDOW_SIZE:]
            future_input = scaler.transform(future_input_raw)

            y_pred_test_scaled, future_prediction_scaled = run_algorithm(
                X_train, y_train, X_test, future_input
            )

            # Evaluate predictions on the test dataset.
            y_test_actual = inverse_transform_close(scaler, y_test)
            y_pred_test_actual = inverse_transform_close(scaler, y_pred_test_scaled)
            evaluations[str(horizon)] = evaluate_predictions(
                y_test_actual, y_pred_test_actual
            )

            # Convert future prediction back to the original scale.
            future_prediction = float(
                inverse_transform_close(scaler, np.array([future_prediction_scaled]))[0]
            )
            predictions[str(horizon)] = future_prediction

            future_dates.append(
                (last_test_date + pd.Timedelta(days=horizon)).strftime("%d %b %Y")
            )
            future_prices.append(future_prediction)

        # Prepare the latest 30 test observations for visualization.
        latest_data = test_df.tail(30)
        history_dates = latest_data[date_column].dt.strftime("%d %b %Y").tolist()
        history_prices = latest_data[close_column].tolist()

        return jsonify(
            {
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
            }
        )

    except Exception as error:
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)