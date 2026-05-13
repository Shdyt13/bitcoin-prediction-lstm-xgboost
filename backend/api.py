import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocessing import clean_bitcoin_data
from utils.windowing import create_sequences

# Stabilitas & Reproducibility
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
WINDOW_SIZE = 30

@app.route('/api/predict', methods=['POST'])
def predict_api():
    # --- 1. Menangkap 2 File dari Request ---
    if 'dataset_train' not in request.files or 'dataset_test' not in request.files:
        return jsonify({"status": "error", "message": "File dataset_train dan dataset_test harus diupload lengkap."}), 400
    
    file_train = request.files['dataset_train']
    file_test = request.files['dataset_test']
    algo = request.form.get('algorithm') # 'xgboost', 'lstm', atau 'hybrid'
    
    try:
        # Menyimpan file sementara
        filepath_train = os.path.join(app.config['UPLOAD_FOLDER'], "train_" + file_train.filename)
        file_train.save(filepath_train)
        
        filepath_test = os.path.join(app.config['UPLOAD_FOLDER'], "test_" + file_test.filename)
        file_test.save(filepath_test)
        
        # --- 2. Membaca & Membersihkan Kedua File Terpisah ---
        df_train, df_train_bersih, _, _ = clean_bitcoin_data(filepath_train)
        df_test, df_test_bersih, kolom_tanggal, kolom_close = clean_bitcoin_data(filepath_test)
        
        train_data_raw = df_train_bersih.values
        test_data_raw = df_test_bersih.values
        
        # --- 3. Normalisasi (HANYA fit pada Data Train) ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data_raw) 
        
        scaled_train = scaler.transform(train_data_raw)
        
        hasil_prediksi = {}
        evaluasi_model = {}
        future_dates = []
        future_prices = []
        last_test_date = df_test[kolom_tanggal].iloc[-1]
        info_data = {}

        col_close_idx = 1 

        for horizon in [1, 3, 7]:
            # --- 4a. Transformasi Windowing (Train) ---
            X_train, y_train = create_sequences(scaled_train, WINDOW_SIZE, horizon, target_col_index=col_close_idx)
            
            # --- 4b. Menjahit Data (Train ujung ke Test awal) ---
            # Mengambil (30 + horizon - 1) hari terakhir dari data train sebagai konteks historis untuk tes pertama
            past_data_needed = train_data_raw[-(WINDOW_SIZE + horizon - 1):]
            test_data_combined = np.vstack((past_data_needed, test_data_raw))
            scaled_test = scaler.transform(test_data_combined) 
            
            # --- 4c. Transformasi Windowing (Test) ---
            X_test, y_test = create_sequences(scaled_test, WINDOW_SIZE, horizon, target_col_index=col_close_idx)
            
            if horizon == 1:
                info_data = {
                    "total_baris": len(train_data_raw) + len(test_data_raw),
                    "data_training_mentah": len(train_data_raw),
                    "data_testing_mentah": len(test_data_raw),
                    "sekuens_training": len(X_train),
                    "sekuens_testing": len(X_test)
                }
            
            # --- 4d. Data untuk Forecasting Masa Depan (Blind Forecast) ---
            # Mengambil 30 hari paling akhir dari DATA TEST
            input_prediksi_raw = test_data_raw[-WINDOW_SIZE:]
            input_prediksi = scaler.transform(input_prediksi_raw)
            
            # Setup parameter tuning XGBoost
            xgb_param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            # --- 5. Pelatihan & Pengujian Model ---
            if algo == 'xgboost':
                X_train_xgb = X_train.reshape(X_train.shape[0], -1) 
                X_test_xgb = X_test.reshape(X_test.shape[0], -1)
                input_2d = input_prediksi.reshape(1, -1)
                
                xgb_base = xgb.XGBRegressor(random_state=42)
                random_search = RandomizedSearchCV(
                    estimator=xgb_base, param_distributions=xgb_param_grid, 
                    n_iter=5, cv=3, scoring='neg_mean_absolute_error', 
                    random_state=42, n_jobs=-1
                )
                random_search.fit(X_train_xgb, y_train)
                model = random_search.best_estimator_
                
                y_pred_test_scaled = model.predict(X_test_xgb)
                prediksi_scaled = float(model.predict(input_2d)[0])
                
            elif algo == 'lstm':
                tf.random.set_seed(42)
                input_3d = input_prediksi.reshape(1, WINDOW_SIZE, 2) 
                
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 2)))
                model.add(Dropout(0.2))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1, callbacks=[early_stopping], verbose=0)
                
                y_pred_test_scaled = model.predict(X_test, verbose=0).flatten()
                prediksi_scaled = float(model.predict(input_3d, verbose=0)[0][0])
                
            elif algo == 'hybrid':
                tf.random.set_seed(42)
                input_3d = input_prediksi.reshape(1, WINDOW_SIZE, 2) 
                
                # a. Melatih LSTM
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 2)))
                model_lstm.add(Dropout(0.2))
                model_lstm.add(LSTM(50, return_sequences=False))
                model_lstm.add(Dropout(0.2))
                model_lstm.add(Dense(25))
                model_lstm.add(Dense(1))
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                model_lstm.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1, callbacks=[early_stopping], verbose=0)
                
                y_pred_lstm_train = model_lstm.predict(X_train, verbose=0).flatten()
                y_pred_lstm_test = model_lstm.predict(X_test, verbose=0).flatten()
                prediksi_lstm_future = float(model_lstm.predict(input_3d, verbose=0)[0][0])

                # b. Menyiapkan Fitur Hybrid
                X_train_xgb = X_train.reshape(X_train.shape[0], -1) 
                X_test_xgb = X_test.reshape(X_test.shape[0], -1)
                input_2d = input_prediksi.reshape(1, -1)

                X_train_hybrid = np.hstack((X_train_xgb, y_pred_lstm_train.reshape(-1, 1)))
                X_test_hybrid = np.hstack((X_test_xgb, y_pred_lstm_test.reshape(-1, 1)))
                input_hybrid = np.hstack((input_2d, np.array([[prediksi_lstm_future]])))

                # c. Melatih Meta-Model XGBoost
                xgb_base = xgb.XGBRegressor(random_state=42)
                random_search = RandomizedSearchCV(
                    estimator=xgb_base, param_distributions=xgb_param_grid, 
                    n_iter=5, cv=3, scoring='neg_mean_absolute_error', 
                    random_state=42, n_jobs=-1
                )
                random_search.fit(X_train_hybrid, y_train)
                model_hybrid = random_search.best_estimator_

                y_pred_test_scaled = model_hybrid.predict(X_test_hybrid)
                prediksi_scaled = float(model_hybrid.predict(input_hybrid)[0])

            
            # --- 6. Evaluasi Model (Denormalisasi) ---
            dummy_test_aktual = np.zeros((len(y_test), 2))
            dummy_test_aktual[:, col_close_idx] = y_test
            y_test_asli = scaler.inverse_transform(dummy_test_aktual)[:, col_close_idx]
            
            dummy_test_pred = np.zeros((len(y_pred_test_scaled), 2))
            dummy_test_pred[:, col_close_idx] = y_pred_test_scaled
            y_pred_test_asli = scaler.inverse_transform(dummy_test_pred)[:, col_close_idx]
            
            rmse = float(np.sqrt(mean_squared_error(y_test_asli, y_pred_test_asli)))
            mae = float(mean_absolute_error(y_test_asli, y_pred_test_asli))
            mape = float(mean_absolute_percentage_error(y_test_asli, y_pred_test_asli) * 100)
            
            evaluasi_model[str(horizon)] = {
                "RMSE": rmse, 
                "MAE": mae, 
                "MAPE": mape,
                "evaluasi_detail": {
                    "actual": y_test_asli.tolist(),
                    "predicted": y_pred_test_asli.tolist()
                }
            }
            
            # --- 7. Prediksi Masa Depan (Blind Forecasting) ---
            dummy_future = np.zeros((1, 2))
            dummy_future[0, col_close_idx] = prediksi_scaled
            prediksi_asli = float(scaler.inverse_transform(dummy_future)[0, col_close_idx])
            
            hasil_prediksi[str(horizon)] = prediksi_asli
            future_dates.append((last_test_date + pd.Timedelta(days=horizon)).strftime('%d %b %Y'))
            future_prices.append(prediksi_asli)

        # Untuk keperluan grafik historis, tampilkan 30 hari terakhir dari DATA TESTING
        df_terakhir = df_test.tail(30)
        history_dates = df_terakhir[kolom_tanggal].dt.strftime('%d %b %Y').tolist()
        history_prices = df_terakhir[kolom_close].tolist()
        
        return jsonify({
            "status": "success",
            "hasil_prediksi": hasil_prediksi,
            "evaluasi": evaluasi_model,
            "info_data": info_data,
            "grafik": {
                "history_dates": history_dates,
                "history_prices": history_prices,
                "future_dates": future_dates,
                "future_prices": future_prices
            }
        })
                                        
    except Exception as e:
        import traceback
        traceback.print_exc() # Ini akan mencetak detail error berwarna merah di terminal
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)