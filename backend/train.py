import os
import random
import numpy as np
import xgboost as xgb
import tensorflow as tf
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 🔷 Import dari utils
from utils.preprocessing import clean_bitcoin_data
from utils.windowing import create_sequences

# Stabilitas & Reproducibility
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

WINDOW_SIZE = 30
MODELS_DIR = 'models'
DATA_PATH = 'data/Data Bitcoin Bersih.csv'

os.makedirs(MODELS_DIR, exist_ok=True)

def train_models():
    print("🚀 Memulai proses training model (XGBoost, LSTM, & Hybrid)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File dataset tidak ditemukan di {DATA_PATH}")
        return

    # 🔷 1. Preprocessing menggunakan Utils
    print("📊 Membaca dan membersihkan dataset...")
    df, df_bersih, _, _ = clean_bitcoin_data(DATA_PATH)
    total_data = len(df_bersih)
    
    # 🔷 2. Pembagian Data
    split_idx = int(total_data * 0.8)
    train_data_raw = df_bersih.iloc[:split_idx].values
    test_data_raw = df_bersih.iloc[split_idx:].values
    
    # 🔷 3. Normalisasi & Simpan Scaler
    print("⚖️ Melakukan scaling dan menyimpan scaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data_raw)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    scaled_train = scaler.transform(train_data_raw)
    
    eval_xgb = {}
    eval_lstm = {}
    eval_hybrid = {} # Dictionary baru untuk Hybrid
    info_data_dict = {}

    for horizon in [1, 3, 7]:
        print(f"\n⏳ --- Memproses Target Horizon: {horizon} Hari ---")
        
        # Windowing Data
        X_train, y_train = create_sequences(scaled_train, WINDOW_SIZE, horizon)
        
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

        # Setup Actual Test Data for evaluation
        dummy_test_aktual = np.zeros((len(y_test), 2))
        dummy_test_aktual[:, 1] = y_test
        y_test_asli = scaler.inverse_transform(dummy_test_aktual)[:, 1]

        # ==========================================
        # 🟢 1. TRAINING XGBOOST
        # ==========================================
        print(f"🌲 Melatih XGBoost untuk horizon {horizon}...")
        X_train_xgb = X_train.reshape(X_train.shape[0], -1) 
        X_test_xgb = X_test.reshape(X_test.shape[0], -1)
        
        xgb_base = xgb.XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        random_search_xgb = RandomizedSearchCV(
            estimator=xgb_base, param_distributions=param_grid, 
            n_iter=15, cv=3, scoring='neg_mean_absolute_error', 
            random_state=42, n_jobs=-1
        )
        random_search_xgb.fit(X_train_xgb, y_train)
        best_xgb = random_search_xgb.best_estimator_
        best_xgb.save_model(os.path.join(MODELS_DIR, f'xgb_model_h{horizon}.json'))
        
        y_pred_xgb_scaled = best_xgb.predict(X_test_xgb)
        dummy_test_pred_xgb = np.zeros((len(y_pred_xgb_scaled), 2))
        dummy_test_pred_xgb[:, 1] = y_pred_xgb_scaled
        y_pred_xgb_asli = scaler.inverse_transform(dummy_test_pred_xgb)[:, 1]
        
        eval_xgb[str(horizon)] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test_asli, y_pred_xgb_asli))), 
            "MAE": float(mean_absolute_error(y_test_asli, y_pred_xgb_asli)), 
            "MAPE": float(mean_absolute_percentage_error(y_test_asli, y_pred_xgb_asli) * 100),
            "evaluasi_detail": {"actual": y_test_asli.tolist(), "predicted": y_pred_xgb_asli.tolist()}
        }

        # ==========================================
        # 🧠 2. TRAINING LSTM
        # ==========================================
        print(f"🧠 Melatih LSTM untuk horizon {horizon}...")
        tf.random.set_seed(42)
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 2)))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(LSTM(50, return_sequences=False))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(Dense(25))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_lstm.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        model_lstm.save(os.path.join(MODELS_DIR, f'lstm_model_h{horizon}.keras'))
        
        y_pred_lstm_scaled = model_lstm.predict(X_test, verbose=0).flatten()
        dummy_test_pred_lstm = np.zeros((len(y_pred_lstm_scaled), 2))
        dummy_test_pred_lstm[:, 1] = y_pred_lstm_scaled
        y_pred_lstm_asli = scaler.inverse_transform(dummy_test_pred_lstm)[:, 1]
        
        eval_lstm[str(horizon)] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test_asli, y_pred_lstm_asli))), 
            "MAE": float(mean_absolute_error(y_test_asli, y_pred_lstm_asli)), 
            "MAPE": float(mean_absolute_percentage_error(y_test_asli, y_pred_lstm_asli) * 100),
            "evaluasi_detail": {"actual": y_test_asli.tolist(), "predicted": y_pred_lstm_asli.tolist()}
        }

        # ==========================================
        # 🤖 3. TRAINING HYBRID (LSTM -> XGBoost)
        # ==========================================
        print(f"🤖 Melatih Stacking HYBRID untuk horizon {horizon}...")
        
        # a. Dapatkan prediksi LSTM untuk data Training
        y_pred_lstm_train_scaled = model_lstm.predict(X_train, verbose=0).flatten()
        
        # b. Buat Fitur Hybrid: Gabungkan X_train_xgb (2D) dengan tebakan LSTM
        # Membentuk array baru dimana kolom terakhir adalah hasil prediksi LSTM
        X_train_hybrid = np.hstack((X_train_xgb, y_pred_lstm_train_scaled.reshape(-1, 1)))
        X_test_hybrid = np.hstack((X_test_xgb, y_pred_lstm_scaled.reshape(-1, 1)))
        
        # c. Train model XGBoost menggunakan fitur Hybrid ini
        hybrid_base = xgb.XGBRegressor(random_state=42)
        random_search_hybrid = RandomizedSearchCV(
            estimator=hybrid_base, param_distributions=param_grid, 
            n_iter=15, cv=3, scoring='neg_mean_absolute_error', 
            random_state=42, n_jobs=-1
        )
        random_search_hybrid.fit(X_train_hybrid, y_train)
        best_hybrid = random_search_hybrid.best_estimator_
        
        # d. Simpan model Hybrid
        best_hybrid.save_model(os.path.join(MODELS_DIR, f'hybrid_model_h{horizon}.json'))
        
        # e. Evaluasi Model Hybrid pada Data Test
        y_pred_hybrid_scaled = best_hybrid.predict(X_test_hybrid)
        dummy_test_pred_hybrid = np.zeros((len(y_pred_hybrid_scaled), 2))
        dummy_test_pred_hybrid[:, 1] = y_pred_hybrid_scaled
        y_pred_hybrid_asli = scaler.inverse_transform(dummy_test_pred_hybrid)[:, 1]

        eval_hybrid[str(horizon)] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test_asli, y_pred_hybrid_asli))), 
            "MAE": float(mean_absolute_error(y_test_asli, y_pred_hybrid_asli)), 
            "MAPE": float(mean_absolute_percentage_error(y_test_asli, y_pred_hybrid_asli) * 100),
            "evaluasi_detail": {"actual": y_test_asli.tolist(), "predicted": y_pred_hybrid_asli.tolist()}
        }

    print("\n💾 Menyimpan semua hasil evaluasi...")
    with open(os.path.join(MODELS_DIR, 'metrics_xgboost.json'), 'w') as f:
        json.dump({"evaluasi": eval_xgb, "info_data": info_data_dict}, f)
    with open(os.path.join(MODELS_DIR, 'metrics_lstm.json'), 'w') as f:
        json.dump({"evaluasi": eval_lstm, "info_data": info_data_dict}, f)
    with open(os.path.join(MODELS_DIR, 'metrics_hybrid.json'), 'w') as f:
        json.dump({"evaluasi": eval_hybrid, "info_data": info_data_dict}, f)
        
    print("✅✅ SEMUA MODEL (XGB, LSTM, HYBRID) BERHASIL DILATIH DAN DISIMPAN! ✅✅")

if __name__ == '__main__':
    train_models()