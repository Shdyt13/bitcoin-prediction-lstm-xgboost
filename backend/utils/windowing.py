# ---------------------------------------------------------
# Copyright (c) Shdyt13
# ---------------------------------------------------------
import numpy as np

# ==========================================
# === PRA-PEMROSESAN DATA (SLIDING WINDOW) ===
# ==========================================
def create_sequences(data, window_size, horizon, target_col_index=1):
    """
    Fungsi untuk membuat sekuens data time series (Sliding Window).
    Menggunakan pendekatan "Direct Multi-Horizon Forecasting".
    
    Args:
        data: array NumPy berdimensi 2 (fitur Open dan Close)
        window_size: jumlah hari historis untuk melihat ke belakang
        horizon: hari ke-n ke depan yang ingin diprediksi
        target_col_index: indeks kolom untuk target Y (default: 1 untuk Close)
        
    Returns:
        X (array fitur), y (array target prediksi)
    """
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i : (i + window_size)])
        target_index = i + window_size + horizon - 1
        y.append(data[target_index, target_col_index])
        
    return np.array(X), np.array(y)