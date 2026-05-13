import numpy as np

def create_sequences(data, window_size, horizon, target_col_index=1):
    """
    Fungsi untuk membuat sekuens/windowing data time series.
    Menggunakan pendekatan "Direct Multi-Horizon Forecasting".
    """
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i : (i + window_size)])
        target_index = i + window_size + horizon - 1
        
        # Mengambil nilai target berdasarkan indeks kolom (Default: 1 untuk Close)
        y.append(data[target_index, target_col_index])
        
    return np.array(X), np.array(y)