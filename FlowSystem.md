# Alur Sistem Prediksi Harga Bitcoin (FlowSystem)
> Hak Cipta: **Shdyt13**

Dokumen ini menjelaskan alur kerja (sistem flow) secara berurutan mulai dari interaksi awal pengguna di Frontend hingga pemrosesan model di Backend.

---

## 1. Start Sistem & Antarmuka (Frontend)
- **File:** `frontend/src/App.jsx`
- **Proses:** Pengguna membuka sistem melalui browser. Sistem memuat UI berupa *Dashboard* analisis *Time-Series*. State dasar seperti `fileTrain`, `fileTest`, dan `isDarkMode` diinisialisasi oleh React.

## 2. Unggah Berkas Data (File Upload)
- **File:** `frontend/src/App.jsx`
- **Fungsi:** `handleFileTrainChange(e)` dan `handleFileTestChange(e)`
- **Proses:** 
  1. Pengguna mengunggah dataset CSV (*Training Data* dan *Testing Data*).
  2. Fungsi pembantu `validateCsvFile(file, label)` berjalan untuk memvalidasi bahwa file berekstensi `.csv` dan ukurannya tidak melebihi 5MB.
  3. Berkas yang valid disimpan di *state* lokal React.

## 3. Eksekusi Prediksi & Pengiriman Data (HTTP Request)
- **File:** `frontend/src/App.jsx`
- **Fungsi:** `handlePredict(algo)`
- **Proses:** Saat pengguna mengklik salah satu tombol eksekusi (contoh: *Run XGBoost Prediction*), fungsi ini membungkus kedua file CSV ke dalam form data (`FormData`) bersama dengan nama algoritma yang dipilih, lalu mengirimkannya ke *backend* via Axios `POST` ke endpoint `http://localhost:5000/api/predict`.

## 4. Penerimaan Data di Backend (Endpoint)
- **File:** `backend/api.py`
- **Fungsi:** `predict_api()`
- **Proses:** 
  Endpoint Flask menerima request. File CSV dibaca menggunakan pustaka Pandas (`pd.read_csv`) dan dikonversi menjadi Dataframe. Nama algoritma yang diminta oleh pengguna dicatat untuk menentukan model mana yang akan dilatih.

## 5. Pra-pemrosesan Data (Data Preprocessing)
Pemrosesan data dipecah menjadi dua tahap utama:

### A. Pembersihan Data & Scaling
- **File:** `backend/utils/preprocessing.py` & `backend/utils/model_utils.py`
- **Fungsi:** 
  - `clean_bitcoin_data(filepath)`: Membersihkan string yang mengandung format koma atau simbol mata uang (`Rp `), mengurutkan secara kronologis berdasarkan tanggal, lalu mengonversinya menjadi tipe *Float*. Menangani *missing values* menggunakan teknik *Forward Fill* (`ffill`) dan `dropna()`.
  - `MinMaxScaler(feature_range=(0, 1))`: Melakukan normalisasi (Min-Max Scaling). Objek `MinMaxScaler` di-*fit* hanya pada data *training* (`scaler.fit(train_data_raw)`), kemudian digunakan untuk men-*transform* data *training* maupun data *testing* agar berada di rentang skala `[0, 1]` tanpa kebocoran informasi (*zero data leakage*).

### B. Pembentukan Sliding Window
- **File:** `backend/utils/windowing.py`
- **Fungsi:** `create_sequences(data, window_size=30, horizon=h, target_col_index=1)`
- **Proses:** Mengubah data berbentuk deret waktu menjadi pasangan matriks Input ($X$) berukuran `(N, 30, 2)` dan Target ($y$) berbasis jendela geser 30 hari dengan pergeseran target sesuai horizon prediksi ($h \in \{1, 3, 7\}$). Untuk data pengujian, sistem menggabungkan ekor data latih sepanjang `WINDOW_SIZE + horizon - 1` hari sebagai konteks awal agar jumlah sekuens uji tetap konsisten sebanyak 140 sekuens pada setiap horizon.

## 6. Training Model & Forecasting
- **File:** `backend/utils/model_utils.py`
- **Fungsi:** `run_all_models_pipeline(train_path, test_path, models_dir)`
- **Proses:** Menerapkan strategi *Direct Multi-Step Forecasting* secara deterministik (`seed = 42`, `TF_DETERMINISTIC_OPS=1`, `n_jobs=1`) untuk memprediksi harga secara independen pada horizon $h=1, 3, 7$:
  - **XGBoost Tunggal**: Meratakan jendela masukan menjadi vektor 60 fitur dan melatih `XGBRegressor` dengan optimasi hyperparameter `RandomizedSearchCV` (`n_iter=5`, `cv=3`, `scoring='neg_mean_absolute_error'`).
  - **LSTM Tunggal**: Membangun jaringan saraf sekuensial Keras/TensorFlow yang terdiri dari 2 layer LSTM (masing-masing 50 unit, `Dropout(0.2)`), `Dense(25)`, dan `Dense(1)`, dilatih dengan optimizer `Adam` (`loss='mean_squared_error'`, `batch_size=32`, `validation_split=0.1`) hingga maksimum 30 *Epochs* menggunakan `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`.
  - **Hybrid LSTM-XGBoost (*Residual-Correction Stacking*)**: Menggunakan model LSTM sebagai *base-learner* untuk memprediksi tren utama, menghitung residu pada data latih ($e_t = y_t - \hat{y}_{\text{LSTM},t}$), lalu melatih `XGBRegressor` (*meta-learner*) melalui `RandomizedSearchCV` untuk memprediksi koreksi residu ($\hat{y}_{\text{Hybrid}} = \hat{y}_{\text{LSTM}} + \hat{e}_{\text{XGBoost}}$).

## 7. Inverse Transform & Prediksi Aktual (Testing)
- **File:** `backend/utils/model_utils.py`
- **Proses:** Hasil prediksi dari *Testing Data* (*out-of-sample*) masih berada pada skala 0-1. Skala ini dikembalikan ke nilai harga asli Bitcoin (Rupiah) menggunakan fungsi `scaler.inverse_transform()`.

## 8. Evaluasi Akurasi (Error Metrics)
- **File:** `backend/utils/model_utils.py`
- **Fungsi:** `calculate_metrics(actual, predicted)`
- **Proses:** Membandingkan nilai Prediksi terhadap nilai Aktual untuk menghitung metrik statistik:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **MAPE** (Mean Absolute Percentage Error)

## 9. Render Visualisasi & Komparasi di Frontend
- **File:** `backend/api.py` $\rightarrow$ `frontend/src/App.jsx`
- **Proses:**
  1. *Backend* menyatukan metrik evaluasi, nilai history, dan array proyeksi masa depan ke dalam format JSON yang terstruktur.
  2. JSON dikirimkan kembali ke pengguna.
  3. *Frontend* menangkap *Response* ini dan menggunakan komponen `react-chartjs-2` untuk merender grafik garis dinamis. Hasil evaluasi model (MAPE dll) ditampilkan dalam visualisasi Diagram Batang (*Bar Chart*) di tab "Model Comparison".
