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
- **File:** `backend/utils/preprocessing.py`
- **Fungsi:** 
  - `load_and_clean_data(df)`: Membersihkan string yang mengandung format koma atau simbol mata uang, lalu mengonversinya menjadi tipe *Float*. Menangani *missing values* (data kosong) menggunakan teknik *Forward Fill* (`ffill`).
  - `scale_data(train_df, test_df)`: Melakukan normalisasi (Min-Max Scaling). Kelas `MinMaxScaler` di-*fit* pada data *training*, kemudian digunakan untuk men-*transform* data *training* maupun data *testing* agar berada di rentang skala 0 hingga 1.

### B. Pembentukan Sliding Window
- **File:** `backend/utils/windowing.py`
- **Fungsi:** `create_sliding_window(scaled_train, scaled_test, window_size=30)`
- **Proses:** Mengubah data berbentuk deret waktu (sekuensial 1D) menjadi pasangan matrix Input ($X$) dan Target ($y$) berbasis jendela geser sebesar 30 hari. Ini adalah bentuk format yang dibutuhkan oleh algoritma Machine Learning & Deep Learning.

## 6. Training Model & Forecasting
- **File:** `backend/utils/model_utils.py`
- **Proses:** Menerapkan strategi *Direct Multi-Step Forecasting* untuk memprediksi harga secara independen pada horizon yang berbeda ($h=1, 3, 7$). Berdasarkan request, sistem memanggil fungsi:
  - **`train_xgboost_direct()`**: Membangun algoritma regresi pohon berbasis gradien (*Gradient Boosting Tree*).
  - **`train_lstm_direct()`**: Membangun Jaringan Saraf Tiruan (*Neural Network*) Sequential menggunakan arsitektur Keras/TensorFlow. Terdiri dari layer LSTM (128 neuron, dropout 0.2) dan dilatih dengan propagasi balik selama 50 *Epochs*.
  - **`train_hybrid_direct()`**: Model gabungan (*Sequential Stacking*). Memanggil LSTM terlebih dahulu untuk memprediksi tren utama, mengekstrak residual (selisih error dari LSTM), lalu melatih XGBoost secara khusus untuk mengoreksi residual error tersebut.

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
