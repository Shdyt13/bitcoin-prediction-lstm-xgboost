# Bitcoin Price Forecaster — Hybrid LSTM-XGBoost

Sistem Pendukung Keputusan (*Decision Support System*/DSS) berbasis web untuk **peramalan harga Bitcoin (BTC/IDR)**. Proyek ini membandingkan tiga skema pemodelan deret waktu — **LSTM**, **XGBoost**, dan **Hybrid LSTM-XGBoost (Residual-Correction Stacking)** — menggunakan strategi **Direct Multi-Step Forecasting** untuk horizon 1, 3, dan 7 hari ke depan.

Repositori ini merupakan implementasi teknis dari Tugas Akhir:

> **"Studi Komparatif Model LSTM, XGBoost, dan Hybrid Model Untuk Peramalan Harga Bitcoin"**
> Sapar Hidayat. S — NIM 2201020003
> Program Studi Teknik Informatika, Fakultas Teknik dan Teknologi Kemaritiman
> Universitas Maritim Raja Ali Haji (UMRAH), Tanjungpinang, 2026

---

## Fitur Utama

- **Perbandingan 3 model** — LSTM (*base-learner*), XGBoost (*single* & *meta-learner*), dan Hybrid Stacking dalam satu dashboard.
- **Direct Multi-Step Forecasting** — model dilatih secara independen untuk tiap horizon (1, 3, 7 hari) guna menghindari akumulasi kesalahan (*error propagation*) khas pendekatan rekursif.
- **Residual-Correction Hybrid** — XGBoost mempelajari residu kesalahan LSTM ( `ŷ_hybrid = ŷ_LSTM + ê_XGBoost` ) untuk mengoreksi *over-smoothing* pada tren.
- **Chronological Split (anti data-leakage)** — data latih dan data uji dipisah secara runtun waktu, bukan acak (data latih 2020–2025, data uji 2026).
- **Blind Forecasting** — proyeksi harga ke depan menggunakan 30 hari data observasi terakhir.
- **Dashboard interaktif** — visualisasi *actual vs prediction*, grafik metrik kesalahan, dan halaman *Model Comparison*.
- **Ekspor laporan ke Excel** (`.xlsx`) berisi RMSE, MAE, MAPE, dan detail angka prediksi.

---

## Ringkasan Metodologi

| Tahap | Deskripsi |
|---|---|
| **Sumber data** | Harga harian BTC/IDR dari CoinMarketCap, periode 1 Jan 2020 – 20 Mei 2026 |
| **Fitur masukan** | Harga *Open* (harga *High*, *Low*, dan *Volume* dieliminasi untuk mencegah multikolinearitas) |
| **Target** | Harga *Close* |
| **Normalisasi** | Min-Max Scaling ke rentang [0, 1], `scaler` di-*fit* hanya pada data latih |
| **Windowing** | *Sliding window* 30 hari |
| **Split data** | Latih: 1 Jan 2020 – 31 Des 2025 · Uji: 1 Jan 2026 – 20 Mei 2026 (*chronological split*) |
| **Horizon prediksi** | t+1, t+3, t+7 hari (Direct Multi-Step Forecasting) |
| **Model** | LSTM (2 layer × 50 unit, dropout 0.2, Adam, Early Stopping) · XGBoost (tuning via `RandomizedSearchCV`) · Hybrid (LSTM sebagai *base-learner*, XGBoost sebagai *meta-learner* atas residu) |
| **Evaluasi** | RMSE, MAE, MAPE |

### Ringkasan hasil (data uji 2026)

| Horizon | Model Terbaik (MAPE) | MAPE | RMSE (Rp) |
|---|---|---|---|
| 1 hari | LSTM | 3,64% | 62.927.542,68 |
| 3 hari | Hybrid LSTM-XGBoost | 4,23% | 73.769.422,33 |
| 7 hari | LSTM (MAPE) / **Hybrid (RMSE terendah)** | 6,77% / 6,90% | 116.982.086,30 / **114.341.538,60** |

LSTM unggul untuk peramalan jangka pendek karena mampu menangkap dependensi temporal, sedangkan Hybrid LSTM-XGBoost lebih stabil pada horizon menengah–panjang karena mekanisme koreksi residu menekan lonjakan kesalahan saat terjadi anomali harga.

Detail lengkap perhitungan, simulasi manual, dan pembahasan tersedia pada laporan Tugas Akhir (`LAPORAN_AKHIR_PUBLIKASI.pdf`).

---

## Struktur Direktori

```text
bitcoin-prediction-lstm-xgboost/
├── backend/                    # Flask REST API & Machine Learning Pipeline
│   ├── data/                   # Dataset CSV (mentah, bersih, training, testing)
│   ├── models/                 # Model hasil training (.keras, .json) & scaler.pkl
│   ├── uploads/                # File CSV yang diunggah melalui dashboard
│   ├── utils/
│   │   ├── preprocessing.py    # Data cleaning & normalisasi
│   │   └── windowing.py        # Pembentukan sliding window (Direct Multi-Step)
│   ├── api.py                  # Entry point Flask Server (endpoint /api/predict)
│   ├── train.py                # Script training LSTM, XGBoost, dan Hybrid
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                   # Dashboard React.js (Vite)
│   ├── src/
│   │   ├── App.jsx             # Komponen utama dashboard
│   │   ├── App.css / index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

## Tech Stack

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Backend & API | Python 3.10/3.11, Flask, Flask-CORS | Server komputasi & REST API |
| Deep Learning | TensorFlow / Keras | Membangun & melatih model LSTM |
| Machine Learning | XGBoost, Scikit-Learn | *Gradient boosting*, *hyperparameter tuning*, normalisasi |
| Data Processing | Pandas, NumPy | Manipulasi data & *sliding window* |
| Frontend & UI | React.js, Vite, Axios | *Single Page Application* & komunikasi API |
| Visualisasi | Chart.js (react-chartjs-2) | Grafik prediksi interaktif |
| Ekspor Data | SheetJS (XLSX), openpyxl | Ekspor laporan ke Excel |
| DevOps | Docker | Isolasi environment ML pada backend |

---

## Cara Menjalankan Aplikasi

### 1. Backend (Flask API via Docker)

Pastikan **Docker Desktop** sudah berjalan.

```bash
cd backend

# Build image (ulangi jika ada perubahan kode Python)
docker build -t btc-ai .

# Jalankan container
docker run -p 5000:5000 -v ${PWD}:/app btc-ai
```

Backend akan aktif di `http://localhost:5000`.

> Alternatif tanpa Docker: `pip install -r requirements.txt` lalu `python api.py` (disarankan menggunakan *virtual environment* Python 3.10+).

### 2. (Opsional) Melatih ulang model

```bash
cd backend
python train.py
```

Script ini akan menghasilkan model LSTM, XGBoost, dan Hybrid untuk horizon 1, 3, dan 7 hari, disimpan ke folder `backend/models/`.

### 3. Frontend (React Dashboard)

Gunakan terminal baru. Pastikan **Node.js** sudah terpasang.

```bash
cd frontend
npm install
npm run dev
```

Akses dashboard di `http://localhost:5173`.

### 4. Menggunakan Dashboard

1. Unggah **CSV data latih** dan **CSV data uji** pada panel *upload* (format kolom mengikuti data CoinMarketCap: `Tanggal`/`Date`, `Buka*`/`Open`, `Tutup**`/`Close`).
2. Pilih tab model (**XGBoost**, **LSTM**, atau **Hybrid**) lalu klik **Run Prediction** untuk melihat proyeksi 1, 3, dan 7 hari ke depan beserta grafiknya.
3. Buka tab **Model Comparison** untuk membandingkan RMSE, MAE, dan MAPE ketiga model, serta mengunduh laporan lewat **Export to Excel**.

---

## Metrik Evaluasi

- **RMSE** (*Root Mean Squared Error*) — memberi bobot lebih besar pada kesalahan bernilai besar (*outliers*).
- **MAE** (*Mean Absolute Error*) — rata-rata besaran kesalahan absolut.
- **MAPE** (*Mean Absolute Percentage Error*) — persentase kesalahan relatif terhadap nilai aktual.

---

## Publikasi

Hasil penelitian ini telah disubmit ke **Scientific Journal of Informatics** (Universitas Negeri Semarang) dengan judul *"Comparative Study of LSTM, XGBoost, and Hybrid Models for Bitcoin Price Forecasting"*. Bukti publikasi dapat dilihat pada lampiran laporan Tugas Akhir.

---

## Penulis

**Sapar Hidayat. S** — NIM 2201020003
Program Studi Teknik Informatika
Jurusan Teknik Elektro dan Informatika
Fakultas Teknik dan Teknologi Kemaritiman
Universitas Maritim Raja Ali Haji (UMRAH), Tanjungpinang

Dibimbing oleh: **Novrizal Fattah Fahmitra** & **Tekad Matulatan**
