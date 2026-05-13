# 📈 Bitcoin Price Forecaster: Hybrid LSTM-XGBoost

Repositori ini berisi *source code* lengkap untuk Sistem Pendukung Keputusan (DSS) peramalan harga Bitcoin berbasis Web. Proyek ini membandingkan kinerja model tunggal (**LSTM** & **XGBoost**) dengan arsitektur **Sequential Stacking Hybrid LSTM-XGBoost** menggunakan pendekatan **Direct Multi-Step Forecasting**.

Sistem ini dikembangkan sebagai bagian dari **Tugas Akhir / Skripsi**.

---

# ✨ Fitur Utama

## 🔬 Analisis 3 Model
Membandingkan secara langsung algoritma **LSTM**, **XGBoost**, dan **Hybrid (LSTM → XGBoost)** dalam satu dashboard.

## 🛡️ Out-of-Sample Testing
Menerapkan evaluasi pemisahan data kronologis secara ketat:

- **Train:** 2020–2025  
- **Test:** 2026+

untuk mencegah *data leakage*.

## 📊 Dashboard Interaktif
Antarmuka modern menggunakan **React.js** dan **Chart.js** dengan fitur:

- Blind Forecasting
- Prediksi 1 Hari
- Prediksi 3 Hari
- Prediksi 7 Hari

## 🗂️ Export to Excel
Fitur untuk mengunduh laporan:

- RMSE
- MAE
- MAPE
- Detail angka prediksi

ke format `.xlsx`.

---

# 🧠 Arsitektur Hybrid Stacking

Sistem ini menggunakan pendekatan **Sequential Stacking**, bukan sekadar rata-rata:

### 🔹 LSTM Layer
Bertugas menangkap pola urutan waktu (*temporal dependencies*) dan tren jangka panjang dari harga Bitcoin.

### 🔹 Feature Stacking
Hasil prediksi dari LSTM diekstrak dan dijadikan sebagai fitur input tambahan.

### 🔹 XGBoost Meta-Learner
Melakukan regresi akhir menggunakan gabungan:

- Data asli
- Fitur hasil prediksi LSTM

untuk:

- Mengoreksi residual error
- Menangkap anomali harga
- Mengatasi volatilitas pasar

---

# ⚙️ Struktur Direktori

```text
btc_predictorv2/
├── backend/                  # Flask REST API & Machine Learning Pipeline
│   ├── data/                 # Dataset CSV master
│   ├── models/               # Auto-generated model storage (.keras & .json)
│   ├── utils/                # Script Preprocessing & Windowing
│   ├── api.py                # Entry point Flask Server
│   ├── Dockerfile            # Konfigurasi Docker Image
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React.js Dashboard (Vite)
│   ├── src/                  # Source code React (App.jsx)
│   ├── package.json          # Node.js dependencies
│   └── vite.config.js        # Konfigurasi Vite
│
├── docs/                     # Dokumentasi gambar & flowchart
└── README.md                 # Dokumentasi proyek
```

---

# 🚀 Cara Menjalankan Aplikasi

Aplikasi ini menggunakan Docker untuk backend agar environment Machine Learning tetap stabil dan konsisten.

---

## 1️⃣ Jalankan Backend (Flask API via Docker)

Pastikan **Docker Desktop** sudah berjalan di komputer Anda.

```bash
# Masuk ke folder backend
cd backend

# Build image Docker (Lakukan ini jika ada perubahan kode python)
docker build -t btc-ai .

# Jalankan container backend
docker run -p 5000:5000 -v ${PWD}:/app btc-ai
```

Server backend akan aktif di:

```text
http://localhost:5000
```

---

## 2️⃣ Jalankan Frontend (React Dashboard)

Gunakan terminal baru untuk menjalankan frontend.

Pastikan **Node.js** sudah terinstal.

```bash
# Masuk ke folder frontend
cd frontend

# Install dependencies (Hanya saat pertama kali)
npm install

# Jalankan aplikasi mode development
npm run dev
```

Akses dashboard melalui browser di:

```text
http://localhost:5173
```

---

# 📊 Metodologi & Evaluasi

Metode penelitian meliputi:

- Preprocessing
- Normalisasi Min-Max Scaler
- Time-Series Windowing
- Direct Multi-Step Forecasting
- Hybrid Sequential Stacking

---

## 🔹 Preprocessing
Melakukan:

- Pembersihan data (*data cleaning*)
- Normalisasi menggunakan **Min-Max Scaler**

## 🔹 Windowing
Menggunakan teknik:

- Sliding Window
- Time-series sequence sepanjang **30 hari**

## 🔹 Forecasting
Pendekatan:

- **Direct Multi-Step Forecasting**
- Model independen untuk setiap horizon prediksi

---

# 📈 Metrik Evaluasi

Metrik evaluasi dihitung secara otomatis:

- RMSE (*Root Mean Squared Error*)
- MAE (*Mean Absolute Error*)
- MAPE (*Mean Absolute Percentage Error*)

---

# 📸 Tampilan Antarmuka

> Ganti gambar berikut dengan screenshot dashboard asli Anda yang berada di folder `docs/`.

Contoh:

```markdown
![Dashboard](docs/dashboard.png)
```

---

# 🛠️ Tech Stack

## Backend
- Python
- Flask
- TensorFlow / Keras
- XGBoost
- Scikit-Learn
- Pandas

## Frontend
- React.js
- Vite
- Chart.js
- Lucide-React
- Axios
- SheetJS (XLSX)

## DevOps
- Docker

---

# 👨‍💻 Penulis

**Sapar Hidayat. S**  

Universitas Maritim Raja Ali Haji (UMRAH)  
Fakultas Teknik dan Teknologi Kemaritiman  
Jurusan Teknik Informatika
2026