# Menggunakan sistem operasi Linux mini dengan Python 3.10
FROM python:3.10-slim

# Menghindari pesan error atau interupsi saat instalasi
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2 

# Membuat direktori kerja di dalam kontainer Docker
WORKDIR /app

# Menyalin daftar library (requirements) terlebih dahulu
COPY requirements.txt /app/

# Menginstal semua library (memakan waktu beberapa menit saat proses build)
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file backend v2 (api.py, train.py, utils, data, dll) ke dalam Docker
COPY . /app/

# Membuka port 5000 agar web Flask bisa diakses dari browser
EXPOSE 5000

# Menjalankan API sebagai proses utama saat container dinyalakan
CMD ["python", "api.py"]