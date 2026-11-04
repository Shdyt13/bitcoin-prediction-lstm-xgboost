# Base image dikunci ke patch spesifik agar build ulang tidak berubah
FROM python:3.10.22-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_DETERMINISTIC_OPS=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=1

WORKDIR /app

# Pasang dependensi dari lock file (semua versi terkunci)
COPY requirements.lock.txt /app/
RUN pip install --no-cache-dir -r requirements.lock.txt

# Copy application source
COPY . /app/

EXPOSE 5000

CMD ["python", "api.py"]
