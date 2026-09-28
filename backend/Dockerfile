# Base image: lightweight Python 3.10
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app/

EXPOSE 5000

CMD ["python", "api.py"]
