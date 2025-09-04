FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
