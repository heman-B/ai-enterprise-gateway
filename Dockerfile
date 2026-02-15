# LLM Gateway Container
# Python 3.11 slim für minimale Image-Größe
FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten (für httpx SSL-Support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Abhängigkeiten zuerst kopieren (Docker Layer-Cache optimal nutzen)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY gateway/ ./gateway/

# Datenpersistenz-Verzeichnis für SQLite
RUN mkdir -p /app/data

# Nicht als Root ausführen (Security-Best-Practice)
RUN useradd -m -u 1000 gateway && chown -R gateway:gateway /app
USER gateway

EXPOSE 8000

# Gesundheitsprüfung direkt im Container
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
