# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps separately so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/     ./app/
COPY inference.py .

# Hugging Face Spaces runs as a non-root user; make sure we can write /tmp
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

# HF Spaces expects the server on port 7860
# --workers 1 keeps state consistent (one env per process)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
