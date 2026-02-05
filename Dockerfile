# Chandra OCR service (HF method — single container)
# Multi-stage: heavy pip layer stays in builder; final image gets COPY only (avoids WSL2 lchown/extract bugs).
# Requires: poppler for PDF rasterization
# First build: 10–20 min; exporting final image is faster (fewer/lighter layers).
# ----------------------------------------
# Stage 1: install dependencies (big layer stays here)
# ----------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --target=/app/deps -r requirements.txt && \
    find /app/deps -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/deps -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app/deps -type d -name tests -exec rm -rf {} + 2>/dev/null || true

# ----------------------------------------
# Stage 2: runtime image (no RUN pip; only COPY from builder)
# ----------------------------------------
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/deps /app/deps
COPY app ./app

ENV PYTHONPATH=/app/deps
ENV PATH="/app/deps/bin:${PATH}"

EXPOSE 8000

# Optional: pre-download model at build time (uncomment and set MODEL_CHECKPOINT)
# ENV MODEL_CHECKPOINT=datalab-to/chandra
# RUN python -c "from chandra.model import InferenceManager; InferenceManager(method='hf')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
