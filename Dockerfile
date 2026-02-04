# Chandra OCR service (HF method — single container)
# Requires: poppler for PDF rasterization
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Optional: pre-download model at build time (uncomment and set MODEL_CHECKPOINT)
# ENV MODEL_CHECKPOINT=datalab-to/chandra
# RUN python -c "from chandra.model import InferenceManager; InferenceManager(method='hf')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
