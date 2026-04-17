FROM python:3.11-slim

WORKDIR /app

# System deps: poppler-utils provides pdftotext (fallback PDF extractor)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Lightweight Python deps only (no torch/sentence-transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pipeline scripts
COPY build_ocw_dataset.py .
COPY data_generator.py .
COPY batch_pipeline.py .
COPY online_features.py .
COPY synthetic_expansion.py .
COPY mock_predict_server.py .
COPY scrape_ocw.py .

ENTRYPOINT ["python", "build_ocw_dataset.py"]
CMD ["--help"]
