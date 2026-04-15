# Dockerfile — Smart File Tagger Serving Layer
# Krish Jani (kj2743) — Serving Lead
# ECE-GY 9183 MLOps, Spring 2026
#
# What this file does:
#   Packages the FastAPI serving code into a self-contained Docker image.
#   Anyone can run this image without installing Python, dependencies,
#   or configuring anything except the .env file.
#
# Build:  docker build -t smart-tagger-serving .
# Run:    docker run -p 8000:8000 --env-file .env smart-tagger-serving

# ── Base image ──────────────────────────────────────────────────────────────
# python:3.10-slim is Python 3.10 on a minimal Debian Linux.
# We use slim (not the full image) to keep the image size small.
# Why 3.10? sentence-transformers and lightgbm both work well on 3.10.
FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────────────────
# These are Linux packages that Python libraries need under the hood.
# tesseract-ocr     : the OCR engine (pytesseract is just a Python wrapper for this)
# poppler-utils     : converts PDF pages to images (needed by pdf2image for OCR fallback)
# libgomp1          : LightGBM needs this for parallel processing
# curl              : useful for health check debugging inside the container
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* removes the package index after install
# to keep the image smaller

# ── Working directory ────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# We copy requirements.txt FIRST (before copying our code) because Docker
# caches each step. If only our code changes (not requirements), Docker
# reuses the cached pip install layer — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SBERT model so it's baked into the image.
# Without this, the first startup downloads 80MB which takes ~30 seconds
# and fails if there's no internet on the Chameleon VM.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── ONNX encoder (predictor USE_ONNX=True expects /models/sbert_onnx.onnx) ──
# Export runs at build time so the image is self-contained (no manual export_onnx.py).
COPY export_onnx.py .
RUN mkdir -p /models \
    && ONNX_EXPORT_DIR=/models SKIP_ONNX_BENCHMARK=1 python export_onnx.py \
    && rm -f export_onnx.py

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/ .

# ── LightGBM + label encoder (generate locally with create_fake_model.py if missing)
COPY models/ /models/

# ── Port ──────────────────────────────────────────────────────────────────────
# Document which port the app listens on.
# This doesn't actually open the port — docker run -p 8000:8000 does that.
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker periodically calls this to confirm the container is healthy.
# If /health returns non-200 three times in a row, Docker marks the
# container as unhealthy (visible in `docker ps` output).
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# Starts the FastAPI app with Uvicorn.
# --host 0.0.0.0 : listen on all network interfaces (not just localhost)
#                  Required so requests from outside the container reach us.
# --port 8000    : the port to listen on
# --workers 1    : number of worker processes (we test different values for
#                  the serving options table — 1 is the baseline)
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5"]