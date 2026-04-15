# Smart File Tagger

> An intelligent ML sidecar for Nextcloud that automatically tags uploaded documents using few-shot, per-user classification — and gets smarter with every correction.

---

## What it does

Most cloud storage becomes a digital junk drawer over time. Smart File Tagger solves this by running alongside your Nextcloud instance as a lightweight sidecar service. When you upload a file, it reads the document, predicts a category, and tags it automatically — or asks you to confirm if it's uncertain.

What makes it different from a standard classifier: **every user defines their own categories**. A lawyer's "Client Contracts" and a researcher's "Lab Reports" are learned from just a handful of examples you provide. The model then retrains weekly on your corrections, continuously adapting to your specific documents and writing style.

---

## Architecture overview

```
Nextcloud (file upload)
    │
    │  webhook (POST /predict)
    ▼
┌─────────────────────────────┐
│   Smart File Tagger Sidecar │  ← FastAPI service (this repo)
│                             │
│  1. Extract text from file  │
│  2. Embed with SBERT        │
│  3. Classify against your   │
│     personal prototypes     │
│  4. Return tag + confidence │
└─────────────────────────────┘
    │
    │  confidence > 0.85 → auto-tag applied
    │  confidence 0.5–0.85 → suggestion shown to user
    │  confidence < 0.5 → no tag, manual labeling
    │
    ▼
PostgreSQL (feedback log)
    │
    │  weekly retrain job
    ▼
Updated classifier (per-user model promoted if metrics improve)
```

The sidecar never replaces Nextcloud's core functionality. If the ML service is down, Nextcloud continues working normally — manual tagging always remains available.

---

## Repository structure

```
smart-file-tagger/
│
├── serving/                  # FastAPI sidecar service (Krish)
│   ├── app/
│   │   ├── main.py           # API entrypoint — /predict, /feedback, /register-category
│   │   ├── predictor.py      # SBERT embedding + classifier inference
│   │   ├── extractor.py      # Text extraction from PDF, image, plain text
│   │   ├── feedback.py       # Feedback endpoint + PostgreSQL write
│   │   └── category_mgr.py   # Per-user category + prototype management
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── training/                 # Model training + retraining pipeline (Vedant)
│   ├── train.py              # Initial training on QS-OCR-Large
│   ├── evaluate.py           # Precision/recall evaluation per category
│   ├── retrain_job.py        # Weekly cron — pulls feedback, retrains, promotes
│   ├── model_registry.py     # Version tracking for promoted models
│   ├── configs/
│   │   └── train_config.yaml
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── data/                     # Data pipeline + PostgreSQL management (Viral)
│   ├── download_dataset.py   # Downloads and caches QS-OCR-Large
│   ├── preprocess.py         # Cleans OCR text, removes artifacts
│   ├── split.py              # Chronological + document-level train/val/test split
│   ├── schema.sql            # PostgreSQL table definitions
│   ├── migrations/           # Schema versioning
│   ├── feedback_export.py    # Exports validated feedback for retraining
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── models/                   # Versioned model artifacts (git-ignored, Docker volume)
│   ├── sbert_classifier_v1.pkl
│   ├── prototype_store.json
│   └── model_registry.json
│
├── infra/                    # Chameleon Cloud + Nextcloud infrastructure
│   ├── nextcloud-compose.yml
│   ├── postgres-init.sql
│   ├── nginx.conf
│   └── chameleon-setup.sh
│
├── docker-compose.yml        # Spins up full stack locally
├── .env.example              # Required environment variables
├── .gitignore
└── README.md
```

---

## Tech stack

| Layer | Technology | Reason |
|---|---|---|
| Serving | FastAPI + Uvicorn | Async, lightweight, < 300ms target |
| Embedding | `sentence-transformers` (all-MiniLM-L6-v2) | ~80MB, CPU-friendly, strong on documents |
| Classifier | SBERT + lightweight classifier head | Few-shot capable, retrainable per user |
| LLM (explainability) | Mistral 7B via Ollama | Self-hosted, explains uncertain predictions |
| Database | PostgreSQL | Feedback logging, prototype storage, model registry |
| Infrastructure | Docker + Chameleon Cloud | Reproducible, self-hosted |
| Training data | QS-OCR-Large (pre-OCR'd text from RVL-CDIP) | 400K labeled text docs, no OCR overhead |

---

## API endpoints

### `POST /predict`
Accepts a file upload. Returns a predicted category, confidence score, and an optional LLM-generated explanation for uncertain predictions.

```json
{
  "tag": "Invoice",
  "confidence": 0.91,
  "suggestion": false,
  "explanation": null
}
```

### `POST /register-category`
Creates a new user-defined category from 3–5 example documents. Computes and stores a prototype embedding vector.

```json
{
  "user_id": "user_abc",
  "category_name": "Client Contracts",
  "examples": ["<base64 file>", "<base64 file>", "<base64 file>"]
}
```

### `POST /feedback`
Logs a user's accept or correction of a predicted tag. This data feeds the weekly retraining pipeline.

```json
{
  "file_id": "file_xyz",
  "predicted_tag": "Invoice",
  "correct_tag": "Receipt",
  "accepted": false,
  "user_id": "user_abc"
}
```

### `GET /health`
Returns `200 OK` if the service is running. Nextcloud checks this before attempting classification.

---

## Performance targets

| Metric | Target |
|---|---|
| End-to-end latency | < 300ms |
| Network overhead | ~50ms |
| SBERT embedding | ~100ms |
| Classifier inference | ~50ms |
| Post-processing + response | ~100ms |
| Throughput | 15 requests/second |
| RAM footprint | < 500MB |

---

## Human-in-the-loop retraining

The system is designed around a weekly retraining cycle:

1. User corrections and confirmations are logged to PostgreSQL with timestamps
2. Viral's `feedback_export.py` filters for high-quality, validated corrections
3. Vedant's `retrain_job.py` re-encodes documents using the frozen SBERT backbone and retrains the classifier head
4. Evaluation runs automatically — the new model is only promoted if precision/recall improves
5. Krish's serving layer hot-reloads the promoted model without downtime

Retraining only touches the **classifier head**, not the SBERT backbone — so even 10–20 user corrections per category are enough to meaningfully improve accuracy.

---

## Fail-open design

The sidecar is designed to never degrade Nextcloud's core experience:

- If the ML service is unreachable, Nextcloud continues normally — no tag is applied
- If the model returns an error, the `/predict` endpoint returns `{"tag": null}` rather than crashing
- Nextcloud's manual tagging UI always remains available regardless of sidecar status
- Health checks allow Nextcloud to skip the webhook call entirely when the sidecar is down

---

## Getting started

### Prerequisites
- Docker and Docker Compose
- A running Nextcloud instance (see `infra/nextcloud-compose.yml`)
- Python 3.10+
- A Chameleon Cloud VM (see `infra/chameleon-setup.sh`)

### Local development

```bash
# Clone the repo
git clone https://github.com/smart-file-tagger/smart-file-tagger.git
cd smart-file-tagger

# Copy and fill in environment variables
cp .env.example .env

# Start the full stack (Nextcloud + PostgreSQL + sidecar)
docker-compose up --build

# Verify the sidecar is running
curl http://localhost:8000/health
```

### Running tests

```bash
# Serving tests
cd serving && pytest tests/

# Training tests
cd training && pytest tests/

# Data pipeline tests
cd data && pytest tests/
```

---

## Team

| Role | Person | Owns |
|---|---|---|
| Serving Lead | Krish | `serving/` — FastAPI sidecar, latency, fail-open design |
| Training Lead | Vedant | `training/` — SBERT classifier, retrain pipeline, model registry |
| Data Lead | Viral | `data/` — preprocessing, PostgreSQL schema, feedback export |
| All members | Joint | `infra/`, `docker-compose.yml`, integration, demo |

---

## Dataset

Training uses **QS-OCR-Large** — 400,000 pre-OCR'd text documents across 15 categories, derived from RVL-CDIP. Text extraction has already been performed, so no image processing is required during training. The dataset provides a warm-start baseline; per-user retraining adapts the model away from the training distribution toward each user's actual documents.

Source: [QuickSign/ocrized-text-dataset](https://github.com/QuickSign/ocrized-text-dataset)

---

## Data pipeline operations (merged from data branch)

The data branch adds a concrete OCW-oriented pipeline path for generating training-ready artifacts and simulated feedback. The canonical ETL entrypoint is:

- `build_ocw_dataset.py` (primary/canonical ETL)
- `data_generator.py` (simulated production feedback JSONL)
- `batch_pipeline.py` (versioned train/eval split with leakage-safe partitioning)
- `online_features.py` (single-document online-path feature extraction)
- `download_ocw_archives.py` (polite OCW archive downloader/extractor)
- `etl_ocw.py` (legacy helper kept for backward compatibility)

Typical local run order:

```bash
python build_ocw_dataset.py --root . --output artifacts/ocw_dataset.parquet --summary-output artifacts/ocw_summary.json --text-backend auto --min-text-chars 200
python data_generator.py --input artifacts/ocw_dataset.parquet --output artifacts/production_feedback.jsonl --num-events 500 --seed 42
python batch_pipeline.py --dataset artifacts/ocw_dataset.parquet --feedback artifacts/production_feedback.jsonl --output-dir artifacts/versions --version v1 --eval-ratio 0.2 --seed 42
python online_features.py path/to/file.pdf --output artifacts/sample_input.json
```

This keeps the repository-level product framing while preserving the data team’s execution details for ETL, feedback simulation, and training-set preparation.

---

## License

MIT
