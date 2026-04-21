# MLOps Project: OCW Document Tagging Pipeline (Data Role)

This repository contains the **Data pipeline track** for the ECE-GY 9183 ML Systems project.
The project emulates an ML-powered document tagging feature for a Nextcloud-style document system using MIT OCW materials.

## 1) What This Repo Is For

The repo implements:

1. Data acquisition helper (polite OCW downloader).
2. Canonical ETL pipeline to build a training dataset.
3. Online (single-document) feature computation path.
4. Production feedback simulator (synthetic user actions).
5. Batch pipeline to prepare versioned train/eval datasets with leakage-safe splitting.
6. Dockerized execution for local parity and Chameleon migration.

## 2) Canonical ETL Decision

The **canonical ETL script is `build_ocw_dataset.py`**.

Reason:

1. It produces the schema used by downstream scripts (`data_generator.py`, `batch_pipeline.py`).
2. It includes candidate selection hooks, dedupe, metadata enrichment, and validation.

`etl_ocw.py` is kept as a legacy helper for an older 5-column schema and is not the primary path for the current end-to-end pipeline.

## 3) Current Status Snapshot (as of April 3, 2026)

From `artifacts/ocw_summary.json` and `artifacts/versions/v1/split_metadata.json`:

1. Canonical dataset rows: **549**
2. Courses: **5**
3. Text backend in successful Docker ETL: **pypdf**
4. Train split rows: **362**
5. Eval split rows: **40**
6. Leakage check: **No course overlap between train and eval**

Label counts in canonical dataset:

1. Lecture Notes: 186
2. Other: 147
3. Problem Set: 96
4. Exam: 60
5. Reading: 60

## 4) Top-Level File Guide

### Core pipeline scripts

1. `build_ocw_dataset.py`
   Canonical ETL. Reads OCW course dumps, extracts text, maps labels, validates schema, writes `artifacts/ocw_dataset.parquet` and summary JSON.
2. `data_generator.py`
   Simulates production user behavior (accept/correct/ignore) and writes feedback events to JSONL.
3. `batch_pipeline.py`
   Applies candidate selection + feedback correction, performs course-level split, writes versioned `train.parquet` and `eval.parquet`.
4. `online_features.py`
   Single-document inference-path feature extractor (real-time path emulation).
5. `download_ocw_archives.py`
   Polite OCW archive downloader/extractor (`sleep >= 2s`).
6. `etl_ocw.py`
   Legacy ETL (older schema: `file_id, course, label, text, source_path`).

### Containerization and dependencies

1. `Dockerfile`
   Python image with dependencies + pipeline scripts.
2. `docker-compose.yml`
   Services for `ocw-pipeline`, `data-generator`, and `batch-pipeline`.
3. `requirements.txt`
   Python dependencies (`pandas`, `pyarrow`, `pypdf`).

### Planning and project docs

1. `course_urls_example.txt`
   Example list format for downloader input.

### Data and outputs

1. `artifacts/ocw_dataset.parquet`
   Canonical ETL output used for simulator/training prep.
2. `artifacts/ocw_summary.json`
   ETL summary stats.
3. `artifacts/production_feedback.jsonl`
   Simulated production feedback events.
4. `artifacts/versions/v1/train.parquet`
   Batch pipeline train split.
5. `artifacts/versions/v1/eval.parquet`
   Batch pipeline eval split.
6. `artifacts/versions/v1/split_metadata.json`
   Split metadata and distributions.
7. `artifacts/sample_input.json`
   Example online-features output payload.
8. `artifacts/sample_output.json`
   Minimal sample prediction output.
9. `artifacts/demo_logs/20260403_140310/*`
   Logged local and Docker rerun evidence.

### Course dump directories (large local data)

1. `18.01-fall-2006/`
2. `5.111sc-fall-2014/`
3. `6.006-spring-2020/`
4. `6.034-fall-2010/`
5. `6.042j-spring-2015/`

These include extracted OCW site content and resources. They are ignored by `.gitignore`.

### Notebook

1. `eda.ipynb`
   Exploratory analysis notebook.

## 5) Dataset Details

### Canonical schema (`artifacts/ocw_dataset.parquet`)

Columns:

1. `doc_id`
2. `extracted_text`
3. `label`
4. `label_source`
5. `course_id`
6. `source_url`
7. `source`
8. `ingestion_timestamp`
9. `dataset_version`
10. `department`
11. `course_title`
12. `semester`
13. `filename`
14. `char_count`
15. `word_count`
16. `file_size_bytes`
17. `text_extraction_method`
18. `instructor`

### Legacy datasets present

1. `dataset.parquet` (single-course older format)
2. `dataset_all.parquet` (multi-course older format)

Legacy schema:

1. `file_id`
2. `course`
3. `label`
4. `text`
5. `source_path`

Use legacy files only for backward reference, not for current pipeline integration.

## 6) End-to-End Pipeline Flow

1. **Acquire source data**
   OCW archives are downloaded/extracted (or copied) into local course folders.
2. **Batch ETL**
   `build_ocw_dataset.py` scans course folders, extracts text, maps labels, dedupes, validates, writes canonical dataset.
3. **Online path**
   `online_features.py` processes one PDF at a time into serving features.
4. **Emulated production**
   `data_generator.py` simulates model predictions + user feedback events.
5. **Training prep**
   `batch_pipeline.py` applies candidate selection and leakage-safe train/eval split into versioned outputs.

## 7) How to Run Locally

### 7.1 Canonical ETL

```bash
python build_ocw_dataset.py \
  --root . \
  --output artifacts/ocw_dataset.parquet \
  --summary-output artifacts/ocw_summary.json \
  --text-backend auto \
  --min-text-chars 200
```

### 7.2 Generate simulated feedback

```bash
python data_generator.py \
  --input artifacts/ocw_dataset.parquet \
  --output artifacts/production_feedback.jsonl \
  --num-events 500 \
  --seed 42
```

### 7.3 Build versioned train/eval outputs

```bash
python batch_pipeline.py \
  --dataset artifacts/ocw_dataset.parquet \
  --feedback artifacts/production_feedback.jsonl \
  --output-dir artifacts/versions \
  --version v1 \
  --eval-ratio 0.2 \
  --seed 42
```

### 7.4 Run online feature extraction for one PDF

```bash
python online_features.py path/to/file.pdf --output artifacts/sample_input.json
```

## 8) Docker Usage

### 8.1 Validate compose config

```bash
docker compose config
```

### 8.2 Run ETL in container

```bash
docker compose run --rm ocw-pipeline
```

### 8.3 Run feedback simulator in container

```bash
docker compose run --rm data-generator
```

### 8.4 Run batch pipeline in container

```bash
docker compose run --rm batch-pipeline
```

## 9) Chameleon Migration Plan

1. Push repository to remote.
2. Launch Chameleon VM and clone repo.
3. Install Docker + compose (or Python venv path).
4. Bring/download OCW course dumps to VM.
5. Run ETL -> data generator -> batch pipeline.
6. Upload artifacts to object storage.
7. Record demo videos showing each pipeline stage.

Primary grading expectation: system must run on Chameleon, not only locally.

## 10) Work Completed vs Remaining

### Completed

1. Task 1: Dockerized ETL/data/batch services.
2. Task 2: Synthetic production feedback generator.
3. Task 3: Online feature computation script.
4. Task 4: Batch pipeline with versioned outputs and leakage-safe split.
5. Git initialized and `.gitignore` configured for large local data/artifacts.
6. Canonical ETL decision documented.
7. Local + Docker rerun evidence logs captured.

### Remaining

1. Task 5: Full Chameleon deployment and execution.
2. Push repo to remote and pull on Chameleon.
3. Configure VM runtime environment and storage.
4. Execute all pipelines on Chameleon and persist outputs.
5. Record and package demo videos.
6. Final documentation submission formatting (course-required report/slides by humans).

## 11) Known Issues / Notes

1. `online_features.py` may show zero extracted text on some local environments if PDF backend tools are unavailable or fail.
2. Docker ETL run with `--text-backend auto` successfully used `pypdf` in the evidence run.
3. `.gitignore` intentionally ignores `artifacts/` and large course dump folders to avoid giant commits.

## 12) Evidence Logs

Reference evidence bundle:

`artifacts/demo_logs/20260403_140310/`

Includes:

1. ETL run logs.
2. Data generator logs.
3. Batch pipeline logs.
4. Docker compose config.
5. Containerized rerun logs.
6. Notes on failed intermediate attempts and final successful runs.

## 13) Team Handoff Notes

For teammates:

1. Use `build_ocw_dataset.py` as the only ETL entrypoint for current pipeline work.
2. Treat `artifacts/ocw_dataset.parquet` as the canonical data contract for data->training integration.
3. Use `artifacts/versions/v1/*` as training-ready split outputs.
