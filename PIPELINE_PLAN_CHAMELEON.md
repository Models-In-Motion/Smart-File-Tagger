# OCW Data Pipeline (Local -> Chameleon)

This plan is designed for your **Data** role in ECE-GY 9183 and follows the project requirements from:
- https://ffund.github.io/ml-sys-ops/docs/project.html

## Canonical ETL Script (team decision)

- **Canonical ETL script:** `build_ocw_dataset.py`
- **Why:** It outputs the schema consumed by `data_generator.py` and `batch_pipeline.py` (`doc_id`, `course_id`, `label_source`, feature columns, etc.).
- **Status of `etl_ocw.py`:** legacy helper for older 5-column parquet format; do not use it for the current end-to-end pipeline.

## 1) What to do right now (local, fast iteration)

Goal: prove pipeline logic works end-to-end before Chameleon deployment.

### A. Build dataset from already-downloaded course dumps

```bash
python build_ocw_dataset.py \
  --root . \
  --output artifacts/ocw_dataset.parquet \
  --summary-output artifacts/ocw_summary.json \
  --text-backend auto \
  --min-text-chars 200
```

This creates the canonical dataset file used by downstream scripts.

### B. Notes about local text quality

For training-quality text, run with `pypdf` available (`--text-backend auto` will select it).

---

## 2) Chameleon setup (where course grading expects execution)

Goal: run the exact same pipeline in Chameleon with real dependency stack and larger data scale.

### A. Environment bootstrap on Chameleon

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas pyarrow pypdf
```

### B. Copy project code to Chameleon

- Copy this folder (or clone your repo there).
- Keep `build_ocw_dataset.py`, `download_ocw_archives.py`, and artifacts/scripts together.

### C. Download OCW course archives politely (>=2s wait)

Prepare URL file (one course URL per line), e.g. `course_urls.txt`:

```text
https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/
https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/
```

Run downloader:

```bash
python download_ocw_archives.py \
  --urls-file course_urls.txt \
  --out-dir ocw_downloads \
  --extract-dir ocw_courses \
  --sleep-seconds 2.0 \
  --max-zips-per-course 1
```

### D. Build training dataset on Chameleon

```bash
python build_ocw_dataset.py \
  --root ocw_courses \
  --output artifacts/ocw_dataset.parquet \
  --summary-output artifacts/ocw_summary.json \
  --text-backend auto \
  --min-text-chars 200
```

On Chameleon, with `pypdf` installed, backend should be `pypdf` (not `strings`).

---

## 3) How this satisfies Data-role requirements

## Data flow (inference/training)
- Raw OCW archives -> extracted course folders.
- Resource metadata (`data.json`) + files (`static_resources`) -> candidate documents.
- Text extraction + metadata enrichment -> dataset parquet.
- `data_generator.py` -> emulated production feedback JSONL.
- `batch_pipeline.py` -> course-level split + versioned `train.parquet` / `eval.parquet`.

## Candidate selection
Implemented in `build_ocw_dataset.py` using:
- Allowed doc types (`Lecture Notes`, `Problem Set`, `Exam`, etc.)
- Minimum text length (`--min-text-chars`)
- Optional dedupe by normalized text hash

## Leakage prevention
Implemented by:
- Deduplication using normalized text hash
- Course-level train/eval split in `batch_pipeline.py` (same course not in both)

---

## 4) Recommended next milestone steps

1. Expand to 50+ course URLs and run on Chameleon.
2. Validate label distribution and text quality from `artifacts/ocw_summary.json`.
3. Freeze snapshot naming convention (e.g., `ocw_dataset_YYYYMMDD.parquet`).
4. Hand off `ocw_dataset.parquet` + versioned train/eval outputs to training teammate.
5. Add periodic refresh job on Chameleon (cron or workflow) to demonstrate ongoing operation.
