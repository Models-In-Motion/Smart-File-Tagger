# Safeguarding Plan — Smart File Tagger

This document describes safeguards that are implemented in code for the integrated
system (Nextcloud + serving + data + training + monitoring).

It is intentionally implementation-specific: each claim maps to files currently in
`version-1/main`.

---

## Current operating scope

- Base ingestion taxonomy supports 7 canonical OCW labels in data pipelines:
  `Lecture Notes`, `Problem Set`, `Exam`, `Reading`, `Solution`, `Project`, `Other`.
- Integrated production training/retraining currently uses **Run B merged 5-label taxonomy**
  via `llm_label_merged`:
  `Lecture Notes`, `Other`, `Problem Set`, `Exam`, `Reading`.
- This 5-label scope is enforced in training configs and quality gates for
  deployment decisions.

---

## Fairness

### Risk
Minority labels and noisy correction data can cause disproportionate error rates.

### Implemented safeguards

- **Ingestion validation before dataset write**
  `data/build_ocw_dataset.py::validate_ingestion()` blocks invalid or low-quality
  rows (null required fields, invalid labels/sources, duplicate `doc_id`, empty
  text, etc.) and warns for sparse labels.

- **Split validation before train/eval artifacts are written**
  `data/batch_pipeline.py::validate_training_set()` blocks invalid train/eval
  splits (required columns, duplicates, tiny class counts, too-small splits,
  leakage) and warns on severe label skew.

- **Retrain-time required-label gate**
  `training/retrain_trigger.py` runs `data/validate_dataset.py` before retraining,
  with Run B requirements:
  - required labels: `Lecture Notes`, `Other`, `Problem Set`, `Exam`, `Reading`
  - minimum examples per label check

- **Human correction upweighting**
  `training/train.py` assigns sample weights (10x) to rows with
  `source == "user_feedback"`, so verified human corrections have higher influence
  than base rows during fit.

---

## Explainability

### Risk
Users and operators cannot trust outcomes if decisions are opaque.

### Implemented safeguards

- **Prediction confidence + ranked alternatives returned to clients**
  Serving responses include predicted tag, confidence, and top predictions
  (`serving/app/main.py`, `serving/app/predictor.py`).

- **Data lineage and split provenance persisted**
  `data/batch_pipeline.py` writes `split_metadata.json` for each dataset version
  (row counts, label distributions, course split info, timestamp).

- **Drift reasoning persisted**
  `data/drift_monitor.py` writes detailed drift diagnostics both to
  `drift_report.json` and `drift_metrics.details` (JSONB), including threshold
  breaches and per-check outcomes.

- **Retrain decision trace**
  `training/retrain_trigger.py` writes retrain events into `retrain_log`.

---

## Transparency

### Risk
Without auditable telemetry, model changes and regressions cannot be defended.

### Implemented safeguards

- **MLflow run tracking for training/retraining**
  `training/train.py` logs parameters, metrics, runtime costs, gate metrics, and
  artifacts (`metrics_summary.json`, model artifacts, bundle).

- **Model registry integration with stage transitions**
  `training/train.py` registers passing models in MLflow and attempts promotion to
  Staging through `training/model_registry.py`.

- **Persistent monitoring signal storage**
  Drift, feedback, predictions, and retrain events are persisted in PostgreSQL
  tables (`drift_metrics`, `feedback`, `predictions`, `retrain_log`).

---

## Privacy

### Risk
Production feedback loop can ingest user-derived text; this must be limited and
controlled.

### Implemented safeguards

- **No raw file bytes stored in training artifacts**
  Training uses extracted text features, not uploaded binary files.

- **Bounded text retention in predictions table**
  `serving/app/feedback.py::log_prediction()` truncates stored `extracted_text`
  to max 10,000 chars before DB write.

- **Scoped feedback schema**
  Feedback/prediction logs store operational identifiers and labels
  (`file_id`, `user_id`, tags, confidence, model version, timestamps), not full
  user account profiles.

- **Synthetic traffic generator sends only snippets**
  `data/data_generator.py` limits text sent to `/predict` (`TEXT_CHARS = 512`) in
  emulated traffic mode.

- **DB credentials isolated to project DB**
  Compose services use the dedicated `tagger` PostgreSQL database.

> Note: In the integrated retrain loop, corrected user feedback is intentionally
> incorporated into retraining (`batch_pipeline.py` append mode). This is required
> for closing the feedback loop, and is controlled by the gates listed above.

---

## Accountability

### Risk
If model quality degrades, we need deterministic attribution and rollback.

### Implemented safeguards

- **Hourly feedback-threshold trigger**
  `docker-compose.yml` runs `retrain-cron` hourly; `training/retrain_trigger.py`
  retrains only when corrected feedback count since last trigger exceeds
  threshold (`FEEDBACK_THRESHOLD`, default 50).

- **Append-only feedback enrichment with explicit source tagging**
  `data/batch_pipeline.py` appends correction-derived rows with
  `source = "user_feedback"`, preserving provenance.

- **Rollback state persisted centrally**
  `serving/app/feedback.py` stores rollback flags in `model_status` table so
  all serving workers share consistent rollback state.

- **Automated monitor decisions logged and actioned**
  `serving/app/monitor.py` evaluates rollback/promotion checks and calls serving
  admin endpoints for rollback/restore/load-model decisions.

---

## Robustness

### Risk
Bad data, unstable retrains, or drift can silently degrade production.

### Implemented safeguards

- **Data quality hard gates before retrain**
  Retrain flow executes:
  1. append feedback rows (`batch_pipeline.py --base-train ... --output ...`)
  2. validate enriched data (`validate_dataset.py`)
  3. run training (`train.py`)

- **Safe first-run fallback for feedback parquet**
  `training/train.py::resolve_train_data_path()` falls back from
  `run_b_train_with_feedback.parquet` to
  `run_b_train_llm_merged_ops.parquet` if enriched data is absent.

- **Quality gates before deployment artifact sync/registration**
  `training/train.py` enforces:
  - `core_macro_f1 >= 0.50`
  - all core-label F1 `>= 0.30`
  for core labels: `Lecture Notes`, `Other`, `Problem Set`, `Exam`, `Reading`.

  Only when gates pass:
  - bundle is copied to serving model path (`sync_bundle_to_serving`)
  - model is registered/logged for promotion.

- **Leakage prevention by course-level split design**
  `data/batch_pipeline.py::course_level_split()` keeps train/eval course sets
  disjoint.

- **Online rollback triggers**
  `serving/app/monitor.py` triggers rollback on high error/correction signals and
  promotes staged models only after canary checks.

- **Fail-open prediction path for non-critical writes**
  Serving continues inference even if prediction/feedback DB logging fails
  (`serving/app/feedback.py` logs warnings without crashing prediction flow).

---

## Safeguard map (code anchors)

| Principle | Mechanism | Code anchor |
|---|---|---|
| Fairness | Ingestion hard checks and sparse-label warnings | `data/build_ocw_dataset.py::validate_ingestion()` |
| Fairness | Train/eval split quality + leakage checks | `data/batch_pipeline.py::validate_training_set()` |
| Fairness | Human-correction sample weighting | `training/train.py::compute_sample_weights()` |
| Explainability | Confidence + top predictions in API output | `serving/app/main.py`, `serving/app/predictor.py` |
| Explainability | Versioned split metadata and drift reports | `data/batch_pipeline.py`, `data/drift_monitor.py` |
| Transparency | MLflow params/metrics/artifacts tracking | `training/train.py` |
| Privacy | Truncated extracted text in predictions logs | `serving/app/feedback.py::log_prediction()` |
| Accountability | Hourly thresholded retrain trigger + retrain_log | `docker-compose.yml`, `training/retrain_trigger.py` |
| Robustness | Validation gate before retrain + quality gates before bundle sync | `training/retrain_trigger.py`, `training/train.py` |
| Robustness | Monitor-driven rollback/promotion | `serving/app/monitor.py` |

