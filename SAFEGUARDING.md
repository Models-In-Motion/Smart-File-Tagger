# Safeguarding Plan — Smart File Tagger

This document describes the concrete mechanisms implemented across the system to
uphold fairness, explainability, transparency, privacy, accountability, and
robustness. Each principle is backed by code that runs automatically — not just
a policy statement.

---

## Fairness

**Problem:** A classifier trained on imbalanced data will perform poorly on
minority classes. Users whose documents fall into underrepresented categories
get worse predictions.

**Mechanisms:**

- `validate_ingestion()` (`data/build_ocw_dataset.py`) warns when any label has
  fewer than 10 examples in the ingested dataset. The ETL run is flagged before
  training data is ever produced.
- `validate_training_set()` (`data/batch_pipeline.py`) hard-blocks writing a
  training split if any label has fewer than 5 training examples. A model cannot
  be trained on a split that would guarantee poor performance on a class.
- `validate_training_set()` warns when a label's proportion in eval is more than
  3× its proportion in train — catching evaluation sets that do not reflect the
  training distribution.
- The 7-label schema (`Lecture Notes`, `Problem Set`, `Exam`, `Reading`,
  `Solution`, `Project`, `Other`) was chosen after auditing the OCW corpus.
  Labels with zero or near-zero representation (`Recitation`, `Lab`, `Syllabus`)
  were dropped rather than kept as phantom classes.

---

## Explainability

**Problem:** Users and operators need to understand why a prediction was made,
and why a model was retrained or rolled back.

**Mechanisms:**

- Every document in the training set carries a `label_source` column
  (`folder_structure`, `filename_pattern`, or `no_rule_matched`) so the training
  team can see exactly which labeling rule produced each training example.
- `drift_report.json` (written by `drift_monitor.py` after every run) records
  the JS divergence value, per-label distribution comparison, correction rate,
  and which specific thresholds were exceeded — giving a human-readable
  explanation of any rollback trigger.
- `split_metadata.json` (written by `batch_pipeline.py` for every version)
  records the exact course IDs in each split, label counts, and the timestamp —
  so any training run can be traced back to the exact data that produced it.

---

## Transparency

**Problem:** Black-box pipelines make it impossible to audit decisions or
reproduce results.

**Mechanisms:**

- All training data versions are written to `data/artifacts/versions/<version>/`
  with a `split_metadata.json` that is human-readable and committed alongside
  the code.
- MLflow tracks every training run with parameters, metrics, and the artifact
  version used (read from `split_metadata.json` by `train.py`).
- The `drift_metrics` PostgreSQL table provides a permanent audit trail of every
  drift monitor run — timestamp, all metric values, and pass/fail status.
- Confidence scores are exposed on every `/predict` response so the serving
  layer can show users when the model is uncertain (confidence thresholds:
  >0.85 auto-tag, 0.5–0.85 suggestion, <0.5 no tag).

---

## Privacy

**Problem:** Training on user-uploaded documents risks memorising or leaking
personal content.

**Mechanisms:**

- The training dataset is built entirely from publicly licensed MIT OCW course
  materials. No user-uploaded files are ever used for training.
- The `data_generator.py` simulation script sends only a 512-character text
  snippet to `/predict` — not the full document.
- The feedback table stores only `doc_id`, `predicted_label`, `user_action`, and
  `user_label` — no document content is persisted in the feedback log.
- PostgreSQL is deployed with credentials scoped to the `tagger` database only
  (`postgresql://tagger:tagger@postgres:5432/tagger`). No cross-database access.

---

## Accountability

**Problem:** When something goes wrong (bad predictions, model degradation), it
must be possible to identify what changed, when, and who/what triggered it.

**Mechanisms:**

- `split_metadata.json` is written for every dataset version with a UTC
  timestamp and full provenance (course IDs, label distributions, row counts).
- `drift_metrics` table records every hourly monitor run with all metric values
  and a `details` JSONB column containing the full report — permanently
  queryable.
- MLflow model registry (`model_registry.py`) records every promoted model
  version with the training run ID, metrics, and promotion timestamp.
- `retrain_log` table (Vedant's `retrain_trigger.py`) records every retraining
  event with the number of corrections that triggered it.
- Rollback events are triggered automatically by `monitor.py` reading the
  `drift_metrics` table — the trigger is logged, not a silent manual action.

---

## Robustness

**Problem:** The system must degrade gracefully under data quality issues,
infrastructure failures, and distribution shift.

**Mechanisms:**

- `validate_ingestion()` has 8 hard checks that block bad data from entering the
  pipeline. A single corrupt ETL run cannot silently poison the training set.
- `validate_training_set()` has 7 hard checks that block the batch pipeline from
  writing a split that would produce a broken model.
- Course-level train/eval splitting (`course_level_split()`) prevents data
  leakage — no course appears in both splits.
- `drift_monitor.py` runs hourly and exits with code 1 if hard drift thresholds
  are exceeded, triggering automatic rollback to the stub model via
  `monitor.py`. The system does not wait for a human to notice degradation.
- All three scripts (`build_ocw_dataset.py`, `batch_pipeline.py`,
  `drift_monitor.py`) fall back gracefully when optional dependencies are
  unavailable (e.g. PostgreSQL unreachable → falls back to JSONL; scipy missing
  → clear error message).
- The serving layer continues operating if the ML service is down — Nextcloud
  manual tagging always remains available as a fallback.

---

## Summary table

| Principle | Key mechanism | Where in code |
|---|---|---|
| Fairness | Block training if any label < 5 examples | `batch_pipeline.py:validate_training_set()` |
| Fairness | Warn on label imbalance at ingestion | `build_ocw_dataset.py:validate_ingestion()` |
| Explainability | `label_source` on every training row | `build_ocw_dataset.py:modern_course_records()` |
| Explainability | `drift_report.json` explains every rollback | `drift_monitor.py` |
| Transparency | `split_metadata.json` per dataset version | `batch_pipeline.py:write_outputs()` |
| Transparency | MLflow tracks every training run | `training/train.py` |
| Privacy | Training data is public OCW only, no user content | `data/build_ocw_dataset.py` |
| Privacy | Feedback stores no document content | `serving/app/feedback.py` |
| Accountability | `drift_metrics` table — permanent audit trail | `drift_monitor.py` |
| Accountability | `retrain_log` table — retraining history | `training/retrain_trigger.py` |
| Robustness | Hard validation gates block bad data/splits | `validate_ingestion()`, `validate_training_set()` |
| Robustness | Automatic rollback on drift threshold breach | `drift_monitor.py` → `monitor.py` |
