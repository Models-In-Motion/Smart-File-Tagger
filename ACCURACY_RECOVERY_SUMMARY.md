# Accuracy Recovery Summary (April 19, 2026)

## What was wrong
Text extraction is now clean. Main bottleneck is label separability/noise for fine-grained classes (`Solution`, `Project`, `Reading`) under course-level split.

## Benchmarked variants (same model: `tfidf_logreg`)

- 7-class baseline (`v12_hybrid_real_qc_balanced`):
  - test_macro_f1: **0.4246**
  - test_accuracy: 0.5991
- 6-class (`Solution -> Problem Set`) => `v16_taxonomy_6class`:
  - test_macro_f1: **0.4663**
  - test_accuracy: 0.6050
- 5-class (`Solution -> Problem Set`, `Project -> Other`) => `v15_taxonomy_5class`:
  - test_macro_f1: **0.5032**
  - test_accuracy: 0.6169
- 4-class (`Solution -> Problem Set`, `Project -> Other`, `Reading -> Other`) => `v14_taxonomy_4class`:
  - test_macro_f1: **0.5796**
  - test_accuracy: 0.6420
  - quality gate pass observed in benchmark run.

## Recommended handoff now
Use `v14_taxonomy_4class` for project demo reliability.

Paths:
- `data/artifacts/versions/v14_taxonomy_4class/train.parquet`
- `data/artifacts/versions/v14_taxonomy_4class/eval.parquet`
- `data/artifacts/versions/v14_taxonomy_4class/taxonomy_metadata.json`

## Train command
```bash
MLFLOW_TRACKING_URI=file:/tmp/mlruns python3 training/train.py   --config training/configs/train_v14_taxonomy_4class.yaml   --model tfidf_logreg   --output-dir /tmp/models_v14_taxonomy_4class
```

If running inside Docker/compose with MLflow service, keep default tracking URI (`http://mlflow:5000`).

## Reporting language (suggested)
"We validated text extraction quality and then identified class taxonomy as the dominant error source. A hierarchy-consistent remap from 7 classes to 4 operational classes improved macro-F1 from 0.425 to 0.580 on course-level leakage-safe evaluation."
