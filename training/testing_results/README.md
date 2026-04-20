# Training Testing Branch Artifacts (Run B Only)

This branch is intentionally locked to **Run B (5 labels)** for teammate integration.

## Labels

- `Lecture Notes`
- `Other`
- `Problem Set`
- `Exam`
- `Reading`

Run B merge rule:
- `Other + Project + Solution -> Other`

## Layout

- `parquet/`: Run B train/eval parquet files only
- `model_bundles/`: committed model artifacts for integration
- `tfidf_lightgbm/`: Run B config + metrics summary
- `tfidf_logreg/`: Run B config + metrics summary
- `sbert/`: Run B config + metrics summary for `sbert_mlp`
- `sbert_logreg/`: Run B config + metrics summary
- `integration_run_b/`: handoff instructions for teammates
- `code/train.py`: training script snapshot

## Integration Baseline

Primary integration bundle:
- `model_bundles/tfidf_lightgbm/run_b/model_bundle_docker.joblib`

Expected metrics (Run B / TF-IDF + LightGBM):
- Accuracy `0.6351`
- Macro F1 `0.5574`
- Weighted F1 `0.6199`
- Min class F1 `0.3008`
- Quality Gate `PASS`
