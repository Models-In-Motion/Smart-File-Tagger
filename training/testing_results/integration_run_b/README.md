# Run B Integration Guide (5 Labels)

This is the only supported setup for teammate integration on this branch.

## Labels

- `Lecture Notes`
- `Other`
- `Problem Set`
- `Exam`
- `Reading`

## Canonical Model Artifact

Use this bundle in serving/integration:

- `training/testing_results/model_bundles/tfidf_lightgbm/run_b/model_bundle_docker.joblib`

Fallback local-built bundle:

- `training/testing_results/model_bundles/tfidf_lightgbm/run_b/model_bundle.joblib`

## Canonical Data Files

- `training/testing_results/parquet/run_b_train_llm_merged_ops.parquet`
- `training/testing_results/parquet/run_b_eval_llm_merged_ops.parquet`

## Canonical Configs

The default configs in this branch are aligned to Run B:

- `training/configs/train.yaml`
- `training/configs/train_docker.yaml`

## Training Command

Local:

```bash
cd training
python train.py --config configs/train.yaml --model tfidf_lightgbm
```

Docker:

```bash
docker compose run --rm training \
  python train.py --config configs/train_docker.yaml --model tfidf_lightgbm
```

## Expected Run B Metrics (TF-IDF + LightGBM)

- Test Accuracy: `0.6351`
- Test Macro F1: `0.5574`
- Test Weighted F1: `0.6199`
- Min Class F1: `0.3008`
- Quality Gate: `PASS`
