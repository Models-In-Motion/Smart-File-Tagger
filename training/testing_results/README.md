# Training Testing Branch Artifacts

This folder contains experiment artifacts for three label strategies (Run A/B/C) evaluated with multiple models.

- `tfidf_lightgbm`
- `sbert_mlp` (under `sbert/`)
- `tfidf_logreg`
- `sbert_logreg`
- `answerdotai/ModernBERT-base` pilot (under `transformer_modernbert/`)

No production dataset was overwritten.

## Integration Baseline

Run B (5-label setup) is the integration baseline for teammates.

- Dataset:
  - `parquet/run_b_train_llm_merged_ops.parquet`
  - `parquet/run_b_eval_llm_merged_ops.parquet`
- Labels: `Lecture Notes, Other, Problem Set, Exam, Reading`
- Best integration model bundle:
  - `model_bundles/tfidf_lightgbm/run_b/model_bundle_docker.joblib`
- Expected test metrics (Run B / TF-IDF + LightGBM):
  - Accuracy `0.6351`
  - Macro F1 `0.5574`
  - Weighted F1 `0.6199`
  - Min class F1 `0.3008`
  - Quality Gate `PASS`

## Layout

- `parquet/`: all parquet datasets used in Run A/B/C
- `tfidf_lightgbm/`: configs + metrics + summary for TF-IDF + LightGBM runs
- `sbert/`: configs + metrics + summary for SBERT + MLP runs
- `tfidf_logreg/`: configs + metrics + summary for TF-IDF + Logistic Regression runs
- `sbert_logreg/`: configs + metrics + summary for SBERT + Logistic Regression runs
- `transformer_modernbert/`: config + metrics + summary for ModernBERT fine-tuning pilot
- `model_bundles/`: committed model artifacts for integration handoff
- `code/train.py`: training script snapshot used for these runs

## Run Definitions

- Run A: unmerged labels (`llm_label`)
  - Labels: `Lecture Notes, Other, Problem Set, Exam, Project, Reading, Solution`
- Run B: merged `Other + Project + Solution -> Other`
  - Labels: `Lecture Notes, Other, Problem Set, Exam, Reading`
- Run C: merged `Other + Project + Solution + Exam -> Other`
  - Labels: `Lecture Notes, Other, Problem Set, Reading`

## Quality Gate Rule

- Macro F1 threshold: `>= 0.50`
- Minimum per-class F1 threshold: `>= 0.30`
