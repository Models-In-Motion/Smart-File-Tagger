# Training Testing Branch Artifacts

This folder contains temporary experiment artifacts for three label strategies (Run A/B/C) evaluated with multiple models:

- `tfidf_lightgbm`
- `sbert_mlp` (under `sbert/`)
- `tfidf_logreg`
- `sbert_logreg`
- `answerdotai/ModernBERT-base` pilot (under `transformer_modernbert/`)

No production dataset was overwritten. These files are for comparison and reproducibility.

## Layout

- `parquet/`: all parquet datasets used in Run A/B/C
- `tfidf_lightgbm/`: configs + metrics + summary for TF-IDF + LightGBM runs
- `sbert/`: configs + metrics + summary for SBERT + MLP runs
- `tfidf_logreg/`: configs + metrics + summary for TF-IDF + Logistic Regression runs
- `sbert_logreg/`: configs + metrics + summary for SBERT + Logistic Regression runs
- `transformer_modernbert/`: config + metrics + summary for ModernBERT fine-tuning pilot
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
