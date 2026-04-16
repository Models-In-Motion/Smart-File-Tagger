# Training Checklist (Vedant)

## 1) Data ready
- [x] Balanced subset created at `data/ocw_subset_200.parquet`
- [ ] Full/larger dataset run planned after smoke tests

## 2) JSON contract with team
- [ ] `input.json` agreed with Krish + Viral
- [ ] `output.json` agreed with Krish + Viral

## 3) Training artifacts in repo
- [x] `train.py`
- [x] `configs/train.yaml`
- [x] `requirements.txt`
- [x] `scripts/run_all_models.sh`

## 4) Experiments to run
- [ ] `tfidf_logreg`
- [ ] `tfidf_lightgbm`
- [ ] `sbert_logreg`
- [ ] `sbert_mlp`

## 5) MLflow requirements
- [ ] MLflow server running on Chameleon
- [ ] All runs logged to MLflow (params, metrics, artifacts)

## 6) Submission artifacts
- [ ] Filled runs table (`reports/training_runs_table_template.md`)
- [ ] Demo video showing container run on Chameleon + MLflow run page
- [ ] Commit messages include AI disclosure when applicable
