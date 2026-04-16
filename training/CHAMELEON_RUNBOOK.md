# Chameleon Runbook (Training)

## 1) On your Chameleon VM: clone repo
```bash
git clone <your-team-repo-url>
cd MLOPS_Training
```

## 2) Start MLflow service in Docker
```bash
./scripts/chameleon/start_mlflow.sh
```

If your VM public IP is `X.X.X.X`, MLflow UI is at:
`http://X.X.X.X:5000`

## 3) Run experiments in Docker
Set tracking URI:
```bash
export MLFLOW_TRACKING_URI=http://X.X.X.X:5000
```

Run each model:
```bash
./scripts/chameleon/run_training_docker.sh tfidf_logreg
./scripts/chameleon/run_training_docker.sh tfidf_lightgbm
./scripts/chameleon/run_training_docker.sh sbert_logreg
./scripts/chameleon/run_training_docker.sh sbert_mlp
```

## 4) Confirm runs in MLflow
Open MLflow UI, click experiment `nextcloud-doc-tagger-training`, copy run links and metrics to:
`reports/training_runs_table_template.md`

## 5) Demo video checklist
- Start MLflow script
- Run one training container end-to-end
- Show run appearing in MLflow UI with params/metrics/artifacts
