# TF-IDF + Logistic Regression Results (Run B Only)

Model command:

```bash
docker compose run --rm training python train.py --config configs/train_docker.yaml --model tfidf_logreg
```

## Run B Summary

- Test Accuracy: `0.5516`
- Test Macro F1: `0.4198`
- Test Weighted F1: `0.5218`
- Min Class F1: `0.1793`
- Gate: `FAIL`
- MLflow Run ID: `ba5800f81651455b89ba375f9c4ea801`
