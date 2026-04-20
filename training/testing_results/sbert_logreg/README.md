# SBERT + Logistic Regression Results (Run B Only)

Model command:

```bash
docker compose run --rm training python train.py --config configs/train_docker.yaml --model sbert_logreg
```

## Run B Summary

- Test Accuracy: `0.5343`
- Test Macro F1: `0.4263`
- Test Weighted F1: `0.5111`
- Min Class F1: `0.1948`
- Gate: `FAIL`
- MLflow Run ID: `4d26d4261a60429db421bcb118584406`
