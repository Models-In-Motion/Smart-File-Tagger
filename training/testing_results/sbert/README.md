# SBERT + MLP Results (Run B Only)

Model command:

```bash
docker compose run --rm training python train.py --config configs/train_docker.yaml --model sbert_mlp
```

## Run B Summary

- Test Accuracy: `0.4627`
- Test Macro F1: `0.3910`
- Test Weighted F1: `0.4626`
- Min Class F1: `0.2059`
- Gate: `FAIL`
- MLflow Run ID: `0367a1c90dc449619e4d92907e39e3ef`
