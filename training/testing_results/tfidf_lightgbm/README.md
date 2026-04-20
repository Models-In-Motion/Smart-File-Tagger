# TF-IDF + LightGBM Results (Run B Only)

Model command:

```bash
docker compose run --rm training python train.py --config configs/train_docker.yaml --model tfidf_lightgbm
```

Quality gates:
- Macro F1 >= 0.50
- Min class F1 >= 0.30

## Run B Summary

- Test Accuracy: `0.6351`
- Test Macro F1: `0.5574`
- Test Weighted F1: `0.6199`
- Min Class F1: `0.3008`
- Gate: `PASS`
- MLflow Run ID: `d3de192e7e8543bba4f856302c662bb8`
