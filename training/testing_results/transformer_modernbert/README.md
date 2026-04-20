# Transformer Results (ModernBERT)

Model command pattern used:

```bash
docker compose run --rm training python scripts/transformer_finetune.py \
  --config configs/train_docker_run_c.yaml \
  --model-name answerdotai/ModernBERT-base \
  --output-dir artifacts/transformers_run_c_pilot \
  --epochs 1 --batch-size 4 --max-length 192 --learning-rate 1e-5 \
  --max-train-rows 2000
```

Quality gate reference (same as train.py runs):
- Macro F1 >= 0.50
- Min class F1 >= 0.30

## Result Summary

| Run | Labels | Test Accuracy | Test Macro F1 | Min Class F1 | Gate (Reference) |
|---|---|---:|---:|---:|---|
| C_pilot | Run C labels | 0.5160 | 0.4605 | 0.2320 | FAIL |

## Notes

- This is a pilot run on Run C labels with `n_train=2000` for faster comparison on CPU.
- Full ModernBERT training on all rows was not committed here; this folder captures the completed pilot experiment.
