# TF-IDF + LightGBM Results

Model command pattern:

```bash
docker compose run --rm training python train.py --config <config> --model tfidf_lightgbm
```

Quality gates:
- Macro F1 >= 0.50
- Min class F1 >= 0.30

## Results Summary

| Run | Labels | Test Accuracy | Test Macro F1 | Min Class F1 | Gate | MLflow Run ID |
|---|---|---:|---:|---:|---|---|
| A | 7-class unmerged | 0.6351 | 0.4246 | 0.0000 | FAIL | `0cebc2cb46e149509676b611a343ec9a` |
| B | Other+Project+Solution merged | 0.6351 | 0.5574 | 0.3008 | PASS | `d3de192e7e8543bba4f856302c662bb8` |
| C | Run B + Exam merged into Other | 0.6321 | 0.6006 | 0.4801 | PASS | `5c89b4f3655f46f796184423a46e333a` |

## Notes on Labels

- Run B `Other` includes: original `Other`, `Project`, `Solution`
- Run C `Other` includes: original `Other`, `Project`, `Solution`, `Exam`
