# SBERT + Logistic Regression Results

Model command pattern:

```bash
docker compose run --rm training python train.py --config <config> --model sbert_logreg
```

Quality gates:
- Macro F1 >= 0.50
- Min class F1 >= 0.30

## Results Summary

| Run | Labels | Test Accuracy | Test Macro F1 | Min Class F1 | Gate | MLflow Run ID |
|---|---|---:|---:|---:|---|---|
| A | 7-class unmerged | 0.5358 | 0.3342 | 0.0000 | FAIL | `848decab02e1482ba3a92e333a8f60b9` |
| B | Other+Project+Solution merged | 0.5343 | 0.4263 | 0.1948 | FAIL | `4d26d4261a60429db421bcb118584406` |
| C | Run B + Exam merged into Other | 0.5496 | 0.5083 | 0.3459 | PASS | `a988a4287c0e457a97ff731783fe93b8` |

## Notes on Labels

- Run B `Other` includes: original `Other`, `Project`, `Solution`
- Run C `Other` includes: original `Other`, `Project`, `Solution`, `Exam`
