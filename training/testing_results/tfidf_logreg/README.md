# TF-IDF + Logistic Regression Results

Model command pattern:

```bash
docker compose run --rm training python train.py --config <config> --model tfidf_logreg
```

Quality gates:
- Macro F1 >= 0.50
- Min class F1 >= 0.30

## Results Summary

| Run | Labels | Test Accuracy | Test Macro F1 | Min Class F1 | Gate | MLflow Run ID |
|---|---|---:|---:|---:|---|---|
| A | 7-class unmerged | 0.5516 | 0.3081 | 0.0000 | FAIL | `280efaa620df4c2fa69c2a9d0deafe68` |
| B | Other+Project+Solution merged | 0.5516 | 0.4198 | 0.1793 | FAIL | `ba5800f81651455b89ba375f9c4ea801` |
| C | Run B + Exam merged into Other | 0.5615 | 0.5158 | 0.3523 | PASS | `34e7983357a1468cad5e876f7fc189d4` |

## Notes on Labels

- Run B `Other` includes: original `Other`, `Project`, `Solution`
- Run C `Other` includes: original `Other`, `Project`, `Solution`, `Exam`
