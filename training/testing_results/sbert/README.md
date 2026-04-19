# SBERT Results (sbert_mlp)

Model command pattern:

```bash
docker compose run --rm training python train.py --config <config> --model sbert_mlp
```

Quality gates:
- Macro F1 >= 0.50
- Min class F1 >= 0.30

## Results Summary

| Run | Labels | Test Accuracy | Test Macro F1 | Min Class F1 | Gate | MLflow Run ID |
|---|---|---:|---:|---:|---|---|
| A | 7-class unmerged | 0.4449 | 0.3030 | 0.0625 | FAIL | `4f8a0b8adce34c20b366733cb26b9c09` |
| B | Other+Project+Solution merged | 0.4627 | 0.3910 | 0.2059 | FAIL | `0367a1c90dc449619e4d92907e39e3ef` |
| C | Run B + Exam merged into Other | 0.4706 | 0.4402 | 0.3032 | FAIL (macro) | `f56c82b9e75944f79d553276dbe8eaa9` |

## Notes

- Same label strategies as TF-IDF runs.
- Merging improves SBERT metrics, but macro F1 did not cross 0.50 in these runs.
- Run C hit an MLP convergence warning (`max_iter=200`).
