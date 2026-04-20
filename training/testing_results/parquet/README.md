# Parquet Datasets (Run B Only)

This folder contains only the Run B parquet files.

## Files

- `run_b_train_llm_merged_ops.parquet`
- `run_b_eval_llm_merged_ops.parquet`

## Label Schema

- Label column: `llm_label_merged`
- Allowed labels:
  - `Lecture Notes`
  - `Other`
  - `Problem Set`
  - `Exam`
  - `Reading`

## Merge Rule

- `Other + Project + Solution -> Other`
