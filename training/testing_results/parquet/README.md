# Parquet Datasets (Run A/B/C)

These are the exact datasets used for the six model runs.

## Files

- `run_a_train_llm.parquet`
- `run_a_eval_llm.parquet`
- `run_b_train_llm_merged_ops.parquet`
- `run_b_eval_llm_merged_ops.parquet`
- `run_c_train_llm_run_c.parquet`
- `run_c_eval_llm_run_c.parquet`

## Label Columns

- Run A files use label column: `llm_label`
- Run B files use label column: `llm_label_merged`
- Run C files use label column: `llm_label_run_c`

## Label Distribution (combined train+eval)

- Run A
  - Lecture Notes: 37.31%
  - Problem Set: 21.86%
  - Reading: 20.84%
  - Solution: 9.86%
  - Exam: 5.85%
  - Other: 2.41%
  - Project: 1.88%

- Run B
  - Lecture Notes: 37.31%
  - Problem Set: 21.86%
  - Reading: 20.84%
  - Other: 14.15%
  - Exam: 5.85%

- Run C
  - Lecture Notes: 37.31%
  - Problem Set: 21.86%
  - Reading: 20.84%
  - Other: 19.99%
