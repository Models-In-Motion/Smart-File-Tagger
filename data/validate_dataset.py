#!/usr/bin/env python3
"""
validate_dataset.py

Standalone data quality validation gate for OCW dataset artifacts.

Checks:
1) Text quality on raw ETL output
2) Label distribution minimums on train/eval
3) No course leakage between train/eval
4) Extraction backend audit (must not contain `strings`)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REQUIRED_LABELS = [
    "Lecture Notes",
    "Problem Set",
    "Exam",
    "Solution",
    "Reading",
    "Project",
    "Other",
]

PDF_GARBAGE_RE = re.compile(r"<<\/Size|XRefStm|\/Root \d+|endobj|endstream", re.IGNORECASE)


def looks_like_garbage(text: str) -> bool:
    if not isinstance(text, str):
        return True
    if PDF_GARBAGE_RE.search(text):
        return True
    sample = text[:500]
    if not sample:
        return True
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in sample) / max(len(sample), 1)
    return alpha_ratio < 0.5


def validate_text_quality(df: pd.DataFrame) -> tuple[list[str], pd.Series, int]:
    issues: list[str] = []

    text_series = df["extracted_text"].astype(str)

    word_counts = text_series.str.split().str.len()
    too_short_mask = word_counts < 20
    too_short = int(too_short_mask.sum())
    issues.append(f"Too short (<20 words): {too_short} rows ({too_short/len(df)*100:.1f}%)")

    garbage_mask = text_series.apply(looks_like_garbage)
    garbage_rows = int(garbage_mask.sum())
    issues.append(f"Likely garbage text: {garbage_rows} rows ({garbage_rows/len(df)*100:.1f}%)")

    pdf_header_mask = text_series.str.contains(PDF_GARBAGE_RE, na=False)
    pdf_headers = int(pdf_header_mask.sum())
    issues.append(f"Contains PDF binary headers: {pdf_headers} rows ({pdf_headers/len(df)*100:.1f}%)")

    methods = df["text_extraction_method"].value_counts(dropna=False)
    issues.append("Extraction methods:\n" + methods.to_string())

    strings_rows = int((df["text_extraction_method"].astype(str).str.lower() == "strings").sum())

    bad_mask = garbage_mask | pdf_header_mask | too_short_mask
    return issues, bad_mask, strings_rows


def validate_label_distribution(df: pd.DataFrame, split_name: str, min_examples: int) -> tuple[list[str], bool]:
    issues: list[str] = []
    dist = df["label"].value_counts()
    ok = True

    for label in REQUIRED_LABELS:
        count = int(dist.get(label, 0))
        if count < min_examples:
            issues.append(f"FAIL [{split_name}]: {label} has only {count} examples (need {min_examples})")
            ok = False
        else:
            issues.append(f"OK   [{split_name}]: {label} = {count}")

    unexpected = sorted(set(dist.index) - set(REQUIRED_LABELS))
    if unexpected:
        issues.append(f"WARNING [{split_name}]: Unexpected labels found: {unexpected}")

    return issues, ok


def validate_no_leakage(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[str, bool]:
    train_courses = set(train_df["course_id"].unique())
    eval_courses = set(eval_df["course_id"].unique())
    overlap = train_courses & eval_courses

    if overlap:
        return (
            f"FAIL: {len(overlap)} courses appear in both train and eval: {sorted(list(overlap))[:10]}",
            False,
        )
    return (
        f"OK: No course overlap between train ({len(train_courses)} courses) and eval ({len(eval_courses)} courses)",
        True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OCW dataset quality and split integrity")
    parser.add_argument("--raw", default="data/artifacts/ocw_dataset.parquet")
    parser.add_argument("--train", default="data/artifacts/versions/v3/train.parquet")
    parser.add_argument("--eval", dest="eval_path", default="data/artifacts/versions/v3/eval.parquet")
    parser.add_argument("--min-examples", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_path = Path(args.raw)
    train_path = Path(args.train)
    eval_path = Path(args.eval_path)

    print("=" * 60)
    print("DATA QUALITY VALIDATION REPORT")
    print("=" * 60)

    missing = [p for p in [raw_path, train_path, eval_path] if not p.exists()]
    if missing:
        for p in missing:
            print(f"[FAIL] Missing file: {p}")
        return 1

    raw = pd.read_parquet(raw_path)
    train = pd.read_parquet(train_path)
    eval_df = pd.read_parquet(eval_path)

    print(f"\nRaw dataset: {len(raw)} rows ({raw_path})")
    print(f"Train split: {len(train)} rows ({train_path})")
    print(f"Eval split:  {len(eval_df)} rows ({eval_path})")

    required_cols = ["extracted_text", "label", "course_id", "text_extraction_method"]
    for col in required_cols:
        if col not in raw.columns:
            print(f"\n[FAIL] Raw dataset missing required column: {col}")
            return 1

    print("\n--- CHECK 1: Text Quality ---")
    text_issues, bad_mask, strings_rows = validate_text_quality(raw)
    for issue in text_issues:
        print(issue)
    bad_rows = int(bad_mask.sum())
    bad_pct = (bad_rows / len(raw) * 100) if len(raw) else 0.0
    print(f"Total bad rows: {bad_rows} ({bad_pct:.1f}%)")
    print(f"Rows using strings backend: {strings_rows}")

    print("\n--- CHECK 2: Label Distribution (train) ---")
    train_issues, train_ok = validate_label_distribution(train, "train", args.min_examples)
    for issue in train_issues:
        print(issue)

    print("\n--- CHECK 2: Label Distribution (eval) ---")
    eval_issues, eval_ok = validate_label_distribution(eval_df, "eval", args.min_examples)
    for issue in eval_issues:
        print(issue)

    print("\n--- CHECK 3: No Data Leakage ---")
    leakage_msg, leakage_ok = validate_no_leakage(train, eval_df)
    print(leakage_msg)

    print("\n" + "=" * 60)
    failures: list[str] = []
    if strings_rows > 0:
        failures.append(f"strings backend present ({strings_rows} rows)")
    if bad_rows > 0:
        failures.append(f"garbage/short text rows present ({bad_rows})")
    if not train_ok:
        failures.append("train label minimum check failed")
    if not eval_ok:
        failures.append("eval label minimum check failed")
    if not leakage_ok:
        failures.append("train/eval leakage check failed")

    if failures:
        print("VERDICT: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
