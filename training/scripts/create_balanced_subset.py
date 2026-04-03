#!/usr/bin/env python3
"""Create a balanced subset from an OCW parquet dataset.

Example:
python create_balanced_subset.py \
  --input /Users/vedantpradhan/Downloads/ocw_dataset.parquet \
  --output data/ocw_subset_200.parquet \
  --total-size 200 \
  --labels "Lecture Notes" "Lecture Transcript" "Problem Set" "Exam" "Reading / Reference"
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_LABELS = [
    "Lecture Notes",
    "Lecture Transcript",
    "Problem Set",
    "Exam",
    "Reading / Reference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced Parquet subset")
    parser.add_argument("--input", required=True, help="Path to source parquet")
    parser.add_argument("--output", required=True, help="Path for subset parquet")
    parser.add_argument("--label-col", default="doc_type", help="Label column name")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--total-size", type=int, default=200, help="Total subset size")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Labels to include (space separated)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.input)

    if args.label_col not in table.column_names:
        raise ValueError(
            f"Label column '{args.label_col}' not found. Available: {table.column_names}"
        )
    if args.text_col not in table.column_names:
        raise ValueError(
            f"Text column '{args.text_col}' not found. Available: {table.column_names}"
        )

    labels = table[args.label_col].to_pylist()
    texts = table[args.text_col].to_pylist()

    indices_by_label: dict[str, list[int]] = defaultdict(list)
    for idx, (label, text) in enumerate(zip(labels, texts)):
        if label in args.labels and isinstance(text, str) and text.strip():
            indices_by_label[label].append(idx)

    n_labels = len(args.labels)
    if n_labels == 0:
        raise ValueError("No labels were provided")

    per_label = args.total_size // n_labels
    if per_label * n_labels != args.total_size:
        raise ValueError(
            f"total-size ({args.total_size}) must be divisible by number of labels ({n_labels})"
        )

    random.seed(args.seed)
    chosen_indices: list[int] = []

    for label in args.labels:
        available = indices_by_label.get(label, [])
        if len(available) < per_label:
            raise ValueError(
                f"Not enough rows for label '{label}'. Needed {per_label}, found {len(available)}"
            )
        chosen_indices.extend(random.sample(available, per_label))

    random.shuffle(chosen_indices)

    subset = table.take(pa.array(chosen_indices, type=pa.int64()))
    pq.write_table(subset, args.output)

    subset_counts = Counter(subset[args.label_col].to_pylist())
    print(f"Wrote subset to: {args.output}")
    print(f"Rows: {subset.num_rows}")
    print("Label distribution:")
    for label in args.labels:
        print(f"  {label}: {subset_counts.get(label, 0)}")


if __name__ == "__main__":
    main()
