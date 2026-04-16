#!/usr/bin/env python3
"""
batch_pipeline.py

Batch pipeline that compiles versioned training and evaluation datasets
from the OCW dataset and simulated production feedback. Applies candidate
selection and course-level splitting for leakage prevention.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VALID_LABELS = {
    "Lecture Notes", "Problem Set", "Exam", "Syllabus",
    "Reading", "Solution", "Project", "Recitation", "Lab", "Other",
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded dataset: {len(df)} rows from {path}")
    return df


def load_feedback(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"[INFO] No feedback file found at {path} — skipping feedback merge")
        return None
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"[INFO] Loaded feedback: {len(df)} events from {path}")
    return df


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def apply_candidate_selection(df: pd.DataFrame, feedback: pd.DataFrame | None) -> pd.DataFrame:
    before = len(df)

    # Drop low-confidence labels (Other rows with no structural rule)
    df = df[df["label_source"] != "no_rule_matched"].copy()
    print(f"[INFO] After dropping no_rule_matched: {len(df)} rows (dropped {before - len(df)})")

    # Drop short texts (likely extraction failures)
    before2 = len(df)
    df = df[df["char_count"] >= 200].copy()
    print(f"[INFO] After dropping char_count < 200: {len(df)} rows (dropped {before2 - len(df)})")

    # Apply feedback corrections
    if feedback is not None and len(feedback) > 0:
        # Only accept/correct actions — ignore means unlabeled
        engaged = feedback[feedback["user_action"].isin(["accept", "correct"])].copy()

        # For corrected events, update the label to user_label
        corrections = engaged[
            (engaged["user_action"] == "correct") & engaged["user_label"].notna()
        ][["doc_id", "user_label"]].drop_duplicates("doc_id")

        if len(corrections) > 0:
            correction_map = dict(zip(corrections["doc_id"], corrections["user_label"]))
            mask = df["doc_id"].isin(correction_map)
            df.loc[mask, "label"] = df.loc[mask, "doc_id"].map(correction_map)
            print(f"[INFO] Applied {mask.sum()} label corrections from feedback")

        # Keep only docs that appeared in feedback (accept or correct)
        feedback_doc_ids = set(engaged["doc_id"])
        in_feedback = df["doc_id"].isin(feedback_doc_ids)
        not_in_feedback = ~in_feedback
        # Docs with accepted/corrected feedback are high-confidence; keep all others too
        # (feedback is a subset of docs — we keep everything that passed quality filters)
        print(f"[INFO] {in_feedback.sum()} docs have accepted/corrected feedback signal")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage-safe course-level split
# ---------------------------------------------------------------------------

def course_level_split(df: pd.DataFrame, eval_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import random
    rng = random.Random(seed)

    courses = sorted(df["course_id"].unique().tolist())
    rng.shuffle(courses)

    n_eval = max(1, round(len(courses) * eval_ratio))
    eval_courses = set(courses[:n_eval])
    train_courses = set(courses[n_eval:])

    train_df = df[df["course_id"].isin(train_courses)].copy()
    eval_df = df[df["course_id"].isin(eval_courses)].copy()

    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def label_distribution(df: pd.DataFrame) -> dict:
    return df["label"].value_counts().to_dict()


def write_outputs(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: str,
    version: str,
) -> Path:
    out = Path(output_dir) / version
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False)
    eval_df.to_parquet(out / "eval.parquet", index=False)

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train": {
            "num_rows": len(train_df),
            "course_ids": sorted(train_df["course_id"].unique().tolist()),
            "label_distribution": label_distribution(train_df),
        },
        "eval": {
            "num_rows": len(eval_df),
            "course_ids": sorted(eval_df["course_id"].unique().tolist()),
            "label_distribution": label_distribution(eval_df),
        },
    }

    with open(out / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out


def print_summary(train_df: pd.DataFrame, eval_df: pd.DataFrame, out: Path) -> None:
    total = len(train_df) + len(eval_df)
    train_courses = set(train_df["course_id"].unique())
    eval_courses = set(eval_df["course_id"].unique())
    overlap = train_courses & eval_courses

    print(f"\n=== Batch Pipeline Summary ===")
    print(f"Total rows:    {total}")
    print(f"Train:         {len(train_df)} rows  |  courses: {sorted(train_courses)}")
    print(f"Eval:          {len(eval_df)} rows  |  courses: {sorted(eval_courses)}")
    print(f"Course overlap (must be 0): {len(overlap)}")

    print(f"\nTrain label distribution:")
    for label, count in sorted(label_distribution(train_df).items()):
        print(f"  {label:<20} {count}")

    print(f"\nEval label distribution:")
    for label, count in sorted(label_distribution(eval_df).items()):
        print(f"  {label:<20} {count}")

    print(f"\nOutputs written to: {out}/")
    print(f"  train.parquet, eval.parquet, split_metadata.json")

    if overlap:
        print(f"\n[ERROR] course_id overlap detected: {overlap}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n[OK] No course_id overlap — leakage prevention verified")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pipeline: candidate selection + versioned split.")
    parser.add_argument("--dataset", default="artifacts/ocw_dataset.parquet")
    parser.add_argument("--feedback", default="artifacts/production_feedback.jsonl",
                        help="Path to feedback JSONL (optional)")
    parser.add_argument("--output-dir", default="artifacts/versions")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    feedback = load_feedback(args.feedback)

    df = apply_candidate_selection(df, feedback)

    train_df, eval_df = course_level_split(df, args.eval_ratio, args.seed)

    out = write_outputs(train_df, eval_df, args.output_dir, args.version)

    print_summary(train_df, eval_df, out)


if __name__ == "__main__":
    main()
