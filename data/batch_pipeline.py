#!/usr/bin/env python3
"""
batch_pipeline.py

Batch pipeline that compiles versioned training and evaluation datasets
from the OCW dataset and simulated production feedback. Applies candidate
selection and course-level splitting for leakage prevention.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import psycopg2
    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

VALID_LABELS = {
    "Lecture Notes", "Problem Set", "Exam",
    "Reading", "Solution", "Project", "Other",
}

RUN_B_LABELS = {"Lecture Notes", "Other", "Problem Set", "Exam", "Reading"}


def normalize_feedback_label_for_run_b(label: str) -> str | None:
    """Map free-form feedback labels into the Run B 5-label taxonomy."""
    canonical = " ".join(str(label).strip().split())
    if not canonical:
        return None

    lookup = {
        "lecture notes": "Lecture Notes",
        "reading": "Reading",
        "exam": "Exam",
        "problem set": "Problem Set",
        "problem sets": "Problem Set",
        "problem set & solution": "Problem Set",
        "problem set and solution": "Problem Set",
        "solution": "Other",
        "project": "Other",
        "other": "Other",
    }
    normalized = lookup.get(canonical.lower())
    if normalized in RUN_B_LABELS:
        return normalized
    return None


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


def load_feedback_from_postgres(db_url: str) -> pd.DataFrame | None:
    """
    Read the feedback table written by Krish's serving layer.
    Returns a DataFrame with the same columns the pipeline expects:
      event_id, timestamp, doc_id, filename, predicted_label,
      user_action, user_label, source
    Returns None (with a warning) if the connection or query fails.
    """
    if not _HAS_PSYCOPG2:
        print("[ERROR] psycopg2 not installed — cannot read from PostgreSQL. "
              "Install psycopg2-binary or use --feedback-source jsonl")
        return None
    try:
        conn = psycopg2.connect(db_url)
        # Current serving schema (feedback_type/action_taken/corrected_tag).
        query = """
            SELECT
                id::text                                            AS event_id,
                created_at                                          AS timestamp,
                file_id                                             AS doc_id,
                file_id                                             AS filename,
                predicted_tag                                       AS predicted_label,
                CASE
                    WHEN feedback_type = 'accepted'  THEN 'accept'
                    WHEN feedback_type = 'corrected' THEN 'correct'
                    ELSE 'reject'
                END                                                 AS user_action,
                corrected_tag                                       AS user_label,
                'postgres'                                          AS source
            FROM feedback
            ORDER BY created_at ASC
        """
        try:
            df = pd.read_sql(query, conn)
        except Exception:
            # Legacy schema fallback.
            legacy_query = """
                SELECT
                    id::text            AS event_id,
                    created_at          AS timestamp,
                    doc_id,
                    filename,
                    predicted_label,
                    user_action,
                    user_label,
                    'postgres'          AS source
                FROM feedback
                ORDER BY created_at ASC
            """
            df = pd.read_sql(legacy_query, conn)
        conn.close()
        print(f"[INFO] Loaded feedback from PostgreSQL: {len(df)} events")
        return df if len(df) > 0 else None
    except Exception as exc:
        print(f"[ERROR] Could not read feedback from PostgreSQL: {exc}")
        return None


def load_corrected_feedback_training_rows(
    db_url: str,
    template_columns: list[str],
) -> pd.DataFrame:
    """
    Build additional training rows from human corrections in PostgreSQL.

    Source:
      - feedback.feedback_type='corrected'  -> corrected_tag (new label)
      - predictions.extracted_text          -> training text
      - join key: file_id
    """
    if not _HAS_PSYCOPG2:
        print("[WARN] psycopg2 not installed — skipping corrected feedback row generation")
        return pd.DataFrame(columns=template_columns)

    try:
        conn = psycopg2.connect(db_url)
        query = """
            WITH latest_corrected AS (
                SELECT DISTINCT ON (f.file_id)
                    f.file_id,
                    f.corrected_tag,
                    f.created_at AS feedback_ts
                FROM feedback f
                WHERE f.feedback_type = 'corrected'
                  AND f.corrected_tag IS NOT NULL
                  AND btrim(f.corrected_tag) <> ''
                ORDER BY f.file_id, f.created_at DESC
            ),
            latest_predictions AS (
                SELECT DISTINCT ON (p.file_id)
                    p.file_id,
                    p.extracted_text,
                    p.created_at AS prediction_ts
                FROM predictions p
                WHERE p.extracted_text IS NOT NULL
                  AND btrim(p.extracted_text) <> ''
                ORDER BY p.file_id, p.created_at DESC
            )
            SELECT
                c.file_id,
                c.corrected_tag,
                p.extracted_text,
                c.feedback_ts
            FROM latest_corrected c
            JOIN latest_predictions p
              ON p.file_id = c.file_id
            ORDER BY c.feedback_ts DESC
        """
        corrected = pd.read_sql(query, conn)
        conn.close()
    except Exception as exc:
        print(f"[WARN] Could not build corrected feedback rows from PostgreSQL: {exc}")
        return pd.DataFrame(columns=template_columns)

    if corrected.empty:
        print("[INFO] No corrected feedback rows found to append to train split")
        return pd.DataFrame(columns=template_columns)

    corrected["corrected_tag"] = corrected["corrected_tag"].astype(str).str.strip()

    rows: list[dict] = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for _, rec in corrected.iterrows():
        file_id = str(rec["file_id"])
        original_label = str(rec["corrected_tag"]).strip()
        normalized_label = normalize_feedback_label_for_run_b(original_label)
        text = str(rec["extracted_text"] or "").strip()
        if file_id == "" or original_label == "" or text == "":
            continue
        if normalized_label is None:
            print(
                f"[INFO] Skipping corrected feedback label outside Run B taxonomy: "
                f"file_id={file_id} label='{original_label}'"
            )
            continue

        # Stable doc id for this feedback-derived training sample.
        doc_id = hashlib.md5(
            f"feedback::{file_id}::{normalized_label}".encode("utf-8")
        ).hexdigest()[:12]

        row = {col: None for col in template_columns}
        row["doc_id"] = doc_id
        row["extracted_text"] = text
        if "label" in row:
            row["label"] = normalized_label
        if "llm_label_merged" in row:
            row["llm_label_merged"] = normalized_label
        if "llm_label" in row:
            row["llm_label"] = normalized_label
        row["label_source"] = "user_feedback_corrected"
        row["course_id"] = "user_feedback"
        row["source_url"] = f"feedback://{file_id}"
        row["source"] = "user_feedback"
        row["ingestion_timestamp"] = now_iso
        row["dataset_version"] = "feedback_loop"

        if "filename" in row:
            row["filename"] = f"feedback_{file_id}.txt"
        if "char_count" in row:
            row["char_count"] = len(text)
        if "word_count" in row:
            row["word_count"] = len(text.split())
        if "file_size_bytes" in row:
            row["file_size_bytes"] = len(text.encode("utf-8"))
        if "text_extraction_method" in row:
            row["text_extraction_method"] = "serving_prediction"
        if "department" in row:
            row["department"] = "User Feedback"
        if "course_title" in row:
            row["course_title"] = "User Feedback"
        if "semester" in row:
            row["semester"] = "N/A"
        if "instructor" in row:
            row["instructor"] = "N/A"

        rows.append(row)

    if not rows:
        print("[INFO] No usable corrected feedback rows after filtering")
        return pd.DataFrame(columns=template_columns)

    out = pd.DataFrame(rows, columns=template_columns)
    if "doc_id" in out.columns:
        out = out.drop_duplicates(subset=["doc_id"], keep="last").reset_index(drop=True)

    print(
        "[INFO] Built "
        f"{len(out)} corrected-feedback training rows "
        "(feedback.corrected_tag + predictions.extracted_text)"
    )
    return out


def append_corrected_feedback_to_train(train_df: pd.DataFrame, db_url: str) -> pd.DataFrame:
    """
    Append corrected-feedback-derived rows to the train split before write.
    """
    extra = load_corrected_feedback_training_rows(db_url, list(train_df.columns))
    if extra.empty:
        return train_df

    # Avoid duplicate inserts across reruns while preserving all existing train rows.
    # We only filter against existing train doc_ids; we do NOT dedupe the merged train set.
    if "doc_id" in train_df.columns and "doc_id" in extra.columns:
        extra = extra[~extra["doc_id"].isin(set(train_df["doc_id"]))].reset_index(drop=True)
        if extra.empty:
            print("[INFO] Corrected-feedback rows already present in train split; nothing to append")
            return train_df

    before = len(train_df)
    merged = pd.concat([train_df, extra], ignore_index=True)

    print(
        f"[INFO] Train rows after corrected feedback append: {before} -> {len(merged)} "
        f"(added {len(merged) - before})"
    )
    return merged


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def apply_candidate_selection(df: pd.DataFrame, feedback: pd.DataFrame | None) -> pd.DataFrame:
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


def validate_training_set(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """
    Runs after train/eval splits are created, before writing to parquet.
    Hard checks raise ValueError and block the write.
    Warnings log but do not block.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    REQUIRED_COLS = {
        "doc_id", "extracted_text", "label", "label_source",
        "course_id", "source_url", "source", "ingestion_timestamp", "dataset_version",
    }

    for split_name, df in [("train", train_df), ("eval", eval_df)]:

        # 1. Required columns present + no nulls
        for col in REQUIRED_COLS:
            if col not in df.columns:
                errors.append(f"[{split_name}] Missing required column: {col}")
            elif df[col].isnull().any():
                errors.append(f"[{split_name}] Nulls in required column '{col}'")

        # 2. No duplicate doc_id within split
        if "doc_id" in df.columns:
            dupes = int(df["doc_id"].duplicated().sum())
            if dupes:
                errors.append(f"[{split_name}] {dupes} duplicate doc_id values")

        # 3. At least 2 labels represented
        if "label" in df.columns:
            n_labels = df["label"].nunique()
            if n_labels < 2:
                errors.append(f"[{split_name}] Only {n_labels} label(s) present — need at least 2")

        # 4. Minimum row counts
        if split_name == "train" and len(df) < 100:
            errors.append(f"[train] Only {len(df)} rows — minimum viable training set is 100")
        if split_name == "eval" and len(df) < 20:
            errors.append(f"[eval] Only {len(df)} rows — minimum viable eval set is 20")

        # 5. No label in train with fewer than 5 examples
        if split_name == "train" and "label" in df.columns:
            for label, count in df["label"].value_counts().items():
                if count < 5:
                    errors.append(f"[train] Label '{label}' has only {count} examples — need at least 5")

    # 6. Zero course_id overlap (leakage check)
    if "course_id" in train_df.columns and "course_id" in eval_df.columns:
        overlap = set(train_df["course_id"].unique()) & set(eval_df["course_id"].unique())
        if overlap:
            errors.append(f"Course overlap detected — data leakage: {overlap}")

    # 7. Label distribution: no label should be >3x its train proportion in eval
    if "label" in train_df.columns and "label" in eval_df.columns:
        train_props = (train_df["label"].value_counts() / len(train_df))
        eval_props  = (eval_df["label"].value_counts()  / len(eval_df))
        for label in train_props.index:
            if label in eval_props.index:
                t = train_props[label]
                e = eval_props[label]
                if t > 0 and (e / t) > 3:
                    warnings.append(
                        f"Label '{label}' is {e:.1%} in eval vs {t:.1%} in train — >3x skew"
                    )

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n=== Training Set Validation ===")
    print(f"{'Label':<20} {'Train':>8} {'Eval':>8}")
    print("-" * 38)
    all_labels = sorted(set(train_df.get("label", pd.Series()).unique()) |
                        set(eval_df.get("label",  pd.Series()).unique()))
    train_counts = train_df["label"].value_counts() if "label" in train_df.columns else {}
    eval_counts  = eval_df["label"].value_counts()  if "label" in eval_df.columns  else {}
    for label in all_labels:
        print(f"  {label:<18} {train_counts.get(label, 0):>8} {eval_counts.get(label, 0):>8}")
    print(f"  {'TOTAL':<18} {len(train_df):>8} {len(eval_df):>8}")
    print(f"\nWarnings : {len(warnings)}")
    for w in warnings:
        print(f"  [WARN] {w}")
    print(f"Errors   : {len(errors)}")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        print("Status   : FAIL — splits NOT written")
        raise ValueError("Training set validation failed")

    print("Status   : PASS")


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
                        help="Path to feedback JSONL (used when --feedback-source jsonl)")
    parser.add_argument("--feedback-source", default="jsonl", choices=["jsonl", "postgres"],
                        help="Where to read feedback from: jsonl (default) or postgres")
    parser.add_argument("--db-url", default=os.environ.get("DB_URL", "postgresql://tagger:tagger@postgres:5432/tagger"),
                        help="PostgreSQL connection string (used when --feedback-source postgres)")
    parser.add_argument("--output-dir", default="artifacts/versions")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # Append-only mode: skip ETL/split, just append feedback rows to an existing train parquet.
    parser.add_argument("--base-train", default=None,
                        help="Path to existing train.parquet to append feedback rows to (skips full ETL)")
    parser.add_argument("--output", default=None,
                        help="Output path for the enriched train.parquet (used with --base-train)")
    args = parser.parse_args()

    # Append-only mode — used by retrain_trigger.py to enrich the base training set with
    # user corrections before each retraining run, without re-running the full ETL.
    if args.base_train:
        if not args.output:
            print("[ERROR] --output is required when using --base-train", file=sys.stderr)
            sys.exit(1)
        if args.feedback_source != "postgres":
            print("[ERROR] --base-train mode requires --feedback-source postgres", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Append-only mode: loading base train from {args.base_train}")
        train_df = pd.read_parquet(args.base_train)
        train_df = append_corrected_feedback_to_train(train_df, args.db_url)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(out_path, index=False)
        print(f"[INFO] Enriched train.parquet written to {out_path} ({len(train_df)} rows)")
        return

    df = load_dataset(args.dataset)

    if args.feedback_source == "postgres":
        feedback = load_feedback_from_postgres(args.db_url)
    else:
        feedback = load_feedback(args.feedback)

    df = apply_candidate_selection(df, feedback)

    train_df, eval_df = course_level_split(df, args.eval_ratio, args.seed)

    # Human-in-the-loop closure:
    # Add corrected feedback rows (label=corrected_tag, text from predictions.extracted_text)
    # to the training split before writing train.parquet.
    if args.feedback_source == "postgres":
        train_df = append_corrected_feedback_to_train(train_df, args.db_url)

    validate_training_set(train_df, eval_df)

    out = write_outputs(train_df, eval_df, args.output_dir, args.version)

    print_summary(train_df, eval_df, out)


if __name__ == "__main__":
    main()
