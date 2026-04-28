"""
retrain_trigger.py - Checks PostgreSQL feedback volume and triggers retraining.

Runs training when corrected feedback since last trigger reaches a threshold.
"""

from __future__ import annotations

import os
import subprocess

import psycopg2

FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", "50"))
DB_URL = os.getenv("DB_URL", "postgresql://tagger:tagger@postgres:5432/tagger")

# Paths inside the retrain-cron container.
BASE_TRAIN_PATH    = "/app/testing_results/parquet/run_b_train_llm_merged_ops.parquet"
FEEDBACK_TRAIN_PATH = "/app/testing_results/parquet/run_b_train_with_feedback.parquet"
BATCH_PIPELINE_PATH = "/app/data_pipeline/batch_pipeline.py"
VALIDATE_SCRIPT_PATH = "/app/data_pipeline/validate_dataset.py"
EVAL_PATH = "/app/testing_results/parquet/run_b_eval_llm_merged_ops.parquet"
RUN_B_LABELS = ["Lecture Notes", "Other", "Problem Set", "Exam", "Reading"]


def ensure_retrain_log_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_log (
                id SERIAL PRIMARY KEY,
                triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reason TEXT NOT NULL
            )
            """
        )
    conn.commit()


def get_unchecked_feedback_count() -> int:
    try:
        conn = psycopg2.connect(DB_URL)
        ensure_retrain_log_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM feedback
                WHERE feedback_type = 'corrected'
                  AND created_at > (
                      SELECT COALESCE(MAX(triggered_at), '1970-01-01'::timestamptz)
                      FROM retrain_log
                  )
                """
            )
            count = int(cur.fetchone()[0])
        conn.close()
        return count
    except Exception as exc:
        print(f"[retrain_trigger] DB error while counting feedback: {exc}")
        return 0


def log_retrain_trigger(reason: str) -> None:
    conn = psycopg2.connect(DB_URL)
    ensure_retrain_log_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO retrain_log (reason) VALUES (%s)",
            (reason,),
        )
    conn.commit()
    conn.close()


def run_batch_pipeline() -> bool:
    """Step 1: Enrich base train parquet with latest user corrections from PostgreSQL."""
    print("[retrain_trigger] Step 1: Running batch pipeline to append user feedback...")
    result = subprocess.run(
        [
            "python", BATCH_PIPELINE_PATH,
            "--base-train", BASE_TRAIN_PATH,
            "--output", FEEDBACK_TRAIN_PATH,
            "--feedback-source", "postgres",
            "--db-url", DB_URL,
        ],
        check=False,
    )
    if result.returncode != 0:
        print("[retrain_trigger] Batch pipeline failed — aborting retraining.")
    return result.returncode == 0


def run_dataset_validation() -> bool:
    """
    Step 2: Validate enriched train/eval data quality before training.
    Uses validate_dataset.py with Run B settings.
    """
    print("[retrain_trigger] Step 2: Running dataset validation gate...")
    result = subprocess.run(
        [
            "python", VALIDATE_SCRIPT_PATH,
            "--raw", FEEDBACK_TRAIN_PATH,
            "--train", FEEDBACK_TRAIN_PATH,
            "--eval", EVAL_PATH,
            "--label-col", "llm_label_merged",
            "--required-labels", *RUN_B_LABELS,
            "--min-examples", "50",
            "--min-words", "10",
            "--max-bad-text-pct", "1.0",
        ],
        check=False,
    )
    if result.returncode != 0:
        print("[retrain_trigger] Dataset validation failed — aborting retraining.")
    return result.returncode == 0


def trigger_retraining() -> bool:
    """Step 3: Train on the feedback-enriched parquet produced by run_batch_pipeline()."""
    print("[retrain_trigger] Step 3: Triggering retraining on enriched dataset...")
    result = subprocess.run(
        [
            "python",
            "train.py",
            "--config",
            "configs/train_docker.yaml",
            "--model",
            "tfidf_lightgbm",
            "--data-path",
            FEEDBACK_TRAIN_PATH,
        ],
        check=False,
    )
    return result.returncode == 0


if __name__ == "__main__":
    count = get_unchecked_feedback_count()
    print(f"[retrain_trigger] Unchecked corrections: {count}")

    # Time-based fallback: force retraining if last run was > 24 hours ago
    force_retrain = False
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(
            "SELECT EXTRACT(EPOCH FROM (NOW() - COALESCE(MAX(triggered_at), "
            "'1970-01-01'::timestamptz))) / 3600 FROM retrain_log"
        )
        hours_since_last = float(cur.fetchone()[0])
        conn.close()
        if hours_since_last >= 24:
            force_retrain = True
            print(f"[retrain_trigger] Force retraining — {hours_since_last:.1f}h since last run (>= 24h)")
    except Exception as e:
        print(f"[retrain_trigger] Could not check last retrain time: {e}")

    if count >= FEEDBACK_THRESHOLD or force_retrain:
        if force_retrain and count < FEEDBACK_THRESHOLD:
            print(f"[retrain_trigger] Time-based trigger (only {count} corrections but 24h elapsed)")
        else:
            print(
                f"[retrain_trigger] Threshold reached "
                f"({count} >= {FEEDBACK_THRESHOLD})"
            )
        pipeline_ok = run_batch_pipeline()
        if not pipeline_ok:
            print("[retrain_trigger] Batch pipeline failed — skipping retraining.")
            raise SystemExit(1)
        else:
            validation_ok = run_dataset_validation()
            if not validation_ok:
                print("[retrain_trigger] Validation failed — skipping retraining.")
                raise SystemExit(1)
            else:
                success = trigger_retraining()
                if success:
                    log_retrain_trigger(f"feedback_threshold_{FEEDBACK_THRESHOLD}")
                    print("[retrain_trigger] Retraining complete, logged.")
                else:
                    print("[retrain_trigger] Retraining failed.")
                    raise SystemExit(1)
    else:
        print("[retrain_trigger] Below threshold, no retraining needed.")
