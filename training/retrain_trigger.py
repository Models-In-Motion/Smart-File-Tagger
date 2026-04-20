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


def trigger_retraining() -> bool:
    """Step 2: Train on the feedback-enriched parquet produced by run_batch_pipeline()."""
    print("[retrain_trigger] Step 2: Triggering retraining on enriched dataset...")
    result = subprocess.run(
        [
            "python",
            "train.py",
            "--config",
            "configs/train_docker.yaml",
            "--model",
            "tfidf_lightgbm",
        ],
        check=False,
    )
    return result.returncode == 0


if __name__ == "__main__":
    count = get_unchecked_feedback_count()
    print(f"[retrain_trigger] Unchecked corrections: {count}")

    if count >= FEEDBACK_THRESHOLD:
        print(
            f"[retrain_trigger] Threshold reached "
            f"({count} >= {FEEDBACK_THRESHOLD})"
        )
        pipeline_ok = run_batch_pipeline()
        if not pipeline_ok:
            print("[retrain_trigger] Batch pipeline failed — skipping retraining.")
            raise SystemExit(1)
        success = trigger_retraining()
        if success:
            log_retrain_trigger(f"feedback_threshold_{FEEDBACK_THRESHOLD}")
            print("[retrain_trigger] Retraining complete, logged.")
        else:
            print("[retrain_trigger] Retraining failed.")
    else:
        print("[retrain_trigger] Below threshold, no retraining needed.")
