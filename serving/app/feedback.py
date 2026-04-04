"""
feedback.py — Feedback Storage
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

When a user accepts, rejects, or corrects a tag prediction, that decision
is valuable training data. This file saves it to PostgreSQL so Vedant's
retraining job can use it later.

Why PostgreSQL and not just a file?
    Multiple users can submit feedback at the same time. A file would get
    corrupted if two writes happen simultaneously. PostgreSQL handles
    concurrent writes safely. It also lets Viral's batch pipeline query
    feedback by date, user, or confidence level cleanly.

Database connection:
    The connection string is read from the DB_URL environment variable.
    It is NEVER hardcoded here. Secrets do not go in source code.
    In Docker, DB_URL is passed via the .env file or docker-compose.yml.
"""

import logging
from datetime import datetime, timezone
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor

from config import get_db_url

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum for feedback type
# Using an Enum means typos like "accpt" are caught at runtime
# ---------------------------------------------------------------------------

class FeedbackType(str, Enum):
    ACCEPTED   = "accepted"     # user clicked Accept — prediction was correct
    REJECTED   = "rejected"     # user clicked Reject — prediction was wrong, user did not correct
    CORRECTED  = "corrected"    # user rejected AND provided the correct label


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_connection():
    """
    Opens a new PostgreSQL connection.

    Why not keep one connection open forever?
        A persistent connection can go stale (timeout, network blip).
        Opening a fresh connection per request is safer for our scale.
        For high-volume production systems you'd use a connection pool
        (like psycopg2.pool or asyncpg), but that's over-engineering for now.
    """
    db_url = get_db_url()
    if db_url is None:
        raise RuntimeError(
            "DB_URL environment variable is not set. "
            "Add it to your .env file: DB_URL=postgresql://user:pass@host:5432/dbname"
        )
    return psycopg2.connect(db_url)


def ensure_feedback_table_exists():
    """
    Creates the feedback table if it doesn't already exist.
    Called once when the server starts.

    Why CREATE TABLE IF NOT EXISTS?
        Idempotent — safe to call multiple times. The server can restart
        without worrying about the table already existing.
    """
    sql = """
        CREATE TABLE IF NOT EXISTS feedback (
            id                  SERIAL PRIMARY KEY,
            file_id             TEXT        NOT NULL,
            user_id             TEXT        NOT NULL,
            predicted_tag       TEXT        NOT NULL,
            confidence          FLOAT       NOT NULL,
            action_taken        TEXT        NOT NULL,
            feedback_type       TEXT        NOT NULL,
            corrected_tag       TEXT,
            model_version       TEXT        NOT NULL,
            extraction_method   TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Index on user_id so Vedant's per-user retraining query is fast
        CREATE INDEX IF NOT EXISTS idx_feedback_user_id
            ON feedback (user_id);

        -- Index on created_at so Viral's batch pipeline can filter by date
        CREATE INDEX IF NOT EXISTS idx_feedback_created_at
            ON feedback (created_at);
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        conn.close()
        log.info("Feedback table ready")
    except Exception as exc:
        # Log the error but don't crash the server.
        # Feedback storage failing should not take down the whole service.
        log.error(f"Could not create feedback table: {exc}")


# ---------------------------------------------------------------------------
# Write feedback
# ---------------------------------------------------------------------------

def save_feedback(
    file_id:            str,
    user_id:            str,
    predicted_tag:      str,
    confidence:         float,
    action_taken:       str,
    feedback_type:      FeedbackType,
    corrected_tag:      str | None,
    model_version:      str,
    extraction_method:  str | None = None,
) -> bool:
    """
    Saves one feedback event to PostgreSQL.

    Args:
        file_id           : Nextcloud file node ID
        user_id           : Nextcloud user ID
        predicted_tag     : what the model predicted (e.g. "Problem Set")
        confidence        : model confidence score (0.0 to 1.0)
        action_taken      : "auto_apply", "suggest", or "no_tag"
        feedback_type     : accepted, rejected, or corrected
        corrected_tag     : the correct label if feedback_type is corrected
        model_version     : which model version made this prediction
        extraction_method : "pdfminer", "ocr", or "plaintext"

    Returns:
        True if saved successfully, False if database write failed.
        We return False instead of raising so the API endpoint can still
        return a 200 to the user — feedback failure is not their problem.
    """
    # Validate: corrected feedback must include the correct label
    if feedback_type == FeedbackType.CORRECTED and not corrected_tag:
        log.warning("Feedback type is 'corrected' but no corrected_tag provided")
        return False

    sql = """
        INSERT INTO feedback (
            file_id, user_id, predicted_tag, confidence,
            action_taken, feedback_type, corrected_tag,
            model_version, extraction_method
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        file_id,
        user_id,
        predicted_tag,
        confidence,
        action_taken,
        feedback_type.value,
        corrected_tag,
        model_version,
        extraction_method,
    )

    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        conn.close()
        log.info(
            f"Feedback saved: user={user_id} file={file_id} "
            f"type={feedback_type.value} predicted={predicted_tag} "
            f"corrected={corrected_tag}"
        )
        return True

    except Exception as exc:
        log.error(f"Failed to save feedback: {exc}")
        return False


# ---------------------------------------------------------------------------
# Read feedback (for Vedant's retraining pipeline)
# ---------------------------------------------------------------------------

def get_feedback_for_user(user_id: str, since_days: int = 30) -> list[dict]:
    """
    Returns all feedback events for a user from the last N days.
    Vedant's retrain_job.py calls this to get user corrections for retraining.

    Args:
        user_id    : Nextcloud user ID
        since_days : how many days back to look (default: 30)

    Returns:
        List of dicts, one per feedback event.
        Empty list if no feedback found or DB is unavailable.
    """
    sql = """
        SELECT *
        FROM feedback
        WHERE user_id = %s
          AND created_at >= NOW() - INTERVAL '%s days'
        ORDER BY created_at DESC
    """
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, since_days))
            rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    except Exception as exc:
        log.error(f"Failed to fetch feedback for user {user_id}: {exc}")
        return []