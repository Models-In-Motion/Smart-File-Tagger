"""
feedback.py — Feedback Storage and Prediction Logging
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

Two responsibilities:
    1. Save user feedback (accept/reject/correct) to PostgreSQL
    2. Log every prediction to PostgreSQL for drift monitoring

Why PostgreSQL and not just a file?
    Multiple users can submit feedback at the same time. A file would get
    corrupted if two writes happen simultaneously. PostgreSQL handles
    concurrent writes safely. It also lets Viral's batch pipeline query
    feedback by date, user, or confidence level cleanly.

Database connection:
    The connection string is read from the DB_URL environment variable.
    It is NEVER hardcoded here. Secrets do not go in source code.
    In Docker, DB_URL is passed via the .env file or docker-compose.yml.

Assisted by Claude Sonnet 4.5
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
# ---------------------------------------------------------------------------

class FeedbackType(str, Enum):
    ACCEPTED   = "accepted"     # user clicked Accept — prediction was correct
    REJECTED   = "rejected"     # user clicked Reject — prediction was wrong
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

        -- Index on feedback_type so retrain trigger can count corrections fast
        CREATE INDEX IF NOT EXISTS idx_feedback_type
            ON feedback (feedback_type);
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        conn.close()
        log.info("Feedback table ready")
    except Exception as exc:
        log.error(f"Could not create feedback table: {exc}")


def ensure_predictions_table_exists():
    """
    Creates the predictions table if it doesn't already exist.
    Called once when the server starts.

    Why a separate predictions table from feedback?
        Not every prediction gets feedback — most users never click
        Accept or Reject. The predictions table captures ALL predictions
        so Viral can monitor the full distribution of what the model is
        predicting in production, not just the subset that got feedback.
    """
    sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id                  SERIAL PRIMARY KEY,
            file_id             TEXT        NOT NULL,
            user_id             TEXT        NOT NULL,
            predicted_tag       TEXT        NOT NULL,
            confidence          FLOAT       NOT NULL,
            action              TEXT        NOT NULL,
            category_type       TEXT,
            model_version       TEXT        NOT NULL,
            extraction_method   TEXT,
            latency_ms          FLOAT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Index on created_at for Viral's drift monitoring time-window queries
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions (created_at);

        -- Index on predicted_tag for label distribution queries
        CREATE INDEX IF NOT EXISTS idx_predictions_tag
            ON predictions (predicted_tag);

        -- Index on model_version so we can compare distributions across versions
        CREATE INDEX IF NOT EXISTS idx_predictions_model_version
            ON predictions (model_version);
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        conn.close()
        log.info("Predictions table ready")
    except Exception as exc:
        log.error(f"Could not create predictions table: {exc}")


# ---------------------------------------------------------------------------
# Write predictions (called on every /predict request)
# ---------------------------------------------------------------------------

def log_prediction(
    file_id:            str,
    user_id:            str,
    predicted_tag:      str,
    confidence:         float,
    action:             str,
    model_version:      str,
    category_type:      str | None = None,
    extraction_method:  str | None = None,
    latency_ms:         float | None = None,
) -> bool:
    """
    Logs every prediction to PostgreSQL.

    Called by main.py on every /predict request regardless of whether
    the user provides feedback. This is what Viral reads for drift monitoring.

    Args:
        file_id           : Nextcloud file node ID
        user_id           : Nextcloud user ID
        predicted_tag     : what the model predicted (e.g. "Problem Set")
        confidence        : model confidence score (0.0 to 1.0)
        action            : "auto_tag", "suggest", or "no_tag"
        model_version     : which model version made this prediction
        category_type     : "fixed_baseline" or "custom"
        extraction_method : "pdfminer", "ocr", or "plaintext"
        latency_ms        : total prediction latency in milliseconds

    Returns:
        True if saved successfully, False if database write failed.
        Fail-open — logging failure never crashes the prediction endpoint.
    """
    sql = """
        INSERT INTO predictions (
            file_id, user_id, predicted_tag, confidence,
            action, category_type, model_version,
            extraction_method, latency_ms
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        file_id,
        user_id,
        predicted_tag,
        confidence,
        action,
        category_type,
        model_version,
        extraction_method,
        latency_ms,
    )

    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        conn.close()
        return True

    except Exception as exc:
        # Never crash the prediction endpoint because of logging failure
        log.warning(f"Failed to log prediction to DB (non-fatal): {exc}")
        return False


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
    """
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


# ---------------------------------------------------------------------------
# Read predictions (for Viral's drift monitoring)
# ---------------------------------------------------------------------------

def get_recent_predictions(hours: int = 1) -> list[dict]:
    """
    Returns all predictions from the last N hours.
    Viral's drift monitoring script calls this to compute label distribution.

    Args:
        hours : how many hours back to look (default: 1)

    Returns:
        List of dicts with predicted_tag, confidence, model_version, created_at.
    """
    sql = """
        SELECT predicted_tag, confidence, model_version, created_at
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '%s hours'
        ORDER BY created_at DESC
    """
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (hours,))
            rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    except Exception as exc:
        log.error(f"Failed to fetch recent predictions: {exc}")
        return []