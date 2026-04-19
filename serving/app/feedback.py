"""
feedback.py — Feedback Storage, Prediction Logging, and Model Status
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

Three responsibilities:
    1. Save user feedback (accept/reject/correct) to PostgreSQL
    2. Log every prediction to PostgreSQL for drift monitoring
    3. Store rollback state in PostgreSQL so all Gunicorn workers share it

Why store rollback state in PostgreSQL?
    Gunicorn runs 4 worker processes. In-memory flags are per-process —
    if worker A sets _rolled_back=True, workers B/C/D still think it's False.
    PostgreSQL is shared across all workers so the state is consistent.

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
    ACCEPTED   = "accepted"
    REJECTED   = "rejected"
    CORRECTED  = "corrected"


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_connection():
    db_url = get_db_url()
    if db_url is None:
        raise RuntimeError(
            "DB_URL environment variable is not set. "
            "Add it to your .env file: DB_URL=postgresql://user:pass@host:5432/dbname"
        )
    return psycopg2.connect(db_url)


# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------

def ensure_feedback_table_exists():
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
        CREATE INDEX IF NOT EXISTS idx_feedback_user_id
            ON feedback (user_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_created_at
            ON feedback (created_at);
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
    Every /predict call is logged here for Viral's drift monitoring.
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
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions (created_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_tag
            ON predictions (predicted_tag);
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


def ensure_model_status_table_exists():
    """
    Creates the model_status table and seeds initial values.

    This table stores the rollback state shared across all Gunicorn workers.
    Schema: key-value store with two rows:
        rolled_back    : "true" or "false"
        rollback_reason: human-readable reason string
    """
    sql = """
        CREATE TABLE IF NOT EXISTS model_status (
            id          SERIAL PRIMARY KEY,
            key         TEXT UNIQUE NOT NULL,
            value       TEXT NOT NULL,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        );
        INSERT INTO model_status (key, value)
        VALUES ('rolled_back', 'false'), ('rollback_reason', '')
        ON CONFLICT (key) DO NOTHING;
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        conn.close()
        log.info("Model status table ready")
    except Exception as exc:
        log.error(f"Could not create model_status table: {exc}")


# ---------------------------------------------------------------------------
# Model status (rollback state) — shared across all workers via PostgreSQL
# ---------------------------------------------------------------------------

def get_model_status(key: str) -> str:
    """
    Reads a model status value from PostgreSQL.
    Called on every /predict request to check rollback state.

    Why not cache this?
        Caching would break the multi-worker consistency guarantee.
        The DB query adds ~1ms per request which is acceptable given
        our latency budget of 50-500ms per prediction.
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM model_status WHERE key = %s",
                (key,)
            )
            row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as exc:
        log.error(f"Could not get model status for key={key}: {exc}")
        return ""


def set_model_status(key: str, value: str):
    """
    Writes a model status value to PostgreSQL.
    Called by /admin/rollback and /admin/restore endpoints.
    Uses INSERT ... ON CONFLICT to upsert safely.
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_status (key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = NOW()
                """, (key, value))
        conn.close()
    except Exception as exc:
        log.error(f"Could not set model status key={key}: {exc}")


# ---------------------------------------------------------------------------
# Write predictions
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
    Logs every prediction to PostgreSQL for Viral's drift monitoring.
    Fail-open — never crashes the prediction endpoint.
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
        file_id, user_id, predicted_tag, confidence,
        action, category_type, model_version,
        extraction_method, latency_ms,
    )
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        conn.close()
        return True
    except Exception as exc:
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
    """Saves one feedback event to PostgreSQL."""
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
        file_id, user_id, predicted_tag, confidence,
        action_taken, feedback_type.value, corrected_tag,
        model_version, extraction_method,
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
# Read feedback
# ---------------------------------------------------------------------------

def get_feedback_for_user(user_id: str, since_days: int = 30) -> list[dict]:
    """Returns all feedback events for a user from the last N days."""
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
    """Returns all predictions from the last N hours."""
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