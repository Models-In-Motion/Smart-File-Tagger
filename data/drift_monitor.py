#!/usr/bin/env python3
"""
drift_monitor.py  —  Checkpoint 3

Reads production data from PostgreSQL (predictions + feedback tables written
by Krish's serving layer), compares against training baseline, and writes
drift metrics back to the drift_metrics table so Krish's monitor.py can
trigger rollback automatically.

Also writes a drift_report.json for audit/Grafana.

Table this script owns:
  drift_metrics  (created automatically on first run)

Tables this script reads (owned by serving):
  predictions  — every /predict call (predicted_tag, confidence, model_version, char_count, ts)
  feedback     — every /feedback call (user_action, user_label, ts)

Exit codes:
  0  — no drift (or warn-only)
  1  — hard drift threshold exceeded (Krish's monitor.py will trigger rollback)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.spatial.distance import jensenshannon
import numpy as np

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


VALID_LABELS = {
    "Lecture Notes", "Problem Set", "Exam",
    "Reading", "Solution", "Project", "Other",
}

# Thresholds
JS_WARN_THRESHOLD  = 0.10
JS_HARD_THRESHOLD  = 0.20
LEN_WARN_RATIO     = 1.5
LEN_HARD_RATIO     = 2.0
CORRECTION_WARN    = 0.15
CORRECTION_HARD    = 0.30
UNKNOWN_LABEL_WARN = 0.01
UNKNOWN_LABEL_HARD = 0.05
MIN_PRODUCTION_ROWS = 20


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------

DDL_DRIFT_METRICS = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    id                        SERIAL PRIMARY KEY,
    timestamp                 TIMESTAMPTZ DEFAULT NOW(),
    js_divergence             FLOAT,
    correction_rate           FLOAT,
    mean_char_count_baseline  FLOAT,
    mean_char_count_production FLOAT,
    text_length_drift_pct     FLOAT,
    unknown_labels            TEXT,
    status                    VARCHAR(20),
    details                   JSONB
);
"""


def get_connection(db_url: str):
    if not HAS_PSYCOPG2:
        raise RuntimeError("psycopg2 not installed — cannot connect to PostgreSQL")
    return psycopg2.connect(db_url)


def ensure_drift_metrics_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_DRIFT_METRICS)
    conn.commit()


def _read_sql(query: str, conn) -> pd.DataFrame:
    """Read SQL via cursor to avoid pandas SQLAlchemy warnings with psycopg2."""
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def load_predictions_from_db(conn) -> pd.DataFrame:
    """Read all rows from the predictions table (owned by Krish's serving layer)."""
    query = """
        SELECT predicted_tag, confidence, model_version,
               LENGTH(extracted_text) AS char_count,
               created_at AS ts
        FROM   predictions
        ORDER  BY created_at DESC
    """
    try:
        df = _read_sql(query, conn)
        print(f"[INFO] Predictions from DB: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"[WARN] Could not read predictions table: {exc} — returning empty")
        return pd.DataFrame()


def load_feedback_from_db(conn) -> pd.DataFrame:
    """Read all rows from the feedback table (owned by Krish's serving layer).

    Column mapping:
      feedback_type -> user_action
      corrected_tag -> user_label
    """
    query = """
        SELECT feedback_type AS user_action,
               corrected_tag AS user_label,
               created_at    AS ts
        FROM   feedback
        ORDER  BY created_at DESC
    """
    try:
        df = _read_sql(query, conn)
        print(f"[INFO] Feedback from DB: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"[WARN] Could not read feedback table: {exc} — returning empty")
        return pd.DataFrame()


def write_drift_metrics_to_db(conn, report: dict) -> None:
    """Insert one row into drift_metrics so Krish's monitor.py can read it."""
    ld = report.get("label_drift", {})
    cr = report.get("correction_rate", {})
    ul = report.get("unknown_labels", {})
    tl = report.get("text_length_drift", {})

    baseline_mean = tl.get("baseline_mean_char_count")
    prod_mean     = tl.get("production_mean_char_count")
    drift_pct     = None
    if baseline_mean and prod_mean and baseline_mean > 0:
        drift_pct = round((prod_mean - baseline_mean) / baseline_mean * 100, 2)

    unknown_vals = ul.get("unknown_label_values", [])

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_metrics
                (js_divergence, correction_rate,
                 mean_char_count_baseline, mean_char_count_production,
                 text_length_drift_pct, unknown_labels,
                 status, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                ld.get("js_divergence"),
                cr.get("correction_rate"),
                baseline_mean,
                prod_mean,
                drift_pct,
                json.dumps(unknown_vals),
                report["status"],
                json.dumps(report),
            ),
        )
    conn.commit()
    print("[INFO] Drift metrics written to drift_metrics table")


# ---------------------------------------------------------------------------
# Fallback: load feedback from JSONL (for local testing without Postgres)
# ---------------------------------------------------------------------------

def load_feedback_from_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] No feedback file at {path} — returning empty")
        return pd.DataFrame()
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"[INFO] Feedback from JSONL: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------

def load_baseline(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[INFO] Baseline: {len(df)} rows from {path}")
    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def label_distribution_vector(series: pd.Series, labels: list[str]) -> np.ndarray:
    counts = series.value_counts()
    vec = np.array([counts.get(lbl, 0) for lbl in labels], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(jensenshannon(p, q, base=2) ** 2)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_label_drift(
    baseline_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> dict:
    labels = sorted(VALID_LABELS)
    baseline_vec = label_distribution_vector(baseline_df["label"], labels)

    pred_col = None
    for col in ("predicted_tag", "predicted_label", "model_label", "label"):
        if col in predictions_df.columns:
            pred_col = col
            break

    if pred_col is None:
        warnings.append("Predictions table has no label column — skipping label drift check")
        return {}

    prod_vec = label_distribution_vector(predictions_df[pred_col], labels)
    jsd = js_divergence(baseline_vec, prod_vec)

    result = {
        "js_divergence": round(jsd, 4),
        "baseline_distribution": {lbl: round(float(v), 4) for lbl, v in zip(labels, baseline_vec)},
        "production_distribution": {lbl: round(float(v), 4) for lbl, v in zip(labels, prod_vec)},
    }

    if jsd >= JS_HARD_THRESHOLD:
        errors.append(
            f"Label distribution drift (JS={jsd:.3f}) exceeds hard threshold {JS_HARD_THRESHOLD}"
        )
    elif jsd >= JS_WARN_THRESHOLD:
        warnings.append(
            f"Label distribution drift (JS={jsd:.3f}) exceeds warn threshold {JS_WARN_THRESHOLD}"
        )

    return result


def check_text_length_drift(
    baseline_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> dict:
    baseline_mean = (
        float(baseline_df["char_count"].mean())
        if "char_count" in baseline_df.columns else None
    )

    prod_mean = None
    for col in ("char_count", "text_length"):
        if col in predictions_df.columns:
            prod_mean = float(predictions_df[col].mean())
            break

    if baseline_mean is None or prod_mean is None:
        warnings.append("Cannot compute text length drift — missing char_count column")
        return {}

    ratio = baseline_mean / prod_mean if prod_mean > 0 else float("inf")
    result = {
        "baseline_mean_char_count": round(baseline_mean, 1),
        "production_mean_char_count": round(prod_mean, 1),
        "ratio_baseline_over_production": round(ratio, 3),
    }

    # Serving stores first 10k chars in predictions.extracted_text. If production
    # text is much shorter than baseline (ratio >10), treat as expected snippet behavior.
    if ratio > 10 or ratio < 0.1:
        warnings.append(
            f"Text length ratio {ratio:.1f} suggests production stores text snippets "
            f"(baseline mean={baseline_mean:.0f} chars, production mean={prod_mean:.0f} chars). "
            "Text length drift check skipped — not comparable."
        )
    elif ratio >= LEN_HARD_RATIO or ratio <= 1 / LEN_HARD_RATIO:
        errors.append(
            f"Text length drift: ratio {ratio:.2f} exceeds hard threshold {LEN_HARD_RATIO}"
        )
    elif ratio >= LEN_WARN_RATIO or ratio <= 1 / LEN_WARN_RATIO:
        warnings.append(
            f"Text length drift: ratio {ratio:.2f} exceeds warn threshold {LEN_WARN_RATIO}"
        )

    return result


def check_correction_rate(
    feedback_df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> dict:
    if "user_action" not in feedback_df.columns:
        warnings.append("Feedback missing 'user_action' column — skipping correction rate check")
        return {}

    n_total = len(feedback_df)
    n_correct = int((feedback_df["user_action"] == "corrected").sum())
    rate = n_correct / n_total if n_total > 0 else 0.0

    result = {
        "total_feedback_events": n_total,
        "correction_events": n_correct,
        "correction_rate": round(rate, 4),
    }

    if rate >= CORRECTION_HARD:
        errors.append(
            f"Correction rate {rate:.1%} exceeds hard threshold {CORRECTION_HARD:.0%} "
            f"— model predictions may be systematically wrong"
        )
    elif rate >= CORRECTION_WARN:
        warnings.append(
            f"Correction rate {rate:.1%} exceeds warn threshold {CORRECTION_WARN:.0%}"
        )

    return result


def check_unknown_labels(
    predictions_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> dict:
    # Check model outputs, not user-corrected labels. Users may define custom categories.
    df = predictions_df
    label_col = None
    for col in ("predicted_tag", "predicted_label", "label"):
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        return {}

    present = df[label_col].dropna()
    unknown_mask = ~present.isin(VALID_LABELS)
    n_unknown = int(unknown_mask.sum())
    rate = n_unknown / len(present) if len(present) > 0 else 0.0

    result = {
        "label_column_checked": label_col,
        "unknown_label_count": n_unknown,
        "unknown_label_rate": round(rate, 4),
        "unknown_label_values": sorted(present[unknown_mask].unique().tolist()),
    }

    if rate >= UNKNOWN_LABEL_HARD:
        errors.append(
            f"Unknown labels: {rate:.1%} of predicted_tag values are out-of-vocabulary"
        )
    elif rate >= UNKNOWN_LABEL_WARN:
        warnings.append(
            f"Unknown labels: {rate:.1%} of predicted_tag values (warn threshold {UNKNOWN_LABEL_WARN:.0%})"
        )

    return result


def check_temporal_trend(
    feedback_df: pd.DataFrame,
    warnings: list[str],
) -> dict:
    ts_col = None
    for col in ("ts", "timestamp", "created_at", "event_timestamp"):
        if col in feedback_df.columns:
            ts_col = col
            break

    if ts_col is None or "user_action" not in feedback_df.columns:
        return {}

    try:
        fb = feedback_df.copy()
        fb["_date"] = pd.to_datetime(fb[ts_col], utc=True).dt.date
        daily = (
            fb.groupby("_date")["user_action"]
            .apply(lambda s: round((s == "corrected").mean(), 4))
            .reset_index()
            .rename(columns={"user_action": "correction_rate"})
        )
        return {"daily_correction_rate": {str(r["_date"]): r["correction_rate"] for _, r in daily.iterrows()}}
    except Exception as exc:
        warnings.append(f"Could not compute temporal trend: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_drift_monitor(
    baseline_path: str,
    output_path: str,
    version: str,
    db_url: str | None,
    feedback_jsonl: str | None,
) -> int:
    baseline_df = load_baseline(baseline_path)

    errors:   list[str] = []
    warnings: list[str] = []

    conn = None
    predictions_df = pd.DataFrame()
    feedback_df    = pd.DataFrame()

    if db_url and HAS_PSYCOPG2:
        try:
            conn = get_connection(db_url)
            ensure_drift_metrics_table(conn)
            predictions_df = load_predictions_from_db(conn)
            feedback_df    = load_feedback_from_db(conn)
        except Exception as exc:
            warnings.append(f"PostgreSQL unavailable ({exc}) — falling back to JSONL if provided")
            conn = None

    if conn is None and feedback_jsonl:
        feedback_df = load_feedback_from_jsonl(feedback_jsonl)
        # When reading from JSONL, treat feedback rows as predictions too
        # (data_generator.py writes predicted_label into JSONL)
        predictions_df = feedback_df

    production_rows = max(len(predictions_df), len(feedback_df))

    report: dict = {
        "version": version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_path": baseline_path,
        "baseline_rows": len(baseline_df),
        "prediction_rows": len(predictions_df),
        "feedback_rows": len(feedback_df),
    }

    if production_rows < MIN_PRODUCTION_ROWS:
        msg = (
            f"Only {production_rows} production rows — need at least {MIN_PRODUCTION_ROWS} "
            "for meaningful drift detection. Skipping checks."
        )
        print(f"[WARN] {msg}")
        warnings.append(msg)
        report.update({
            "label_drift": {}, "text_length_drift": {},
            "correction_rate": {}, "unknown_labels": {}, "temporal_trend": {},
        })
    else:
        report["label_drift"]       = check_label_drift(baseline_df, predictions_df, errors, warnings)
        report["text_length_drift"] = check_text_length_drift(baseline_df, predictions_df, errors, warnings)
        report["correction_rate"]   = check_correction_rate(feedback_df, errors, warnings)
        report["unknown_labels"]    = check_unknown_labels(predictions_df, feedback_df, errors, warnings)
        report["temporal_trend"]    = check_temporal_trend(feedback_df, warnings)

    report["warnings"] = warnings
    report["errors"]   = errors
    report["status"]   = "FAIL" if errors else "PASS"

    # Print summary
    print("\n=== Drift Monitor Report ===")
    ld = report.get("label_drift", {})
    if "js_divergence" in ld:
        print(f"JS Divergence  : {ld['js_divergence']:.4f}  (warn≥{JS_WARN_THRESHOLD}, hard≥{JS_HARD_THRESHOLD})")
    cr = report.get("correction_rate", {})
    if cr:
        print(f"Correction rate: {cr.get('correction_rate', 0):.1%}  "
              f"({cr.get('correction_events')} / {cr.get('total_feedback_events')} events)")
    tl = report.get("text_length_drift", {})
    if tl:
        print(f"Length ratio   : {tl.get('ratio_baseline_over_production', 'N/A')}")
    print(f"\nWarnings : {len(warnings)}")
    for w in warnings:
        print(f"  [WARN] {w}")
    print(f"Errors   : {len(errors)}")
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"Status   : {report['status']}")

    # Write JSON report to disk (audit trail)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out}")

    # Write metrics row to PostgreSQL so Krish's monitor.py can trigger rollback
    if conn is not None:
        try:
            write_drift_metrics_to_db(conn, report)
        except Exception as exc:
            print(f"[WARN] Could not write to drift_metrics table: {exc}")
        finally:
            conn.close()

    return 1 if errors else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drift monitor: reads predictions/feedback from PostgreSQL, "
                    "detects distribution drift, writes to drift_metrics table."
    )
    parser.add_argument("--baseline",  default="artifacts/versions/v2/train.parquet")
    parser.add_argument("--output",    default="artifacts/drift_report.json")
    parser.add_argument("--version",   default="v2")
    parser.add_argument("--db-url",    default=os.environ.get("DB_URL"),
                        help="PostgreSQL connection string (or set DB_URL env var)")
    parser.add_argument("--feedback",  default=None,
                        help="Fallback JSONL path when PostgreSQL is unavailable (local testing)")
    parser.add_argument("--interval",  type=int, default=0,
                        help="If >0, run in a loop every N seconds (cron mode)")
    args = parser.parse_args()

    if args.interval > 0:
        print(f"[INFO] Running in cron mode — checking every {args.interval}s")
        while True:
            print(f"\n[drift-monitor] Run at {datetime.now(timezone.utc).isoformat()}")
            try:
                run_drift_monitor(
                    baseline_path=args.baseline,
                    output_path=args.output,
                    version=args.version,
                    db_url=args.db_url,
                    feedback_jsonl=args.feedback,
                )
            except Exception as exc:
                print(f"[ERROR] Drift monitor run failed: {exc}")
            time.sleep(args.interval)
    else:
        exit_code = run_drift_monitor(
            baseline_path=args.baseline,
            output_path=args.output,
            version=args.version,
            db_url=args.db_url,
            feedback_jsonl=args.feedback,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
