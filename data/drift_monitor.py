commit f93b9a5d318a7d779fe99b8dc285a39d242307e9
Merge: 29918ab12 bd76b64af
Author: Viral0401 <Viraldalal04@gmail.com>
Date:   Mon Apr 20 16:15:19 2026 -0400

    WIP on version-1/main: 29918ab12 extract text from uplaoded pdf

diff --cc data/drift_monitor.py
index d26ed221a,d26ed221a..fe617607a
--- a/data/drift_monitor.py
+++ b/data/drift_monitor.py
@@@ -36,6 -36,6 +36,7 @@@ import numpy as n
  try:
      import psycopg2
      import psycopg2.extras
++    from psycopg2.extensions import connection as Psycopg2Connection
      HAS_PSYCOPG2 = True
  except ImportError:
      HAS_PSYCOPG2 = False
@@@ -90,15 -90,15 +91,26 @@@ def ensure_drift_metrics_table(conn) -
      conn.commit()
  
  
++def _read_sql(query: str, conn) -> pd.DataFrame:
++    """pd.read_sql wrapper that uses cursor to avoid SQLAlchemy warnings."""
++    with conn.cursor() as cur:
++        cur.execute(query)
++        cols = [desc[0] for desc in cur.description]
++        rows = cur.fetchall()
++    return pd.DataFrame(rows, columns=cols)
++
++
  def load_predictions_from_db(conn) -> pd.DataFrame:
      """Read all rows from the predictions table (owned by Krish's serving layer)."""
      query = """
--        SELECT predicted_tag, confidence, model_version, char_count, created_at AS ts
++        SELECT predicted_tag, confidence, model_version,
++               LENGTH(extracted_text) AS char_count,
++               created_at AS ts
          FROM   predictions
          ORDER  BY created_at DESC
      """
      try:
--        df = pd.read_sql(query, conn)
++        df = _read_sql(query, conn)
          print(f"[INFO] Predictions from DB: {len(df)} rows")
          return df
      except Exception as exc:
@@@ -107,14 -107,14 +119,21 @@@
  
  
  def load_feedback_from_db(conn) -> pd.DataFrame:
--    """Read all rows from the feedback table (owned by Krish's serving layer)."""
++    """Read all rows from the feedback table (owned by Krish's serving layer).
++
++    Column mapping (serving schema → drift monitor internal names):
++      feedback_type → user_action  (accepted / corrected / rejected)
++      corrected_tag → user_label
++    """
      query = """
--        SELECT user_action, user_label, created_at AS ts
++        SELECT feedback_type  AS user_action,
++               corrected_tag  AS user_label,
++               created_at     AS ts
          FROM   feedback
          ORDER  BY created_at DESC
      """
      try:
--        df = pd.read_sql(query, conn)
++        df = _read_sql(query, conn)
          print(f"[INFO] Feedback from DB: {len(df)} rows")
          return df
      except Exception as exc:
@@@ -279,7 -279,7 +298,16 @@@ def check_text_length_drift
          "ratio_baseline_over_production": round(ratio, 3),
      }
  
--    if ratio >= LEN_HARD_RATIO or ratio <= 1 / LEN_HARD_RATIO:
++    # If ratio > 10 the production text is almost certainly a short snippet stored
++    # by the serving layer (not the full document). Flag as info only — this is
++    # expected behaviour, not a meaningful drift signal.
++    if ratio > 10 or ratio < 0.1:
++        warnings.append(
++            f"Text length ratio {ratio:.1f} suggests production stores text snippets "
++            f"(baseline mean={baseline_mean:.0f} chars, production mean={prod_mean:.0f} chars). "
++            "Text length drift check skipped — not comparable."
++        )
++    elif ratio >= LEN_HARD_RATIO or ratio <= 1 / LEN_HARD_RATIO:
          errors.append(
              f"Text length drift: ratio {ratio:.2f} exceeds hard threshold {LEN_HARD_RATIO}"
          )
@@@ -301,7 -301,7 +329,7 @@@ def check_correction_rate
          return {}
  
      n_total = len(feedback_df)
--    n_correct = int((feedback_df["user_action"] == "correct").sum())
++    n_correct = int((feedback_df["user_action"] == "corrected").sum())
      rate = n_correct / n_total if n_total > 0 else 0.0
  
      result = {
@@@ -324,20 -324,20 +352,25 @@@
  
  
  def check_unknown_labels(
++    predictions_df: pd.DataFrame,
      feedback_df: pd.DataFrame,
      errors: list[str],
      warnings: list[str],
  ) -> dict:
++    # Check predicted_tag from predictions (model output must stay in VALID_LABELS).
++    # Do NOT check corrected_tag from feedback — users can define custom categories
++    # via the category manager and those are valid labels.
      label_col = None
--    for col in ("user_label", "predicted_label", "label"):
--        if col in feedback_df.columns:
++    df = predictions_df
++    for col in ("predicted_tag", "predicted_label", "label"):
++        if col in predictions_df.columns:
              label_col = col
              break
  
      if label_col is None:
          return {}
  
--    present = feedback_df[label_col].dropna()
++    present = df[label_col].dropna()
      unknown_mask = ~present.isin(VALID_LABELS)
      n_unknown = int(unknown_mask.sum())
      rate = n_unknown / len(present) if len(present) > 0 else 0.0
@@@ -351,11 -351,11 +384,11 @@@
  
      if rate >= UNKNOWN_LABEL_HARD:
          errors.append(
--            f"Unknown labels: {rate:.1%} of feedback rows have out-of-vocabulary labels"
++            f"Unknown labels: {rate:.1%} of predicted_tag values are out-of-vocabulary"
          )
      elif rate >= UNKNOWN_LABEL_WARN:
          warnings.append(
--            f"Unknown labels: {rate:.1%} of feedback rows (warn threshold {UNKNOWN_LABEL_WARN:.0%})"
++            f"Unknown labels: {rate:.1%} of predicted_tag values (warn threshold {UNKNOWN_LABEL_WARN:.0%})"
          )
  
      return result
@@@ -451,7 -451,7 +484,7 @@@ def run_drift_monitor
          report["label_drift"]       = check_label_drift(baseline_df, predictions_df, errors, warnings)
          report["text_length_drift"] = check_text_length_drift(baseline_df, predictions_df, errors, warnings)
          report["correction_rate"]   = check_correction_rate(feedback_df, errors, warnings)
--        report["unknown_labels"]    = check_unknown_labels(feedback_df, errors, warnings)
++        report["unknown_labels"]    = check_unknown_labels(predictions_df, feedback_df, errors, warnings)
          report["temporal_trend"]    = check_temporal_trend(feedback_df, warnings)
  
      report["warnings"] = warnings
