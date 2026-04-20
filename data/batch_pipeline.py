commit f93b9a5d318a7d779fe99b8dc285a39d242307e9
Merge: 29918ab12 bd76b64af
Author: Viral0401 <Viraldalal04@gmail.com>
Date:   Mon Apr 20 16:15:19 2026 -0400

    WIP on version-1/main: 29918ab12 extract text from uplaoded pdf

diff --cc data/batch_pipeline.py
index effa368c8,effa368c8..fc8eeda02
--- a/data/batch_pipeline.py
+++ b/data/batch_pipeline.py
@@@ -8,6 -8,6 +8,7 @@@ selection and course-level splitting fo
  """
  
  import argparse
++import hashlib
  import json
  import os
  import sys
@@@ -68,20 -68,20 +69,42 @@@ def load_feedback_from_postgres(db_url
          return None
      try:
          conn = psycopg2.connect(db_url)
++        # Current serving schema (feedback_type/action_taken/corrected_tag).
          query = """
              SELECT
--                id::text            AS event_id,
--                created_at          AS timestamp,
--                doc_id,
--                filename,
--                predicted_label,
--                user_action,
--                user_label,
--                'postgres'          AS source
++                id::text                                            AS event_id,
++                created_at                                          AS timestamp,
++                file_id                                             AS doc_id,
++                file_id                                             AS filename,
++                predicted_tag                                       AS predicted_label,
++                CASE
++                    WHEN feedback_type = 'accepted'  THEN 'accept'
++                    WHEN feedback_type = 'corrected' THEN 'correct'
++                    ELSE 'reject'
++                END                                                 AS user_action,
++                corrected_tag                                       AS user_label,
++                'postgres'                                          AS source
              FROM feedback
              ORDER BY created_at ASC
          """
--        df = pd.read_sql(query, conn)
++        try:
++            df = pd.read_sql(query, conn)
++        except Exception:
++            # Legacy schema fallback.
++            legacy_query = """
++                SELECT
++                    id::text            AS event_id,
++                    created_at          AS timestamp,
++                    doc_id,
++                    filename,
++                    predicted_label,
++                    user_action,
++                    user_label,
++                    'postgres'          AS source
++                FROM feedback
++                ORDER BY created_at ASC
++            """
++            df = pd.read_sql(legacy_query, conn)
          conn.close()
          print(f"[INFO] Loaded feedback from PostgreSQL: {len(df)} events")
          return df if len(df) > 0 else None
@@@ -90,13 -90,13 +113,160 @@@
          return None
  
  
++def load_corrected_feedback_training_rows(
++    db_url: str,
++    template_columns: list[str],
++) -> pd.DataFrame:
++    """
++    Build additional training rows from human corrections in PostgreSQL.
++
++    Source:
++      - feedback.feedback_type='corrected'  -> corrected_tag (new label)
++      - predictions.extracted_text          -> training text
++      - join key: file_id
++    """
++    if not _HAS_PSYCOPG2:
++        print("[WARN] psycopg2 not installed — skipping corrected feedback row generation")
++        return pd.DataFrame(columns=template_columns)
++
++    try:
++        conn = psycopg2.connect(db_url)
++        query = """
++            WITH latest_corrected AS (
++                SELECT DISTINCT ON (f.file_id)
++                    f.file_id,
++                    f.corrected_tag,
++                    f.created_at AS feedback_ts
++                FROM feedback f
++                WHERE f.feedback_type = 'corrected'
++                  AND f.corrected_tag IS NOT NULL
++                  AND btrim(f.corrected_tag) <> ''
++                ORDER BY f.file_id, f.created_at DESC
++            ),
++            latest_predictions AS (
++                SELECT DISTINCT ON (p.file_id)
++                    p.file_id,
++                    p.extracted_text,
++                    p.created_at AS prediction_ts
++                FROM predictions p
++                WHERE p.extracted_text IS NOT NULL
++                  AND btrim(p.extracted_text) <> ''
++                ORDER BY p.file_id, p.created_at DESC
++            )
++            SELECT
++                c.file_id,
++                c.corrected_tag,
++                p.extracted_text,
++                c.feedback_ts
++            FROM latest_corrected c
++            JOIN latest_predictions p
++              ON p.file_id = c.file_id
++            ORDER BY c.feedback_ts DESC
++        """
++        corrected = pd.read_sql(query, conn)
++        conn.close()
++    except Exception as exc:
++        print(f"[WARN] Could not build corrected feedback rows from PostgreSQL: {exc}")
++        return pd.DataFrame(columns=template_columns)
++
++    if corrected.empty:
++        print("[INFO] No corrected feedback rows found to append to train split")
++        return pd.DataFrame(columns=template_columns)
++
++    corrected["corrected_tag"] = corrected["corrected_tag"].astype(str).str.strip()
++
++    rows: list[dict] = []
++    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
++
++    for _, rec in corrected.iterrows():
++        file_id = str(rec["file_id"])
++        label = str(rec["corrected_tag"]).strip()
++        text = str(rec["extracted_text"] or "").strip()
++        if file_id == "" or label == "" or text == "":
++            continue
++
++        # Stable doc id for this feedback-derived training sample.
++        doc_id = hashlib.md5(f"feedback::{file_id}::{label}".encode("utf-8")).hexdigest()[:12]
++
++        row = {col: None for col in template_columns}
++        row["doc_id"] = doc_id
++        row["extracted_text"] = text
++        row["label"] = label
++        row["label_source"] = "user_feedback_corrected"
++        row["course_id"] = "user_feedback"
++        row["source_url"] = f"feedback://{file_id}"
++        row["source"] = "user_feedback"
++        row["ingestion_timestamp"] = now_iso
++        row["dataset_version"] = "feedback_loop"
++
++        if "filename" in row:
++            row["filename"] = f"feedback_{file_id}.txt"
++        if "char_count" in row:
++            row["char_count"] = len(text)
++        if "word_count" in row:
++            row["word_count"] = len(text.split())
++        if "file_size_bytes" in row:
++            row["file_size_bytes"] = len(text.encode("utf-8"))
++        if "text_extraction_method" in row:
++            row["text_extraction_method"] = "serving_prediction"
++        if "department" in row:
++            row["department"] = "User Feedback"
++        if "course_title" in row:
++            row["course_title"] = "User Feedback"
++        if "semester" in row:
++            row["semester"] = "N/A"
++        if "instructor" in row:
++            row["instructor"] = "N/A"
++
++        rows.append(row)
++
++    if not rows:
++        print("[INFO] No usable corrected feedback rows after filtering")
++        return pd.DataFrame(columns=template_columns)
++
++    out = pd.DataFrame(rows, columns=template_columns)
++    if "doc_id" in out.columns:
++        out = out.drop_duplicates(subset=["doc_id"], keep="last").reset_index(drop=True)
++
++    print(
++        "[INFO] Built "
++        f"{len(out)} corrected-feedback training rows "
++        "(feedback.corrected_tag + predictions.extracted_text)"
++    )
++    return out
++
++
++def append_corrected_feedback_to_train(train_df: pd.DataFrame, db_url: str) -> pd.DataFrame:
++    """
++    Append corrected-feedback-derived rows to the train split before write.
++    """
++    extra = load_corrected_feedback_training_rows(db_url, list(train_df.columns))
++    if extra.empty:
++        return train_df
++
++    # Avoid duplicate inserts across reruns while preserving all existing train rows.
++    # We only filter against existing train doc_ids; we do NOT dedupe the merged train set.
++    if "doc_id" in train_df.columns and "doc_id" in extra.columns:
++        extra = extra[~extra["doc_id"].isin(set(train_df["doc_id"]))].reset_index(drop=True)
++        if extra.empty:
++            print("[INFO] Corrected-feedback rows already present in train split; nothing to append")
++            return train_df
++
++    before = len(train_df)
++    merged = pd.concat([train_df, extra], ignore_index=True)
++
++    print(
++        f"[INFO] Train rows after corrected feedback append: {before} -> {len(merged)} "
++        f"(added {len(merged) - before})"
++    )
++    return merged
++
++
  # ---------------------------------------------------------------------------
  # Candidate selection
  # ---------------------------------------------------------------------------
  
  def apply_candidate_selection(df: pd.DataFrame, feedback: pd.DataFrame | None) -> pd.DataFrame:
--    before = len(df)
--
      # Drop short texts (likely extraction failures)
      before2 = len(df)
      df = df[df["char_count"] >= 200].copy()
@@@ -121,7 -121,7 +291,6 @@@
          # Keep only docs that appeared in feedback (accept or correct)
          feedback_doc_ids = set(engaged["doc_id"])
          in_feedback = df["doc_id"].isin(feedback_doc_ids)
--        not_in_feedback = ~in_feedback
          # Docs with accepted/corrected feedback are high-confidence; keep all others too
          # (feedback is a subset of docs — we keep everything that passed quality filters)
          print(f"[INFO] {in_feedback.sum()} docs have accepted/corrected feedback signal")
@@@ -342,6 -342,6 +511,12 @@@ def main() -> None
  
      train_df, eval_df = course_level_split(df, args.eval_ratio, args.seed)
  
++    # Human-in-the-loop closure:
++    # Add corrected feedback rows (label=corrected_tag, text from predictions.extracted_text)
++    # to the training split before writing train.parquet.
++    if args.feedback_source == "postgres":
++        train_df = append_corrected_feedback_to_train(train_df, args.db_url)
++
      validate_training_set(train_df, eval_df)
  
      out = write_outputs(train_df, eval_df, args.output_dir, args.version)
