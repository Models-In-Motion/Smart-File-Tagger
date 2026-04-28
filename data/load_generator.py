#!/usr/bin/env python3
"""
load_generator.py

Simulates continuous production traffic against the real serving layer.
Runs indefinitely, uploading documents and submitting feedback.

Usage:
    python load_generator.py
"""

import os
import random
import time
import uuid
import requests
import pandas as pd

PREDICT_URL  = os.getenv("PREDICT_URL",  "http://serving:8000/predict-text")
FEEDBACK_URL = os.getenv("FEEDBACK_URL", "http://serving:8000/feedback")
DATA_PATH    = os.getenv("DATA_PATH",    "/app/artifacts/ocw_dataset.parquet")
DELAY_SECS   = float(os.getenv("DELAY_SECS", "30"))
USER_ID      = os.getenv("USER_ID", "load-generator")

def load_data():
    df = pd.read_parquet(DATA_PATH)
    # Keep only rows with valid labels and non-empty text
    valid = ["Lecture Notes", "Problem Set", "Exam", "Reading", "Other"]
    df = df[df["label"].isin(valid)].dropna(subset=["extracted_text"])
    print(f"[load_generator] Loaded {len(df)} documents", flush=True)
    return df.to_dict("records")

def predict(text: str, user_id: str, file_id: str) -> dict:
    try:
        resp = requests.post(
            PREDICT_URL,
            json={"text": text[:1000], "user_id": user_id, "file_id": file_id},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[load_generator] Predict failed: {e}", flush=True)
        return {}

def submit_feedback(file_id: str, user_id: str, predicted_tag: str,
                    confidence: float, feedback_type: str,
                    corrected_tag: str | None, model_version: str):
    try:
        resp = requests.post(
            FEEDBACK_URL,
            json={
                "file_id":           file_id,
                "user_id":           user_id,
                "predicted_tag":     predicted_tag,
                "confidence":        confidence,
                "action_taken":      feedback_type,
                "feedback_type":     feedback_type,
                "corrected_tag":     corrected_tag,
                "model_version":     model_version,
                "extraction_method": "load_generator",
            },
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[load_generator] Feedback failed: {e}", flush=True)

def simulate_feedback(predicted: str, real: str, rng: random.Random):
    """Simulate realistic user feedback behavior."""
    if predicted == real:
        # Correct prediction: accept 80%, ignore 20%
        action = rng.choices(["accepted", "ignored"], weights=[80, 20])[0]
        return action, None
    else:
        # Wrong prediction: correct 25%, ignore 75%
        # Most users don't bother correcting wrong tags
        action = rng.choices(["corrected", "ignored"], weights=[25, 75])[0]
        corrected = real if action == "corrected" else None
        return action, corrected

def main():
    print("[load_generator] Starting — waiting for serving layer...", flush=True)
    # Wait for serving to be ready
    for _ in range(30):
        try:
            r = requests.get("http://serving:8000/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(10)

    print("[load_generator] Serving ready — loading dataset...", flush=True)
    records = load_data()
    rng = random.Random()
    event_count = 0

    while True:
        doc = rng.choice(records)
        text = str(doc.get("extracted_text", ""))[:1000]
        real_label = str(doc.get("label", "Other"))
        file_id = f"gen_{uuid.uuid4().hex[:12]}"

        result = predict(text, USER_ID, file_id)
        if not result:
            time.sleep(DELAY_SECS)
            continue

        predicted_tag  = result.get("predicted_tag", "Other")
        confidence     = float(result.get("confidence", 0.0))
        action         = result.get("action", "no_tag")
        model_version  = result.get("model_version", "unknown")

        # Simulate feedback for all actions except no_tag with very low confidence
        if action in ("auto_apply", "suggest") or (action == "no_tag" and confidence > 0.2):
            feedback_type, corrected_tag = simulate_feedback(
                predicted_tag, real_label, rng
            )
            if feedback_type != "ignored":
                submit_feedback(
                    file_id, USER_ID, predicted_tag,
                    confidence, feedback_type, corrected_tag,
                    model_version
                )
                event_count += 1
                print(
                    f"[load_generator] event={event_count} "
                    f"predicted={predicted_tag} real={real_label} "
                    f"feedback={feedback_type} corrected={corrected_tag}",
                    flush=True
                )

        time.sleep(DELAY_SECS)

if __name__ == "__main__":
    main()
