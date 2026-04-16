#!/usr/bin/env python3
"""
data_generator.py

Simulates production traffic for the document auto-tagging system.

For each event:
  1. Picks a random document from the dataset
  2. POSTs it to POST /predict (the serving endpoint)
  3. Simulates user feedback based on the response (accept / correct / ignore)
  4. Logs the full event to a JSONL feedback file

If the HTTP call fails, falls back to in-memory simulation so the script
works standalone without the mock server running.

Usage:
    # With mock server (recommended for demo):
    docker compose up -d mock-predict
    docker compose run --rm data-generator

    # Standalone (no server needed):
    python data_generator.py --num-events 500
"""

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

VALID_LABELS = [
    "Lecture Notes", "Problem Set", "Exam", "Syllabus",
    "Reading", "Solution", "Project", "Recitation", "Lab", "Other",
]

TEXT_CHARS = 512   # chars sent to /predict


# ---------------------------------------------------------------------------
# Prediction: HTTP or fallback
# ---------------------------------------------------------------------------

def predict_via_http(doc: dict, predict_url: str) -> str | None:
    """POST document to /predict, return predicted_tag or None on failure."""
    if not _REQUESTS_AVAILABLE:
        return None
    payload = {
        "doc_id":         doc["doc_id"],
        "filename":       doc["filename"],
        "extracted_text": doc["extracted_text"][:TEXT_CHARS],
        "file_size_bytes": doc.get("file_size_bytes", 0),
        "real_label":     doc["label"],   # hint so mock gives realistic accuracy
    }
    try:
        resp = _requests.post(predict_url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json().get("predicted_tag")
    except Exception:
        return None


def predict_fallback(real_label: str, rng: random.Random) -> str:
    """In-memory fallback: model right ~70% of the time."""
    if rng.random() < 0.70:
        return real_label
    wrong = [l for l in VALID_LABELS if l != real_label]
    return rng.choice(wrong)


# ---------------------------------------------------------------------------
# Feedback simulation
# ---------------------------------------------------------------------------

def simulate_feedback(predicted: str, real_label: str, rng: random.Random) -> tuple[str, str | None]:
    """
    Returns (user_action, user_label).
    - Correct prediction: accept 90%, ignore 10%
    - Wrong prediction:   correct 60%, ignore 40%
    """
    if predicted == real_label:
        action = rng.choices(["accept", "ignore"], weights=[90, 10])[0]
        user_label = None
    else:
        action = rng.choices(["correct", "ignore"], weights=[60, 40])[0]
        user_label = real_label if action == "correct" else None
    return action, user_label


def random_timestamp(rng: random.Random, days_back: int = 30) -> str:
    now = datetime.now(timezone.utc)
    offset = timedelta(seconds=rng.randint(0, days_back * 24 * 3600))
    return (now - offset).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate(
    df: pd.DataFrame,
    num_events: int,
    rng: random.Random,
    predict_url: str | None,
    delay: float,
) -> list[dict]:
    records = df.to_dict("records")
    events = []
    http_ok = 0
    http_fail = 0

    for i in range(num_events):
        doc = rng.choice(records)
        real_label = doc["label"]

        # Try HTTP first, fall back to in-memory
        predicted = None
        if predict_url:
            predicted = predict_via_http(doc, predict_url)
            if predicted:
                http_ok += 1
            else:
                http_fail += 1

        if predicted is None:
            predicted = predict_fallback(real_label, rng)

        action, user_label = simulate_feedback(predicted, real_label, rng)

        events.append({
            "event_id":       str(uuid.UUID(int=rng.getrandbits(128))),
            "timestamp":      random_timestamp(rng),
            "doc_id":         doc["doc_id"],
            "filename":       doc["filename"],
            "predicted_label": predicted,
            "user_action":    action,
            "user_label":     user_label,
            "source":         "user_feedback",
        })

        if (i + 1) % 50 == 0:
            mode = "http" if predict_url and http_ok > 0 else "fallback"
            print(f"[INFO] {i+1}/{num_events} events generated  (mode: {mode})")

        if delay > 0:
            time.sleep(delay)

    if predict_url:
        print(f"[INFO] HTTP calls: {http_ok} succeeded, {http_fail} failed (fallback used)")

    return events


def print_summary(events: list[dict]) -> None:
    action_counts = {"accept": 0, "correct": 0, "ignore": 0}
    accepted_correct = 0

    for e in events:
        action_counts[e["user_action"]] = action_counts.get(e["user_action"], 0) + 1
        if e["user_action"] == "accept":
            accepted_correct += 1

    engaged = action_counts["accept"] + action_counts["correct"]
    accuracy = accepted_correct / engaged if engaged > 0 else 0.0

    print(f"\n=== Data Generator Summary ===")
    print(f"Total events:       {len(events)}")
    print(f"  accept:           {action_counts['accept']}")
    print(f"  correct:          {action_counts['correct']}")
    print(f"  ignore:           {action_counts['ignore']}")
    print(f"Simulated accuracy: {accuracy:.1%}  (accepted / engaged)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic production feedback by hitting /predict.")
    parser.add_argument("--input", default="artifacts/ocw_dataset.parquet")
    parser.add_argument("--output", default="artifacts/production_feedback.jsonl")
    parser.add_argument("--num-events", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-url", default=None,
                        help="URL of POST /predict endpoint (e.g. http://mock-predict:8000/predict)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between events (0 = as fast as possible)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df)} documents from {args.input}")

    if args.predict_url:
        print(f"[INFO] Will POST to {args.predict_url}")
    else:
        print(f"[INFO] No --predict-url given — using in-memory fallback simulation")

    rng = random.Random(args.seed)
    events = generate(df, args.num_events, rng, args.predict_url, args.delay)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    print(f"[INFO] Wrote {len(events)} events to {args.output}")
    print_summary(events)


if __name__ == "__main__":
    main()
