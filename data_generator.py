#!/usr/bin/env python3
"""
data_generator.py

Simulates production traffic for the document auto-tagging system.
Reads the OCW dataset, generates synthetic user upload + feedback events,
and writes them to a JSONL feedback log.
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

VALID_LABELS = [
    "Lecture Notes", "Problem Set", "Exam", "Syllabus",
    "Reading", "Solution", "Project", "Recitation", "Lab", "Other",
]


def random_timestamp(rng: random.Random, days_back: int = 30) -> str:
    """Return an ISO 8601 timestamp spread uniformly over the past N days."""
    now = datetime.now(timezone.utc)
    offset_seconds = rng.randint(0, days_back * 24 * 3600)
    ts = now - timedelta(seconds=offset_seconds)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def simulate_prediction(real_label: str, rng: random.Random) -> str:
    """Model is right ~70% of the time, wrong ~30%."""
    if rng.random() < 0.70:
        return real_label
    wrong_labels = [l for l in VALID_LABELS if l != real_label]
    return rng.choice(wrong_labels)


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


def generate(df: pd.DataFrame, num_events: int, rng: random.Random) -> list[dict]:
    records = df.to_dict("records")
    events = []

    for _ in range(num_events):
        doc = rng.choice(records)
        real_label = doc["label"]
        predicted = simulate_prediction(real_label, rng)
        action, user_label = simulate_feedback(predicted, real_label, rng)

        events.append({
            "event_id": str(uuid.UUID(int=rng.getrandbits(128))),
            "timestamp": random_timestamp(rng),
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "predicted_label": predicted,
            "user_action": action,
            "user_label": user_label,
            "source": "user_feedback",
        })

    return events


def print_summary(events: list[dict]) -> None:
    total = len(events)
    action_counts = {"accept": 0, "correct": 0, "ignore": 0}
    correct_predictions = 0

    for e in events:
        action_counts[e["user_action"]] += 1
        # Prediction was correct if user accepted or ignored without correcting
        if e["user_action"] in ("accept", "ignore") and e["user_label"] is None:
            if e["user_action"] == "accept":
                correct_predictions += 1

    # Simulated accuracy: accepted / (accepted + corrected)
    engaged = action_counts["accept"] + action_counts["correct"]
    accuracy = correct_predictions / engaged if engaged > 0 else 0.0

    print(f"\n=== Data Generator Summary ===")
    print(f"Total events:       {total}")
    print(f"  accept:           {action_counts['accept']}")
    print(f"  correct:          {action_counts['correct']}")
    print(f"  ignore:           {action_counts['ignore']}")
    print(f"Simulated accuracy: {accuracy:.1%}  (accepted / engaged)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic production feedback events.")
    parser.add_argument("--input", default="artifacts/ocw_dataset.parquet",
                        help="Path to input Parquet dataset")
    parser.add_argument("--output", default="artifacts/production_feedback.jsonl",
                        help="Path to output JSONL feedback log")
    parser.add_argument("--num-events", type=int, default=500,
                        help="Number of feedback events to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df)} documents from {args.input}")

    rng = random.Random(args.seed)
    events = generate(df, args.num_events, rng)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    print(f"[INFO] Wrote {len(events)} events to {args.output}")
    print_summary(events)


if __name__ == "__main__":
    main()
