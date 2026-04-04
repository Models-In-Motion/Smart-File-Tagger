#!/usr/bin/env python3
"""
mock_predict_server.py

Lightweight mock serving endpoint that simulates the Smart File Tagger
POST /predict API. Used by data_generator.py to demonstrate real HTTP
traffic without requiring the full Nextcloud + serving stack.

Endpoint:
    POST /predict
    Body: { doc_id, filename, extracted_text, file_size_bytes }
    Returns: { file_id, predicted_tag, confidence, action, ... }

Action thresholds (matching serving team spec):
    confidence > 0.85  → auto_apply
    confidence > 0.50  → suggest
    else               → no_tag
"""

import random
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request

app = Flask(__name__)

VALID_LABELS = [
    "Lecture Notes", "Problem Set", "Exam", "Syllabus",
    "Reading", "Solution", "Project", "Recitation", "Lab", "Other",
]

# Simulate a model that is right ~70% of the time
ACCURACY = 0.70


def pick_label(real_label: str | None) -> tuple[str, float]:
    """Return (predicted_label, confidence)."""
    if real_label and random.random() < ACCURACY:
        predicted = real_label
        confidence = round(random.uniform(0.55, 0.97), 4)
    else:
        candidates = [l for l in VALID_LABELS if l != real_label] if real_label else VALID_LABELS
        predicted = random.choice(candidates)
        confidence = round(random.uniform(0.10, 0.60), 4)
    return predicted, confidence


def confidence_to_action(confidence: float) -> str:
    if confidence > 0.85:
        return "auto_apply"
    if confidence > 0.50:
        return "suggest"
    return "no_tag"


def top_predictions(predicted: str, confidence: float) -> list[dict]:
    """Build a realistic top-3 probability distribution."""
    others = [l for l in VALID_LABELS if l != predicted]
    random.shuffle(others)
    remaining = round(1.0 - confidence, 4)
    second = round(remaining * random.uniform(0.4, 0.7), 4)
    third = round(remaining - second, 4)
    return [
        {"tag": predicted,  "confidence": confidence},
        {"tag": others[0],  "confidence": second},
        {"tag": others[1],  "confidence": third},
    ]


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    body = request.get_json(force=True, silent=True) or {}

    doc_id   = body.get("doc_id", "unknown")
    filename = body.get("filename", "unknown.pdf")
    real_label = body.get("real_label")          # optional hint for realistic simulation

    predicted, confidence = pick_label(real_label)
    action = confidence_to_action(confidence)
    latency_ms = round((time.perf_counter() - t0) * 1000 + random.uniform(20, 80), 2)

    return jsonify({
        "file_id":        doc_id,
        "predicted_tag":  predicted,
        "confidence":     confidence,
        "action":         action,
        "category_type":  "fixed_baseline",
        "top_predictions": top_predictions(predicted, confidence),
        "explanation":    None,
        "model_version":  "lgbm_v1",
        "latency_ms":     latency_ms,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_version": "lgbm_v1"})


if __name__ == "__main__":
    print("[mock-predict] Starting on 0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000)
