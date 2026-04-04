"""
predictor.py — Model Inference
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This file takes extracted text and returns a predicted label + confidence.

There are TWO modes this file can run in:

MODE 1 — STUB (right now, before Vedant delivers the trained model)
    Returns a realistic-looking fake prediction.
    Everything else in the system still works — endpoints, feedback,
    load testing — without needing a real model.

MODE 2 — REAL MODEL (after Vedant delivers lgbm_classifier_v1.pkl)
    Loads SBERT, encodes the text into a 384-dimensional embedding,
    passes that into LightGBM, returns real predictions.

To switch from MODE 1 to MODE 2:
    Set USE_STUB_MODEL = False in this file,
    and make sure the model file path in MODEL_PATH is correct.

Why SBERT + LightGBM (and not just one model)?
    SBERT is like a very smart translator. It reads a paragraph of text
    and converts it into 384 numbers that capture the *meaning* of the
    text. Two Problem Sets from different courses will have similar numbers.
    A Syllabus will have very different numbers.

    LightGBM is a fast decision-tree classifier. It takes those 384 numbers
    and learns which combinations mean "Problem Set" vs "Lecture Notes".

    SBERT alone can't classify — it just translates.
    LightGBM alone can't understand text — it needs numbers.
    Together they work perfectly.
"""

import os
import logging
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set this to False once Vedant delivers the trained model
USE_STUB_MODEL = True

# Path to the trained LightGBM model file
# This will be a Docker volume mount in production
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/lgbm_classifier_v1.pkl"))

# Path to the label encoder (maps "Problem Set" → integer and back)
LABEL_ENCODER_PATH = Path(os.getenv("LABEL_ENCODER_PATH", "/models/label_encoder.pkl"))

# All valid labels — must match DATASET_SCHEMA.md exactly
VALID_LABELS = [
    "Lecture Notes",
    "Problem Set",
    "Exam",
    "Syllabus",
    "Reading",
    "Solution",
    "Project",
    "Recitation",
    "Lab",
    "Other",
]

# Confidence thresholds — must match the table in the project summary
AUTO_APPLY_THRESHOLD = 0.85   # above this → auto-apply the tag
SUGGEST_THRESHOLD    = 0.50   # between this and 0.85 → suggest to user
                               # below 0.50 → no tag, user labels manually


# ---------------------------------------------------------------------------
# Data class for the prediction result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Everything the /predict endpoint needs to build its response.
    A dataclass is like a simple container — just holds fields together.
    """
    predicted_tag:   str
    confidence:      float
    action:          str          # "auto_apply", "suggest", or "no_tag"
    top_predictions: list[dict]   # top 3 labels with their confidence scores
    explanation:     str | None   # LLM explanation, or None
    model_version:   str
    latency_ms:      float


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class Predictor:
    """
    Wraps SBERT + LightGBM into a single object.

    Why a class and not just a function?
        Loading SBERT takes ~2-3 seconds. We load it ONCE when the
        server starts, keep it in memory, and reuse it for every request.
        If we loaded it inside a function, every request would take 3 extra
        seconds. A class lets us load once in __init__ and reuse in predict().
    """

    def __init__(self):
        self.model_version = "stub_v0"
        self.sbert_model   = None
        self.lgbm_model    = None
        self.label_encoder = None

        if USE_STUB_MODEL:
            log.info("Predictor running in STUB mode — no real model loaded")
            log.info("Set USE_STUB_MODEL = False and provide model files to use real predictions")
        else:
            self._load_models()

    def _load_models(self):
        """
        Load SBERT and LightGBM from disk.
        Called once at server startup.
        """
        import pickle
        from sentence_transformers import SentenceTransformer

        # Load SBERT
        # "all-MiniLM-L6-v2" is a small (80MB), fast model that works
        # well for document classification tasks. It maps text → 384 floats.
        log.info("Loading SBERT model (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("SBERT loaded")

        # Load LightGBM classifier (trained by Vedant)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Has Vedant delivered lgbm_classifier_v1.pkl yet?"
            )
        with open(MODEL_PATH, "rb") as f:
            self.lgbm_model = pickle.load(f)
        log.info(f"LightGBM model loaded from {MODEL_PATH}")

        # Load label encoder
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)
        log.info("Label encoder loaded")

        self.model_version = "lgbm_v1"

    def predict(self, text: str, user_id: str) -> PredictionResult:
        """
        Main prediction function. Takes text, returns a PredictionResult.

        Args:
            text    : extracted text from the uploaded file
            user_id : Nextcloud user ID — used to load the per-user model
                      (per-user models are a stretch goal, for now all users
                       share the same baseline model)

        Returns:
            PredictionResult with label, confidence, action, etc.
        """
        import time
        start = time.time()

        if USE_STUB_MODEL:
            result = self._stub_predict(text)
        else:
            result = self._real_predict(text)

        # Calculate latency
        result.latency_ms = round((time.time() - start) * 1000, 2)
        return result

    def _stub_predict(self, text: str) -> PredictionResult:
        """
        Returns a fake but realistic prediction.
        Used until Vedant delivers the trained model.

        The stub is not completely random — it uses simple keyword matching
        so the fake predictions at least make intuitive sense during testing.
        This makes it easier to spot bugs in the integration.
        """
        text_lower = text.lower()

        # Simple keyword-based heuristic for the stub
        if any(w in text_lower for w in ["problem set", "due", "submit", "homework"]):
            top_label, top_conf = "Problem Set", round(random.uniform(0.82, 0.96), 2)
        elif any(w in text_lower for w in ["exam", "quiz", "midterm", "final"]):
            top_label, top_conf = "Exam", round(random.uniform(0.80, 0.95), 2)
        elif any(w in text_lower for w in ["syllabus", "office hours", "grading policy"]):
            top_label, top_conf = "Syllabus", round(random.uniform(0.85, 0.98), 2)
        elif any(w in text_lower for w in ["lecture", "theorem", "definition", "proof"]):
            top_label, top_conf = "Lecture Notes", round(random.uniform(0.75, 0.92), 2)
        elif any(w in text_lower for w in ["solution", "answer", "solved"]):
            top_label, top_conf = "Solution", round(random.uniform(0.78, 0.94), 2)
        else:
            top_label, top_conf = random.choice(VALID_LABELS[:-1]), round(random.uniform(0.55, 0.85), 2)

        # Build fake top-3 predictions
        # The remaining confidence is split among runner-up labels
        remaining = round(1.0 - top_conf, 2)
        second_conf = round(remaining * 0.65, 2)
        third_conf  = round(remaining - second_conf, 2)

        other_labels = [l for l in VALID_LABELS if l != top_label]
        random.shuffle(other_labels)

        top_predictions = [
            {"tag": top_label,          "confidence": top_conf},
            {"tag": other_labels[0],    "confidence": second_conf},
            {"tag": other_labels[1],    "confidence": third_conf},
        ]

        action = _determine_action(top_conf)

        return PredictionResult(
            predicted_tag   = top_label,
            confidence      = top_conf,
            action          = action,
            top_predictions = top_predictions,
            explanation     = None,   # LLM explanation not implemented yet
            model_version   = "stub_v0",
            latency_ms      = 0.0,    # will be set by predict()
        )

    def _real_predict(self, text: str) -> PredictionResult:
        """
        Real prediction using SBERT + LightGBM.
        Only called when USE_STUB_MODEL = False.

        Step 1: SBERT encodes text → 384-dim embedding (a list of 384 floats)
        Step 2: LightGBM takes those 384 floats → probability for each label
        Step 3: Pick the label with highest probability
        Step 4: Determine action based on confidence thresholds
        """
        # Step 1 — SBERT embedding
        # encode() returns a numpy array of shape (384,)
        embedding = self.sbert_model.encode(
            text,
            normalize_embeddings=True,  # normalize so cosine similarity works
            show_progress_bar=False,
        )

        # LightGBM expects a 2D array: (n_samples, n_features)
        # We have 1 sample, so reshape to (1, 384)
        embedding_2d = embedding.reshape(1, -1)

        # Step 2 — LightGBM prediction
        # predict_proba returns a (1, n_classes) array of probabilities
        probabilities = self.lgbm_model.predict_proba(embedding_2d)[0]

        # Step 3 — pick winner
        top_idx  = int(np.argmax(probabilities))
        top_conf = float(probabilities[top_idx])

        # Convert integer index back to label string
        predicted_tag = self.label_encoder.inverse_transform([top_idx])[0]

        # Build top-3 predictions
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "tag":        self.label_encoder.inverse_transform([i])[0],
                "confidence": round(float(probabilities[i]), 4),
            }
            for i in top3_indices
        ]

        action = _determine_action(top_conf)

        return PredictionResult(
            predicted_tag   = predicted_tag,
            confidence      = round(top_conf, 4),
            action          = action,
            top_predictions = top_predictions,
            explanation     = None,
            model_version   = self.model_version,
            latency_ms      = 0.0,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _determine_action(confidence: float) -> str:
    """
    Maps a confidence score to one of three action strings.
    These strings must match exactly what TagController.php expects.
    """
    if confidence >= AUTO_APPLY_THRESHOLD:
        return "auto_apply"
    elif confidence >= SUGGEST_THRESHOLD:
        return "suggest"
    else:
        return "no_tag"