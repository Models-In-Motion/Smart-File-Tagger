"""
predictor.py — Model Inference
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

FOUR MODES — controlled by environment variables:

MODE 1 — STUB (USE_STUB_MODEL=true)
    Keyword matching, no real model. ~0.35ms latency.
    Used for integration testing only.

MODE 2 — BUNDLE/TF-IDF (USE_STUB_MODEL=false, USE_ONNX=false, bundle exists)
    Loads model_bundle.joblib from Vedant's training pipeline.
    Bundle contains: TF-IDF vectorizer + LightGBM classifier + label encoder.
    ~5-15ms latency. This is the PRIMARY production mode.

MODE 3 — ONNX SBERT (USE_STUB_MODEL=false, USE_ONNX=true)
    SBERT encoder exported to ONNX + LightGBM.
    Used if bundle is not available or for SBERT-based models.
    ~30-80ms latency.

MODE 4 — PYTORCH SBERT (USE_STUB_MODEL=false, USE_ONNX=false, no bundle)
    Real SBERT (PyTorch) + LightGBM.
    ~50-150ms latency.

Priority: STUB > BUNDLE > ONNX > PYTORCH

Assisted by Claude Sonnet 4.5
"""

import os
import logging
import random
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_STUB_MODEL = os.getenv("USE_STUB_MODEL", "false").lower() == "true"
USE_ONNX       = os.getenv("USE_ONNX", "false").lower() == "true"

# Vedant's trained model bundle (TF-IDF + LightGBM)
BUNDLE_PATH = Path(os.getenv("BUNDLE_PATH", "/models/model_bundle.joblib"))

# Legacy separate model files (SBERT-based modes)
MODEL_PATH         = Path(os.getenv("MODEL_PATH",         "/models/lgbm_classifier_v1.pkl"))
LABEL_ENCODER_PATH = Path(os.getenv("LABEL_ENCODER_PATH", "/models/label_encoder.pkl"))
ONNX_MODEL_PATH    = Path(os.getenv("ONNX_MODEL_PATH",    "/models/sbert_onnx.onnx"))

VALID_LABELS = [
    "Lecture Notes", "Problem Set", "Exam",
    "Reading", "Solution", "Project", "Other",
]

AUTO_APPLY_THRESHOLD = 0.85
SUGGEST_THRESHOLD    = 0.50


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    predicted_tag:   str
    confidence:      float
    action:          str
    top_predictions: list[dict]
    explanation:     str | None
    model_version:   str
    latency_ms:      float


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class Predictor:

    def __init__(self):
        self.model_version   = "stub_v0"
        self.sbert_model     = None
        self.onnx_session    = None
        self.lgbm_model      = None
        self.label_encoder   = None
        self.tokenizer       = None
        self.tfidf_vectorizer = None
        self._mode           = "stub"

        if USE_STUB_MODEL:
            log.info("Predictor running in STUB mode")
            self._mode = "stub"
        elif BUNDLE_PATH.exists():
            # Vedant's trained model bundle — primary production mode
            self._load_bundle()
        elif USE_ONNX:
            self._load_onnx_models()
        else:
            self._load_pytorch_models()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_bundle(self):
        """
        MODE 2 — Load Vedant's model_bundle.joblib.

        Bundle format:
            {
                "model_name":      str,
                "classifier":      fitted LGBMClassifier or LogisticRegression,
                "label_encoder":   fitted LabelEncoder,
                "featurizer_kind": "tfidf" or "sbert",
                "tfidf_vectorizer": fitted TfidfVectorizer or None,
                "sbert_model_name": str or None,
                "text_col":        "extracted_text",
                "label_col":       "label",
            }

        Why TF-IDF instead of SBERT here?
            Vedant's tfidf_lightgbm model uses TF-IDF features, not SBERT
            embeddings. TF-IDF is much faster (~5ms vs ~50ms) and achieves
            0.928 macro F1 on our dataset — sufficient for production.
            SBERT is still used for custom category matching in category_mgr.py.
        """
        import joblib

        log.info(f"Loading model bundle from {BUNDLE_PATH}...")
        bundle = joblib.load(BUNDLE_PATH)

        self.lgbm_model       = bundle["classifier"]
        self.label_encoder    = bundle["label_encoder"]
        self.tfidf_vectorizer = bundle.get("tfidf_vectorizer")
        featurizer_kind       = bundle.get("featurizer_kind", "tfidf")
        model_name            = bundle.get("model_name", "bundle")

        log.info(f"Bundle loaded: model={model_name}, featurizer={featurizer_kind}")
        log.info(f"Labels: {self.label_encoder.classes_.tolist()}")

        if featurizer_kind == "sbert":
            # SBERT-based bundle — load the sentence transformer
            sbert_name = bundle.get("sbert_model_name", "all-MiniLM-L6-v2")
            log.info(f"Loading SBERT for bundle: {sbert_name}...")
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer(sbert_name)
            log.info("SBERT loaded for bundle")
            self._mode = "bundle_sbert"
        else:
            # TF-IDF based bundle (default — tfidf_lightgbm)
            if self.tfidf_vectorizer is None:
                raise ValueError("Bundle has featurizer_kind=tfidf but tfidf_vectorizer is None")
            self._mode = "bundle_tfidf"

        # Also load SBERT for custom category matching (category_mgr.py needs it)
        # This is separate from the classifier featurizer
        if self.sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Loading SBERT for custom category matching...")
                self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("SBERT loaded for custom categories")
            except Exception as e:
                log.warning(f"Could not load SBERT for custom categories: {e}")

        self.model_version = f"{model_name}_bundle"
        log.info(f"Bundle predictor ready in mode: {self._mode}")

    def _load_pytorch_models(self):
        """MODE 4 — Load real SBERT (PyTorch) + LightGBM (legacy mode)."""
        import pickle
        from sentence_transformers import SentenceTransformer

        log.info("Loading SBERT (PyTorch mode)...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("SBERT loaded")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
        with open(MODEL_PATH, "rb") as f:
            self.lgbm_model = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self._mode = "pytorch"
        self.model_version = "lgbm_v1_pytorch"
        log.info("PyTorch predictor fully loaded")

    def _load_onnx_models(self):
        """MODE 3 — Load SBERT ONNX + LightGBM (legacy mode)."""
        import pickle
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}.")

        log.info(f"Loading ONNX session from {ONNX_MODEL_PATH}...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.onnx_session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        log.info("ONNX session loaded")

        log.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        log.info("Tokenizer loaded")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"LightGBM model not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            self.lgbm_model = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self._mode = "onnx"
        self.model_version = "lgbm_v1_onnx"
        log.info("ONNX predictor fully loaded")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, text: str, user_id: str) -> PredictionResult:
        start = time.time()

        if self._mode == "stub":
            result = self._stub_predict(text)
        elif self._mode == "bundle_tfidf":
            result = self._bundle_tfidf_predict(text)
        elif self._mode == "bundle_sbert":
            result = self._bundle_sbert_predict(text)
        elif self._mode == "onnx":
            result = self._onnx_predict(text)
        else:
            result = self._pytorch_predict(text)

        result.latency_ms = round((time.time() - start) * 1000, 2)
        return result

    def _bundle_tfidf_predict(self, text: str) -> PredictionResult:
        """
        MODE 2 (TF-IDF) — vectorize with TF-IDF then classify with LightGBM.
        This is the primary production mode with Vedant's trained model.
        """
        features = self.tfidf_vectorizer.transform([text])
        probabilities = self.lgbm_model.predict_proba(features)[0]
        return self._build_result(probabilities, self.model_version)

    def _bundle_sbert_predict(self, text: str) -> PredictionResult:
        """MODE 2 (SBERT bundle) — encode with SBERT then classify."""
        embedding = self.sbert_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).reshape(1, -1)
        probabilities = self.lgbm_model.predict_proba(embedding)[0]
        return self._build_result(probabilities, self.model_version)

    def _stub_predict(self, text: str) -> PredictionResult:
        """MODE 1 — keyword matching, no real model."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["problem set", "due", "submit", "homework", "pset"]):
            top_label, top_conf = "Problem Set", round(random.uniform(0.82, 0.96), 2)
        elif any(w in text_lower for w in ["exam", "quiz", "midterm", "final"]):
            top_label, top_conf = "Exam", round(random.uniform(0.80, 0.95), 2)
        elif any(w in text_lower for w in ["lecture", "theorem", "definition", "proof"]):
            top_label, top_conf = "Lecture Notes", round(random.uniform(0.75, 0.92), 2)
        elif any(w in text_lower for w in ["solution", "answer", "sol"]):
            top_label, top_conf = "Solution", round(random.uniform(0.75, 0.92), 2)
        elif any(w in text_lower for w in ["project", "report", "design"]):
            top_label, top_conf = "Project", round(random.uniform(0.75, 0.92), 2)
        elif any(w in text_lower for w in ["reading", "chapter", "reference"]):
            top_label, top_conf = "Reading", round(random.uniform(0.75, 0.92), 2)
        else:
            top_label = random.choice(VALID_LABELS[:-1])
            top_conf  = round(random.uniform(0.55, 0.85), 2)

        remaining   = round(1.0 - top_conf, 2)
        second_conf = round(remaining * 0.65, 2)
        third_conf  = round(remaining - second_conf, 2)
        others      = [l for l in VALID_LABELS if l != top_label]
        random.shuffle(others)

        return PredictionResult(
            predicted_tag   = top_label,
            confidence      = top_conf,
            action          = _determine_action(top_conf),
            top_predictions = [
                {"tag": top_label, "confidence": top_conf},
                {"tag": others[0], "confidence": second_conf},
                {"tag": others[1], "confidence": third_conf},
            ],
            explanation     = None,
            model_version   = "stub_v0",
            latency_ms      = 0.0,
        )

    def _pytorch_predict(self, text: str) -> PredictionResult:
        """MODE 4 — Real SBERT (PyTorch) + LightGBM."""
        embedding = self.sbert_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).reshape(1, -1)
        probabilities = self.lgbm_model.predict_proba(embedding)[0]
        return self._build_result(probabilities, "lgbm_v1_pytorch")

    def _onnx_predict(self, text: str) -> PredictionResult:
        """MODE 3 — ONNX SBERT + LightGBM."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            max_length=128,
            padding=True,
            truncation=True,
        )

        ort_inputs = {
            "input_ids":      inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        ort_outputs      = self.onnx_session.run(None, ort_inputs)
        token_embeddings = ort_outputs[0]

        attention_mask = inputs["attention_mask"]
        mask_expanded  = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask       = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embedding = sum_embeddings / sum_mask

        norm = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
        sentence_embedding = sentence_embedding / np.clip(norm, a_min=1e-9, a_max=None)

        probabilities = self.lgbm_model.predict_proba(sentence_embedding)[0]
        return self._build_result(probabilities, "lgbm_v1_onnx")

    def _build_result(self, probabilities: np.ndarray, version: str) -> PredictionResult:
        """Shared result builder for all real model modes."""
        top_idx       = int(np.argmax(probabilities))
        top_conf      = float(probabilities[top_idx])
        predicted_tag = self.label_encoder.inverse_transform([top_idx])[0]

        top3 = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "tag":        self.label_encoder.inverse_transform([i])[0],
                "confidence": round(float(probabilities[i]), 4),
            }
            for i in top3
        ]

        return PredictionResult(
            predicted_tag   = predicted_tag,
            confidence      = round(top_conf, 4),
            action          = _determine_action(top_conf),
            top_predictions = top_predictions,
            explanation     = None,
            model_version   = version,
            latency_ms      = 0.0,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _determine_action(confidence: float) -> str:
    if confidence >= AUTO_APPLY_THRESHOLD:
        return "auto_apply"
    elif confidence >= SUGGEST_THRESHOLD:
        return "suggest"
    else:
        return "no_tag"