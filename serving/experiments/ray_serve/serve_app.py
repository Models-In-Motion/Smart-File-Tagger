"""
serve_app.py — Ray Serve Deployment with Native Batching
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

WHY RAY SERVE OVER FASTAPI:
    FastAPI processes each request independently — if 10 requests arrive
    at the same time, SBERT runs 10 separate forward passes.

    Ray Serve has a built-in @serve.batch decorator that automatically
    collects concurrent requests and processes them in a SINGLE forward
    pass through SBERT. For transformer models, batched inference is
    significantly faster than sequential because the GPU/CPU matrix
    operations are the same cost whether you encode 1 or 8 texts.

    This is a framework-level optimization that FastAPI cannot do without
    custom async queuing code.

USAGE:
    python serve_app.py
    # Server starts on port 8100

    curl -X POST http://localhost:8100/predict \
        -F "user_id=test_user" -F "file_id=1" -F "file=@test.txt"
"""

import os
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

import onnxruntime as ort
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ray_serve")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH         = Path(os.getenv("MODEL_PATH",         "/models/lgbm_classifier_v1.pkl"))
LABEL_ENCODER_PATH = Path(os.getenv("LABEL_ENCODER_PATH", "/models/label_encoder.pkl"))
ONNX_MODEL_PATH    = Path(os.getenv("ONNX_MODEL_PATH",    "/models/sbert_onnx.onnx"))

AUTO_APPLY_THRESHOLD = 0.85
SUGGEST_THRESHOLD    = 0.50


def _determine_action(confidence: float) -> str:
    if confidence >= AUTO_APPLY_THRESHOLD:
        return "auto_apply"
    elif confidence >= SUGGEST_THRESHOLD:
        return "suggest"
    else:
        return "no_tag"


# ---------------------------------------------------------------------------
# Ray Serve Deployment
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2},
)
class SmartTaggerDeployment:
    """
    Ray Serve deployment with @serve.batch for automatic request batching.

    When multiple requests arrive concurrently, Ray Serve collects them
    into a batch and calls _batched_encode() ONCE with all texts.
    SBERT encodes all texts in a single forward pass, which is faster
    than encoding them one at a time because:
        - Tokenizer pads to the longest text in the batch (not max_length)
        - One matrix multiplication handles all texts simultaneously
        - CPU SIMD instructions are better utilized with larger matrices
    """

    def __init__(self):
        log.info("Loading ONNX session...")
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

        with open(MODEL_PATH, "rb") as f:
            self.lgbm_model = pickle.load(f)
        log.info(f"LightGBM loaded from {MODEL_PATH}")

        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)
        log.info("Label encoder loaded")

        self.model_version = "lgbm_v1_ray_onnx"
        log.info("Ray Serve deployment ready")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def _batched_encode(self, texts: list[str]) -> list[np.ndarray]:
        """
        THIS IS THE KEY ADVANTAGE OF RAY SERVE.

        @serve.batch automatically collects up to 8 concurrent requests
        and passes all their texts here as a single list. We tokenize
        and encode them in ONE forward pass.

        FastAPI would need custom async queue + background worker code
        to achieve this. Ray Serve does it with one decorator.

        Args:
            texts: list of 1-8 strings (automatically collected by Ray)

        Returns:
            list of embeddings, one per input text
        """
        batch_size = len(texts)
        log.info(f"Batched encode: {batch_size} texts in one forward pass")

        # Tokenize all texts together — pads to longest in batch, not 128
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            max_length=128,
            padding=True,
            truncation=True,
        )

        ort_inputs = {
            "input_ids":      inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        # Single ONNX forward pass for entire batch
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        token_embeddings = ort_outputs[0]  # shape: (batch_size, seq_len, 384)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embeddings = sum_embeddings / sum_mask  # (batch_size, 384)

        # L2 normalize
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        # Return as list of individual embeddings
        return [sentence_embeddings[i] for i in range(batch_size)]

    def _extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Simple plaintext extraction (matches extractor.py for .txt files)."""
        try:
            return file_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    async def predict_single(self, text: str, file_id: str, user_id: str) -> dict:
        """Process one prediction request using batched encoding."""
        start = time.time()

        # This call may be batched with other concurrent requests
        embedding = await self._batched_encode(text)
        embedding = embedding.reshape(1, -1)

        # LightGBM classification
        probabilities = self.lgbm_model.predict_proba(embedding)[0]
        top_idx = int(np.argmax(probabilities))
        top_conf = float(probabilities[top_idx])
        predicted_tag = self.label_encoder.inverse_transform([top_idx])[0]

        top3 = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "tag": self.label_encoder.inverse_transform([i])[0],
                "confidence": round(float(probabilities[i]), 4),
            }
            for i in top3
        ]

        latency = round((time.time() - start) * 1000, 2)

        log.info(
            f"Prediction: file={file_id} user={user_id} "
            f"tag={predicted_tag} conf={round(top_conf, 4)} "
            f"action={_determine_action(top_conf)} latency={latency}ms"
        )

        return {
            "file_id":       file_id,
            "user_id":       user_id,
            "predicted_tag": predicted_tag,
            "confidence":    round(top_conf, 4),
            "action":        _determine_action(top_conf),
            "category_type": "fixed_baseline",
            "top_predictions": top_predictions,
            "explanation":   None,
            "model_version": self.model_version,
            "latency_ms":    latency,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

    async def __call__(self, request: Request) -> JSONResponse:
        """
        HTTP handler — same multipart/form-data API as FastAPI /predict.
        """
        path = request.url.path

        if path == "/health":
            return JSONResponse({"status": "ok", "model_version": self.model_version})

        if path != "/predict":
            return JSONResponse({"error": "Not found"}, status_code=404)

        # Parse multipart form data
        form = await request.form()
        user_id = form.get("user_id", "unknown")
        file_id = form.get("file_id", "0")
        file_obj = form.get("file")

        if file_obj is None:
            return JSONResponse({
                "file_id": file_id, "user_id": user_id,
                "predicted_tag": None, "confidence": 0.0,
                "action": "no_tag", "latency_ms": 0.0,
            })

        file_bytes = await file_obj.read()
        text = self._extract_text(file_bytes, file_obj.filename or "unknown.txt")

        if not text:
            return JSONResponse({
                "file_id": file_id, "user_id": user_id,
                "predicted_tag": None, "confidence": 0.0,
                "action": "no_tag", "latency_ms": 0.0,
            })

        result = await self.predict_single(text, file_id, user_id)
        return JSONResponse(result)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

app = SmartTaggerDeployment.bind()

# Replace your current serve.run call with this:
if __name__ == "__main__":
    # Initialize serve with the desired host and port
    serve.start(
        http_options={"host": "0.0.0.0", "port": 8100}
    )
    # Deploy the application
    serve.run(app)
