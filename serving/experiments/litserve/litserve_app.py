"""
litserve_app.py — LitServe Deployment with Native Request Batching
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

WHY LITSERVE OVER FASTAPI:
    FastAPI processes each /predict request independently. If 8 requests
    arrive at the same time, SBERT runs 8 separate forward passes.

    LitServe (by Lightning AI) is built on top of FastAPI but adds a
    native batching layer: when max_batch_size is set, LitServe
    automatically collects concurrent requests, stacks their inputs,
    runs ONE forward pass through the model, then splits the outputs
    back to each caller. This is called "dynamic request batching."

    For transformer encoders like SBERT, batched inference is faster
    than sequential because matrix multiplications have the same
    fixed overhead whether you encode 1 or 8 texts — the GPU/CPU
    SIMD instructions process the full batch in roughly the same time.

    Unlike Ray Serve (which requires a full distributed runtime with
    ~500MB+ memory overhead), LitServe adds only ~20MB on top of
    FastAPI. This makes it ideal for resource-constrained environments
    like our 2-vCPU / 3.8GB RAM Chameleon Cloud instance.

CONCRETE EXAMPLE:
    Nextcloud deployment with 10 students uploading files simultaneously
    at a homework deadline. FastAPI: 10 serial SBERT forward passes.
    LitServe: collects into 2 batches of ~5, runs 2 forward passes.
    Result: ~2-3x higher throughput under concurrent load.

USAGE:
    python litserve_app.py
    # Server starts on port 8100

    # JSON endpoint (for benchmarking):
    curl -X POST http://localhost:8100/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "Problem Set 3 Due Friday", "user_id": "test", "file_id": "1"}'
"""

import os
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import litserve as ls
import onnxruntime as ort
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("litserve")

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
    return "no_tag"


# ---------------------------------------------------------------------------
# LitServe API — implements the 4-method interface
# ---------------------------------------------------------------------------

class SmartTaggerAPI(ls.LitAPI):
    """
    LitServe API with dynamic request batching.

    LitServe lifecycle per request:
        1. decode_request()  — called per-request, extracts text
        2. predict()         — called per-BATCH (when max_batch_size > 1)
                               receives a list of decoded inputs
        3. encode_response() — called per-request, formats JSON output

    When max_batch_size=8 and batch_timeout=0.05:
        LitServe waits up to 50ms to collect up to 8 concurrent requests,
        then calls predict() ONCE with all of them. This is the key
        advantage over FastAPI.
    """

    def setup(self, device):
        """
        Called once at startup. Load all models into memory.
        Equivalent to Predictor.__init__() in our FastAPI code.
        """
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

        self.model_version = "lgbm_v1_litserve_onnx"
        log.info("LitServe API ready")

    def decode_request(self, request):
        """
        Called once per request BEFORE batching.
        Extracts the text and metadata from the JSON body.

        Returns a dict that will be collected into a batch list.
        """
        return {
            "text":    request.get("text", ""),
            "user_id": request.get("user_id", "unknown"),
            "file_id": request.get("file_id", "0"),
            "start":   time.time(),
        }

    def batch(self, inputs):
        """
        Custom batching — LitServe calls this to combine decoded requests.
        Default auto-stacking doesn't work for dicts, so we handle it.
        """
        return inputs  # keep as list of dicts; we batch in predict()

    def predict(self, batch):
        """
        THIS IS WHERE BATCHING HAPPENS.

        When max_batch_size=8, `batch` is a list of 1-8 dicts from
        decode_request(). We tokenize ALL texts together and run
        ONE ONNX forward pass for the entire batch.

        FastAPI equivalent: 8 separate calls to _onnx_predict(),
        each with its own tokenize + forward pass.
        """
        texts = [item["text"] for item in batch]
        batch_size = len(texts)
        log.info(f"Batched predict: {batch_size} texts in one forward pass")

        # --- Tokenize all texts together ---
        # padding=True pads to longest in batch (not max_length=128)
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

        # --- Single ONNX forward pass for entire batch ---
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        token_embeddings = ort_outputs[0]  # (batch_size, seq_len, 384)

        # --- Mean pooling ---
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embeddings = sum_embeddings / sum_mask  # (batch_size, 384)

        # --- L2 normalize ---
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        # --- LightGBM classification for all embeddings ---
        probabilities = self.lgbm_model.predict_proba(sentence_embeddings)

        # --- Build results for each item in batch ---
        results = []
        for i in range(batch_size):
            probs = probabilities[i]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            predicted_tag = self.label_encoder.inverse_transform([top_idx])[0]

            top3 = np.argsort(probs)[::-1][:3]
            top_predictions = [
                {
                    "tag": self.label_encoder.inverse_transform([j])[0],
                    "confidence": round(float(probs[j]), 4),
                }
                for j in top3
            ]

            latency = round((time.time() - batch[i]["start"]) * 1000, 2)

            log.info(
                f"Prediction: file={batch[i]['file_id']} user={batch[i]['user_id']} "
                f"tag={predicted_tag} conf={round(top_conf, 4)} "
                f"action={_determine_action(top_conf)} latency={latency}ms"
            )

            results.append({
                "file_id":         batch[i]["file_id"],
                "user_id":         batch[i]["user_id"],
                "predicted_tag":   predicted_tag,
                "confidence":      round(top_conf, 4),
                "action":          _determine_action(top_conf),
                "category_type":   "fixed_baseline",
                "top_predictions": top_predictions,
                "explanation":     None,
                "model_version":   self.model_version,
                "latency_ms":      latency,
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            })

        return results

    def unbatch(self, output):
        """
        Custom unbatching — split the list of results back to
        individual responses, one per original request.
        """
        return output

    def encode_response(self, output):
        """
        Called once per request AFTER unbatching.
        Returns the final JSON response for this request.
        """
        return output


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api = SmartTaggerAPI()
    server = ls.LitServer(
        api,
        max_batch_size=8,       # collect up to 8 concurrent requests
        batch_timeout=0.05,     # wait up to 50ms to fill the batch
        accelerator="cpu",
        workers_per_device=1,
        timeout=120,
    )
    server.run(port=8100, host="0.0.0.0")
