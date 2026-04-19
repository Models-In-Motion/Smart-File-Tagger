"""
main.py — FastAPI Entrypoint
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This is the front door of the entire serving system.
It defines all the API endpoints and coordinates between
extractor.py, predictor.py, feedback.py, and category_mgr.py.

Assisted by Claude Sonnet 4.5
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from extractor import extract_text
from predictor import Predictor
from feedback import (
    save_feedback,
    get_feedback_for_user,
    ensure_feedback_table_exists,
    ensure_predictions_table_exists,
    log_prediction,
    FeedbackType,
)
from category_mgr import (
    register_category,
    find_best_custom_category,
    list_user_categories,
    delete_category,
    ensure_categories_table_exists,
)
import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application startup and shutdown
# ---------------------------------------------------------------------------
predictor: Predictor = None   # global, set during startup

# ---------------------------------------------------------------------------
# Rollback state — in-memory flag (Option B)
# monitor.py calls /admin/rollback to flip this flag.
# When rolled_back=True, /predict uses stub mode instead of the real model.
# Flag resets to False on container restart, which is intentional —
# monitor.py will re-detect and re-trigger rollback if the problem persists.
# ---------------------------------------------------------------------------
_rolled_back: bool = False
_rollback_reason: str = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    log.info("Server starting up...")

    ensure_feedback_table_exists()
    ensure_predictions_table_exists()
    ensure_categories_table_exists()

    predictor = Predictor()

    log.info("Server ready to accept requests")

    yield

    log.info("Server shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Smart File Tagger — Serving API",
    description = "Predicts document categories for Nextcloud files",
    version     = "0.1.0",
    lifespan    = lifespan,
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    "predictions_total",
    "Total predictions made, broken down by predicted label and action taken",
    ["label", "action"]
)

confidence_histogram = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores across all requests",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

feedback_counter = Counter(
    "feedback_total",
    "Total feedback events received, broken down by feedback type",
    ["feedback_type"]
)

correction_counter = Counter(
    "feedback_corrections_total",
    "Total times users corrected the predicted label (feedback_type=corrected)"
)

rollback_counter = Counter(
    "rollback_total",
    "Total number of times the model was rolled back"
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    file_id:            str
    user_id:            str
    predicted_tag:      str
    confidence:         float
    action_taken:       str
    feedback_type:      str
    corrected_tag:      str | None = None
    model_version:      str
    extraction_method:  str | None = None


class RegisterCategoryRequest(BaseModel):
    user_id:        str
    category_name:  str
    example_texts:  list[str]


class DeleteCategoryRequest(BaseModel):
    user_id:        str
    category_name:  str


class RollbackRequest(BaseModel):
    reason: str = "manual"


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    bundle_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Reports model status, rollback state, and DB reachability.
    """
    db_ok = True
    try:
        from feedback import get_connection
        conn = get_connection()
        conn.close()
    except Exception:
        db_ok = False

    return {
        "status":          "ok",
        "model_loaded":    predictor is not None,
        "model_version":   predictor.model_version if predictor else None,
        "rolled_back":     _rolled_back,
        "rollback_reason": _rollback_reason if _rolled_back else None,
        "db_reachable":    db_ok,
    }


@app.post("/predict")
async def predict(
    file:     UploadFile = File(...),
    user_id:  str        = Form(...),
    file_id:  str        = Form(...),
):
    """
    Main prediction endpoint. Called by Nextcloud Flow webhook when
    a user uploads a file.

    If the system is rolled back, returns a null prediction rather than
    serving potentially bad model outputs.
    """
    start_time = time.time()

    # If rolled back, fail open with null prediction
    if _rolled_back:
        log.warning(f"System is rolled back ({_rollback_reason}), returning null prediction")
        return _null_prediction(file_id, user_id)

    # Step 1 — read file bytes
    try:
        file_bytes = await file.read()
    except Exception as exc:
        log.error(f"Failed to read uploaded file: {exc}")
        return _null_prediction(file_id, user_id)

    if not file_bytes:
        log.warning(f"Empty file received: {file.filename}")
        return _null_prediction(file_id, user_id)

    # Step 2 — extract text
    extracted_text, extraction_method = extract_text(file_bytes, file.filename or "unknown.pdf")

    if not extracted_text:
        log.warning(f"Text extraction returned empty for file {file_id}")
        extracted_text = ""

    # Step 3 — check custom categories (Layer 2: few-shot prototype matching)
    custom_tag   = None
    custom_score = 0.0

    if predictor.sbert_model is not None and extracted_text:
        custom_tag, custom_score = find_best_custom_category(
            user_id     = user_id,
            text        = extracted_text,
            sbert_model = predictor.sbert_model,
        )

    # Step 4 — get prediction
    if custom_tag:
        from predictor import _determine_action
        action = _determine_action(custom_score)
        prediction_response = {
            "file_id":          file_id,
            "user_id":          user_id,
            "predicted_tag":    custom_tag,
            "confidence":       round(custom_score, 4),
            "action":           action,
            "category_type":    "custom",
            "top_predictions":  [{"tag": custom_tag, "confidence": round(custom_score, 4)}],
            "explanation":      None,
            "model_version":    predictor.model_version,
            "latency_ms":       round((time.time() - start_time) * 1000, 2),
            "timestamp":        _now_iso(),
        }
    else:
        result = predictor.predict(text=extracted_text, user_id=user_id)
        prediction_response = {
            "file_id":          file_id,
            "user_id":          user_id,
            "predicted_tag":    result.predicted_tag,
            "confidence":       result.confidence,
            "action":           result.action,
            "category_type":    "fixed_baseline",
            "top_predictions":  result.top_predictions,
            "explanation":      result.explanation,
            "model_version":    result.model_version,
            "latency_ms":       result.latency_ms,
            "timestamp":        _now_iso(),
        }

    log.info(
        f"Prediction: file={file_id} user={user_id} "
        f"tag={prediction_response['predicted_tag']} "
        f"conf={prediction_response['confidence']} "
        f"action={prediction_response['action']} "
        f"latency={prediction_response['latency_ms']}ms"
    )

    # ── Record Prometheus metrics ────────────────────────────────────────────
    prediction_counter.labels(
        label=prediction_response["predicted_tag"],
        action=prediction_response["action"]
    ).inc()
    confidence_histogram.observe(prediction_response["confidence"])

    # Log to PostgreSQL for Viral's drift monitoring
    log_prediction(
        file_id           = prediction_response["file_id"],
        user_id           = prediction_response["user_id"],
        predicted_tag     = prediction_response["predicted_tag"],
        confidence        = prediction_response["confidence"],
        action            = prediction_response["action"],
        model_version     = prediction_response["model_version"],
        category_type     = prediction_response["category_type"],
        latency_ms        = prediction_response["latency_ms"],
    )

    return JSONResponse(content=prediction_response)


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """
    Receives user feedback (accept/reject/correction) and saves to PostgreSQL.
    """
    try:
        feedback_type = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feedback_type '{request.feedback_type}'. "
                   f"Must be one of: accepted, rejected, corrected"
        )

    saved = save_feedback(
        file_id           = request.file_id,
        user_id           = request.user_id,
        predicted_tag     = request.predicted_tag,
        confidence        = request.confidence,
        action_taken      = request.action_taken,
        feedback_type     = feedback_type,
        corrected_tag     = request.corrected_tag,
        model_version     = request.model_version,
        extraction_method = request.extraction_method,
    )

    # ── Record Prometheus metrics ────────────────────────────────────────────
    feedback_counter.labels(feedback_type=request.feedback_type).inc()
    if request.feedback_type == "corrected":
        correction_counter.inc()

    return {
        "success": saved,
        "message": "Feedback saved" if saved else "Feedback could not be saved (logged server-side)"
    }


@app.post("/register-category")
def register_category_endpoint(request: RegisterCategoryRequest):
    """Creates a custom category for a user from 3-10 example files."""
    if predictor is None or predictor.sbert_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet — try again in a few seconds"
        )

    result = register_category(
        user_id       = request.user_id,
        category_name = request.category_name,
        example_texts = request.example_texts,
        sbert_model   = predictor.sbert_model,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.get("/categories")
def list_categories(user_id: str):
    """Returns all custom categories for a user."""
    categories = list_user_categories(user_id)
    return {
        "user_id":    user_id,
        "categories": categories,
        "count":      len(categories),
    }


@app.delete("/delete-category")
def delete_category_endpoint(request: DeleteCategoryRequest):
    """Deletes a custom category for a user."""
    success = delete_category(
        user_id       = request.user_id,
        category_name = request.category_name,
    )
    return {
        "success": success,
        "message": (
            f"Category '{request.category_name}' deleted"
            if success else
            "Delete failed (logged server-side)"
        ),
    }


# ---------------------------------------------------------------------------
# Admin endpoints — called by monitor.py only, not by users
# ---------------------------------------------------------------------------

@app.post("/admin/rollback")
def admin_rollback(request: RollbackRequest):
    """
    Triggers a rollback — monitor.py calls this when it detects a problem.

    Effect: sets _rolled_back=True so /predict returns null predictions
    instead of potentially bad model outputs. Safe for users — they just
    don't get tag suggestions until the model is restored.

    This endpoint is internal — it should not be exposed to the internet.
    In production you would add an API key check here.
    """
    global _rolled_back, _rollback_reason

    _rolled_back = True
    _rollback_reason = request.reason
    rollback_counter.inc()

    log.warning(f"ROLLBACK TRIGGERED — reason: {request.reason}")

    return {
        "success": True,
        "rolled_back": True,
        "reason": request.reason,
        "message": "Model rolled back. /predict will return null predictions until restored.",
    }


@app.post("/admin/restore")
def admin_restore():
    """
    Restores normal prediction after a rollback.
    Called by monitor.py when a new good model is promoted,
    or manually by the team after investigating.
    """
    global _rolled_back, _rollback_reason

    previous_reason = _rollback_reason
    _rolled_back = False
    _rollback_reason = ""

    log.info(f"ROLLBACK RESTORED — was rolled back due to: {previous_reason}")

    return {
        "success": True,
        "rolled_back": False,
        "message": "Model restored. /predict will return predictions normally.",
    }


@app.post("/admin/load-model")
def admin_load_model(request: LoadModelRequest):
    """
    Hot-reloads the model without restarting the container.
    Called by monitor.py when a new model version is promoted to Production.

    Args:
        bundle_path: optional path to the new model bundle.
                     If not provided, reloads from the default BUNDLE_PATH.
    """
    global predictor, _rolled_back, _rollback_reason

    import os
    from pathlib import Path

    bundle_path = request.bundle_path or os.getenv("BUNDLE_PATH", "/models/model_bundle.joblib")

    if not Path(bundle_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model bundle not found at {bundle_path}"
        )

    try:
        log.info(f"Hot-reloading model from {bundle_path}...")
        old_version = predictor.model_version if predictor else "none"

        # Set env var so Predictor picks up the new path
        os.environ["BUNDLE_PATH"] = bundle_path
        predictor = Predictor()

        # Clear rollback state if model loaded successfully
        _rolled_back = False
        _rollback_reason = ""

        log.info(f"Model reloaded: {old_version} → {predictor.model_version}")

        return {
            "success": True,
            "old_version": old_version,
            "new_version": predictor.model_version,
            "bundle_path": bundle_path,
            "message": "Model reloaded successfully. Rollback state cleared.",
        }

    except Exception as exc:
        log.error(f"Failed to reload model: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {exc}"
        )


@app.get("/admin/status")
def admin_status():
    """
    Returns current system status for monitor.py to read.
    Includes rollback state, model version, and basic metrics.
    """
    return {
        "model_version":   predictor.model_version if predictor else None,
        "model_mode":      predictor._mode if predictor else None,
        "rolled_back":     _rolled_back,
        "rollback_reason": _rollback_reason if _rolled_back else None,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _null_prediction(file_id: str, user_id: str) -> JSONResponse:
    """Fail-open response for errors and rollback state."""
    return JSONResponse(content={
        "file_id":          file_id,
        "user_id":          user_id,
        "predicted_tag":    None,
        "confidence":       0.0,
        "action":           "no_tag",
        "category_type":    None,
        "top_predictions":  [],
        "explanation":      None,
        "model_version":    predictor.model_version if predictor else "unknown",
        "latency_ms":       0.0,
        "timestamp":        _now_iso(),
    })


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()