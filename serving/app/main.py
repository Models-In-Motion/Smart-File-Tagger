"""
main.py — FastAPI Entrypoint
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This is the front door of the entire serving system.
It defines all the API endpoints and coordinates between
extractor.py, predictor.py, feedback.py, and category_mgr.py.

How FastAPI works (plain English):
    You define a Python function and put a decorator on it like @app.post("/predict").
    FastAPI automatically handles the HTTP part — receiving the request,
    parsing it, and sending back your return value as JSON.
    You just write Python. FastAPI does the HTTP plumbing.

Startup:
    When the server starts, we load the ML model into memory ONCE.
    This is the Predictor object. Every request reuses the same object.
    Loading it per-request would add 2-3 seconds to every single prediction.

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
# Instrumentator automatically tracks:
#   - http_requests_total (by method, endpoint, status code)
#   - http_request_duration_seconds (latency histogram)
# We add custom metrics on top for ML-specific tracking.
Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    "predictions_total",
    "Total predictions made, broken down by predicted label and action taken",
    ["label", "action"]
)
# Why track by label AND action? If "Lecture Notes" always gets auto_tag
# but "Solution" always gets no_tag, that tells you the model is
# underconfident on Solution — useful signal for retraining.

confidence_histogram = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores across all requests",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
# Why a histogram? If the distribution shifts over time (e.g. average
# confidence drops from 0.8 to 0.4), that's a sign of model drift
# even before user feedback tells you something is wrong.

feedback_counter = Counter(
    "feedback_total",
    "Total feedback events received, broken down by feedback type",
    ["feedback_type"]
)
# Tracks accepted / rejected / corrected separately so you can
# compute correction rate = corrected / (accepted + rejected + corrected)

correction_counter = Counter(
    "feedback_corrections_total",
    "Total times users corrected the predicted label (feedback_type=corrected)"
)
# Kept separate for easy alerting: if this spikes, something is wrong.


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Nextcloud pings this before sending webhook requests.
    Returns 200 OK if the service is up.
    Also reports whether the model is loaded and whether DB is reachable.
    """
    db_ok = True
    try:
        from feedback import get_connection
        conn = get_connection()
        conn.close()
    except Exception:
        db_ok = False

    return {
        "status":        "ok",
        "model_loaded":  predictor is not None,
        "model_version": predictor.model_version if predictor else None,
        "db_reachable":  db_ok,
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

    Request format: multipart/form-data (file upload)
        - file    : the actual file bytes
        - user_id : Nextcloud user ID (e.g. "nc_user_vsp7234")
        - file_id : Nextcloud file node ID (e.g. "8821")

    Response: JSON matching sample_output.json schema
    """
    start_time = time.time()

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
    Called by the Nextcloud app when a user clicks Accept or Reject.

    Always returns 200 even if DB write fails — we never surface
    database errors to the user. We log them server-side.
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
    """
    Creates a custom category for a user from 3-10 example files.
    Called by the Nextcloud settings page when a user creates a new category.
    """
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
    """
    Returns all custom categories for a user.
    Usage: GET /categories?user_id=nc_user_vsp7234
    """
    categories = list_user_categories(user_id)
    return {
        "user_id":    user_id,
        "categories": categories,
        "count":      len(categories),
    }


@app.delete("/delete-category")
def delete_category_endpoint(request: DeleteCategoryRequest):
    """
    Deletes a custom category for a user.
    """
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
# Private helpers
# ---------------------------------------------------------------------------

def _null_prediction(file_id: str, user_id: str) -> JSONResponse:
    """
    Fail-open response. Returned when file reading or extraction fails.
    """
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