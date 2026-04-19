"""
main.py — FastAPI Entrypoint
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

Assisted by Claude Sonnet 4.5
"""

import time
import logging
import os
from contextlib import asynccontextmanager

import httpx
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
    ensure_model_status_table_exists,
    log_prediction,
    get_model_status,
    set_model_status,
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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------
predictor: Predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    log.info("Server starting up...")

    ensure_feedback_table_exists()
    ensure_predictions_table_exists()
    ensure_model_status_table_exists()
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


class NextcloudWebhookRequest(BaseModel):
    event: dict
    user: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — reports model status, rollback state, DB reachability."""
    db_ok = True
    try:
        from feedback import get_connection
        conn = get_connection()
        conn.close()
    except Exception:
        db_ok = False

    rolled_back = get_model_status("rolled_back") == "true"
    reason = get_model_status("rollback_reason")

    return {
        "status":          "ok",
        "model_loaded":    predictor is not None,
        "model_version":   predictor.model_version if predictor else None,
        "rolled_back":     rolled_back,
        "rollback_reason": reason if rolled_back else None,
        "db_reachable":    db_ok,
    }


@app.post("/predict")
async def predict(
    file:     UploadFile = File(...),
    user_id:  str        = Form(...),
    file_id:  str        = Form(...),
):
    """
    Main prediction endpoint.
    Returns null prediction if system is rolled back.
    """
    start_time = time.time()

    # Check rollback state from PostgreSQL (shared across all workers)
    if get_model_status("rolled_back") == "true":
        reason = get_model_status("rollback_reason")
        log.warning(f"System is rolled back ({reason}), returning null prediction")
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

    # Step 3 — check custom categories
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

    # ── Prometheus metrics ───────────────────────────────────────────────────
    prediction_counter.labels(
        label=prediction_response["predicted_tag"],
        action=prediction_response["action"]
    ).inc()
    confidence_histogram.observe(prediction_response["confidence"])

    # ── Log to PostgreSQL for drift monitoring ───────────────────────────────
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


@app.post("/nextcloud/predict")
async def nextcloud_predict(request: NextcloudWebhookRequest):
    """
    Receives Nextcloud Flow webhook and fetches file via WebDAV.
    Alternative to TagController.php calling /predict directly.
    """
    node = request.event.get("node", {})
    file_id = str(node.get("id", "unknown"))
    file_path = node.get("path", "")
    user_id = request.user.get("uid", "unknown")

    nextcloud_url = os.getenv("NEXTCLOUD_INTERNAL_URL", "http://nextcloud")
    # file_path format from webhook: "/admin/files/lecture.txt"
    # WebDAV format needed: "/remote.php/dav/files/admin/lecture.txt"
    # Strip the /username/files/ prefix and rebuild correctly
    parts = file_path.strip("/").split("/", 2)
    # parts = ["admin", "files", "lecture.txt"]
    if len(parts) >= 3 and parts[1] == "files":
        webdav_path = f"/remote.php/dav/files/{parts[0]}/{parts[2]}"
    else:
        webdav_path = f"/remote.php/dav/files{file_path}"
    webdav_url = f"{nextcloud_url}{webdav_path}"
    nc_user = os.getenv("NEXTCLOUD_ADMIN_USER", "admin")
    nc_pass = os.getenv("NEXTCLOUD_ADMIN_PASSWORD", "admin")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(webdav_url, auth=(nc_user, nc_pass), timeout=30)
            resp.raise_for_status()
            file_bytes = resp.content
            filename = file_path.split("/")[-1]
    except Exception as exc:
        log.error(f"Failed to fetch file from Nextcloud WebDAV: {exc}")
        return {"success": False, "error": str(exc)}

    extracted_text, _ = extract_text(file_bytes, filename)
    if not extracted_text:
        extracted_text = ""

    result = predictor.predict(text=extracted_text, user_id=user_id)

    log_prediction(
        file_id=file_id,
        user_id=user_id,
        predicted_tag=result.predicted_tag,
        confidence=result.confidence,
        action=result.action,
        model_version=result.model_version,
        category_type="fixed_baseline",
        latency_ms=result.latency_ms,
    )

    return {
        "file_id": file_id,
        "user_id": user_id,
        "predicted_tag": result.predicted_tag,
        "confidence": result.confidence,
        "action": result.action,
        "model_version": result.model_version,
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Receives user feedback and saves to PostgreSQL."""
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

    feedback_counter.labels(feedback_type=request.feedback_type).inc()
    if request.feedback_type == "corrected":
        correction_counter.inc()

    return {
        "success": saved,
        "message": "Feedback saved" if saved else "Feedback could not be saved (logged server-side)"
    }


@app.post("/register-category")
def register_category_endpoint(request: RegisterCategoryRequest):
    """Creates a custom category for a user."""
    if predictor is None or predictor.sbert_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

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
    return {"user_id": user_id, "categories": categories, "count": len(categories)}


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
            if success else "Delete failed (logged server-side)"
        ),
    }


# ---------------------------------------------------------------------------
# Admin endpoints — called by monitor.py only
# ---------------------------------------------------------------------------

@app.post("/admin/rollback")
def admin_rollback(request: RollbackRequest):
    """
    Triggers rollback — monitor.py calls this when it detects a problem.
    Sets rolled_back=true in PostgreSQL so ALL workers stop serving predictions.
    """
    set_model_status("rolled_back", "true")
    set_model_status("rollback_reason", request.reason)
    rollback_counter.inc()
    log.warning(f"ROLLBACK TRIGGERED — reason: {request.reason}")
    return {
        "success":      True,
        "rolled_back":  True,
        "reason":       request.reason,
        "message":      "Model rolled back. /predict will return null predictions until restored.",
    }


@app.post("/admin/restore")
def admin_restore():
    """
    Restores normal prediction after a rollback.
    Called by monitor.py when a new good model is promoted.
    """
    previous_reason = get_model_status("rollback_reason")
    set_model_status("rolled_back", "false")
    set_model_status("rollback_reason", "")
    log.info(f"ROLLBACK RESTORED — was: {previous_reason}")
    return {
        "success":     True,
        "rolled_back": False,
        "message":     "Model restored. /predict will return predictions normally.",
    }


@app.post("/admin/load-model")
def admin_load_model(request: LoadModelRequest):
    """
    Hot-reloads the model without restarting the container.
    Called by monitor.py when a new model version is promoted.
    """
    global predictor

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
        os.environ["BUNDLE_PATH"] = bundle_path
        predictor = Predictor()

        # Clear rollback state on successful load
        set_model_status("rolled_back", "false")
        set_model_status("rollback_reason", "")

        log.info(f"Model reloaded: {old_version} → {predictor.model_version}")
        return {
            "success":     True,
            "old_version": old_version,
            "new_version": predictor.model_version,
            "bundle_path": bundle_path,
            "message":     "Model reloaded successfully. Rollback state cleared.",
        }
    except Exception as exc:
        log.error(f"Failed to reload model: {exc}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {exc}")


@app.get("/admin/status")
def admin_status():
    """Returns current system status for monitor.py to read."""
    rolled_back = get_model_status("rolled_back") == "true"
    reason = get_model_status("rollback_reason")
    return {
        "model_version":   predictor.model_version if predictor else None,
        "model_mode":      predictor._mode if predictor else None,
        "rolled_back":     rolled_back,
        "rollback_reason": reason if rolled_back else None,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _null_prediction(file_id: str, user_id: str) -> JSONResponse:
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