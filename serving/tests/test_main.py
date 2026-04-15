"""
test_main.py — Basic Tests
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

These are smoke tests — they check the most basic things work.
They do NOT test the ML model quality (that's Vedant's job).
They check: endpoints respond, response format is correct, error cases are handled.

How to run:
    pip install pytest httpx
    cd serving/
    pytest tests/test_main.py -v

Why httpx?
    FastAPI's test client uses httpx under the hood.
    It lets us make real HTTP requests to our app without starting a server.
"""

import io
import pytest
from fastapi.testclient import TestClient

# We need to set env vars before importing main.py
# because main.py imports config.py which reads env vars at import time
import os
os.environ.setdefault("DB_URL", "")   # empty = DB unavailable, which is fine for tests

from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Minimal fake PDF for tests
# ---------------------------------------------------------------------------

FAKE_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 << /Type /Font
/Subtype /Type1 /BaseFont /Helvetica >> >> >> >> >> endobj
4 0 obj << /Length 80 >>
stream
BT /F1 12 Tf 100 700 Td (Problem Set 3 Due Friday) Tj ET
endstream
endobj
xref 0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref 400
%%EOF"""


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

def test_health_returns_200():
    """The /health endpoint must always return 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_required_fields():
    """Health response must include status and model_loaded fields."""
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Predict endpoint tests
# ---------------------------------------------------------------------------

def test_predict_returns_200_with_valid_file():
    """Uploading a valid PDF should return 200."""
    response = client.post(
        "/predict",
        files={"file": ("test.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
        data={"user_id": "test_user", "file_id": "file_001"},
    )
    assert response.status_code == 200


def test_predict_response_has_required_fields():
    """Prediction response must match the agreed sample_output.json schema."""
    response = client.post(
        "/predict",
        files={"file": ("test.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
        data={"user_id": "test_user", "file_id": "file_001"},
    )
    data = response.json()

    required_fields = [
        "file_id", "user_id", "predicted_tag", "confidence",
        "action", "category_type", "top_predictions",
        "explanation", "model_version", "latency_ms", "timestamp"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_predict_action_is_valid():
    """Action field must be one of the three valid values."""
    response = client.post(
        "/predict",
        files={"file": ("test.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
        data={"user_id": "test_user", "file_id": "file_001"},
    )
    data = response.json()
    assert data["action"] in {"auto_apply", "suggest", "no_tag"}


def test_predict_confidence_is_between_0_and_1():
    """Confidence must be a float between 0 and 1."""
    response = client.post(
        "/predict",
        files={"file": ("test.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
        data={"user_id": "test_user", "file_id": "file_001"},
    )
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_with_empty_file_does_not_crash():
    """
    Empty file upload should return a valid null prediction, not a 500 error.
    This tests the fail-open design — Nextcloud must keep working even if
    the sidecar gets a bad file.
    """
    response = client.post(
        "/predict",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        data={"user_id": "test_user", "file_id": "file_002"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "no_tag"


def test_predict_with_text_file():
    """Plain text files should also work."""
    text_content = b"This is a syllabus. Office hours: Monday 2-4pm. Grading policy: 40% homework."
    response = client.post(
        "/predict",
        files={"file": ("syllabus.txt", io.BytesIO(text_content), "text/plain")},
        data={"user_id": "test_user", "file_id": "file_003"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Feedback endpoint tests
# ---------------------------------------------------------------------------

def test_feedback_accepted_returns_200():
    """Valid accepted feedback should return 200."""
    response = client.post(
        "/feedback",
        json={
            "file_id":           "file_001",
            "user_id":           "test_user",
            "predicted_tag":     "Problem Set",
            "confidence":        0.91,
            "action_taken":      "auto_apply",
            "feedback_type":     "accepted",
            "corrected_tag":     None,
            "model_version":     "stub_v0",
            "extraction_method": "pdfminer",
        }
    )
    assert response.status_code == 200


def test_feedback_corrected_requires_corrected_tag():
    """
    Feedback type 'corrected' must include a corrected_tag.
    Without it the feedback is useless for retraining.
    """
    response = client.post(
        "/feedback",
        json={
            "file_id":       "file_001",
            "user_id":       "test_user",
            "predicted_tag": "Problem Set",
            "confidence":    0.91,
            "action_taken":  "suggest",
            "feedback_type": "corrected",
            "corrected_tag": None,        # missing — should fail
            "model_version": "stub_v0",
        }
    )
    # Should return 200 but with success=False (we don't raise for this)
    data = response.json()
    assert data["success"] is False


def test_feedback_invalid_type_returns_422():
    """Invalid feedback_type should return 422 Unprocessable Entity."""
    response = client.post(
        "/feedback",
        json={
            "file_id":       "file_001",
            "user_id":       "test_user",
            "predicted_tag": "Problem Set",
            "confidence":    0.91,
            "action_taken":  "suggest",
            "feedback_type": "invalid_type",   # not valid
            "corrected_tag": None,
            "model_version": "stub_v0",
        }
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Categories endpoint tests
# ---------------------------------------------------------------------------

def test_list_categories_returns_200():
    """GET /categories should return 200 even for a user with no categories."""
    response = client.get("/categories?user_id=test_user")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "count" in data