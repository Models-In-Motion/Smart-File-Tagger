"""
locustfile.py — Load Testing
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This file simulates many users hitting the serving API at the same time.
It measures latency and throughput so we can fill the serving options table.

What Locust does:
    You tell it "pretend there are N users, each doing these actions".
    It fires off requests as fast as it can and records how long each one takes.
    At the end you get p50/p95 latency and requests/second numbers.

How to run:
    pip install locust
    locust -f locustfile.py --host http://<your-server>:8000

    Then open http://localhost:8089 in your browser.
    Set number of users and spawn rate, click Start.

For the serving options table you need to run three tests:
    Test 1 (baseline)  : 10 users, record numbers
    Test 2 (medium)    : 50 users, record numbers
    Test 3 (peak)      : 100 users, record numbers (check if error rate spikes)
"""

import os
import io
from locust import HttpUser, task, between


# ---------------------------------------------------------------------------
# Sample test data
# We create a tiny fake PDF in memory rather than reading a real file.
# This keeps the test focused on server performance, not file I/O.
# ---------------------------------------------------------------------------

# Minimal valid PDF — 1 page, contains some text about problem sets
# This is a real PDF structure — just very small
FAKE_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 << /Type /Font
/Subtype /Type1 /BaseFont /Helvetica >> >> >> >> endobj
4 0 obj << /Length 120 >>
stream
BT /F1 12 Tf 100 700 Td
(Problem Set 3 - Due Friday at 11:59pm) Tj
0 -20 Td (1. Show that L is undecidable.) Tj ET
endstream
endobj
xref 0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref 446
%%EOF"""


class SmartTaggerUser(HttpUser):
    """
    Simulates one Nextcloud user uploading files and submitting feedback.

    wait_time = between(0.5, 2):
        Each simulated user waits 0.5-2 seconds between requests.
        This mimics realistic human behavior.
        For pure throughput testing, set this to between(0, 0.1).
    """
    wait_time = between(0.5, 2)

    # ── Tasks ──────────────────────────────────────────────────────────────
    # @task(3) means this task is 3x more likely to run than @task(1).
    # /predict is called most often because that's the hot path.

    @task(5)
    def predict(self):
        """Simulates a file upload to /predict."""
        # Use a plain text file instead of fake PDF bytes
        # This is simpler and works reliably with multipart uploads
        text_content = (
            b"Problem Set 3 - Due Friday at 11:59pm. "
            b"Question 1: Show that L is undecidable. "
            b"Question 2: Prove the halting problem cannot be solved."
        )
        self.client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(text_content), "text/plain")},
            data={
                "user_id": "locust_test_user",
                "file_id": "test_file_001",
            },
        )

    @task(2)
    def submit_feedback(self):
        """
        Simulates a user clicking Accept after seeing a suggestion.
        """
        self.client.post(
            "/feedback",
            json={
                "file_id":           "test_file_001",
                "user_id":           "locust_test_user",
                "predicted_tag":     "Problem Set",
                "confidence":        0.88,
                "action_taken":      "suggest",
                "feedback_type":     "accepted",
                "corrected_tag":     None,
                "model_version":     "stub_v0",
                "extraction_method": "pdfminer",
            }
        )

    @task(1)
    def health_check(self):
        """
        Simulates Nextcloud's periodic health pings.
        """
        self.client.get("/health")

    @task(1)
    def list_categories(self):
        """
        Simulates loading the custom categories settings page.
        """
        self.client.get("/categories?user_id=locust_test_user")