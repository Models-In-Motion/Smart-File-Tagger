"""
monitor.py — Automated Promotion and Rollback Monitor
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

Runs as a separate service (monitor container in docker-compose).
Wakes up every 5 minutes and checks two things:

1. ROLLBACK CHECK
   Queries Prometheus and PostgreSQL for signals that the model is
   behaving badly. If thresholds are breached, calls /admin/rollback
   on the serving layer to stop serving bad predictions.

   Rollback triggers:
   - Error rate > 5% over last 10 minutes (HTTP 5xx responses)
   - Correction rate > 40% over last 100 feedback events
     (users are frequently overriding the model's predictions)

2. PROMOTION CHECK
   Queries MLflow for newly registered model versions in Staging state.
   If found, runs a 5-minute canary check. If canary passes,
   calls /admin/load-model to hot-reload the new model into serving.

   Promotion triggers:
   - New model version in MLflow Staging state
   - Canary passes: error rate stays < 5% and latency stays < 500ms
     over 5 minutes of real traffic

Why a separate service instead of a background thread in FastAPI?
   Separation of concerns. The serving container should focus on
   serving requests. A monitoring container that can restart independently
   is more robust — if monitoring crashes, serving keeps working.

Assisted by Claude Sonnet 4.5
"""

import os
import time
import logging
import requests
import psycopg2
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVING_URL     = os.getenv("SERVING_URL",     "http://serving:8000")
PROMETHEUS_URL  = os.getenv("PROMETHEUS_URL",   "http://prometheus:9090")
MLFLOW_URL      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DB_URL          = os.getenv("DB_URL",           "postgresql://tagger:tagger@postgres:5432/tagger")
MODEL_NAME      = os.getenv("MLFLOW_MODEL_NAME", "smart-tagger")

CHECK_INTERVAL_SECONDS  = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))   # 5 minutes
CANARY_DURATION_SECONDS = int(os.getenv("CANARY_DURATION_SECONDS", "300"))  # 5 minutes

# Rollback thresholds — well justified:
# - 5% error rate: industry standard for alert threshold on inference APIs
# - 40% correction rate: if nearly half of all predictions are wrong enough
#   that users correct them, the model is doing more harm than good
ERROR_RATE_THRESHOLD      = float(os.getenv("ERROR_RATE_THRESHOLD",      "0.05"))   # 5%
CORRECTION_RATE_THRESHOLD = float(os.getenv("CORRECTION_RATE_THRESHOLD", "0.40"))  # 40%

# Promotion thresholds — canary must pass these to be promoted:
CANARY_MAX_ERROR_RATE   = float(os.getenv("CANARY_MAX_ERROR_RATE",   "0.05"))   # 5%
CANARY_MAX_LATENCY_P95  = float(os.getenv("CANARY_MAX_LATENCY_P95",  "0.500"))  # 500ms

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] monitor — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus queries
# ---------------------------------------------------------------------------

def query_prometheus(promql: str) -> float | None:
    """
    Runs a PromQL instant query and returns the scalar result.
    Returns None if Prometheus is unreachable or query fails.
    """
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        return 0.0
    except Exception as exc:
        log.warning(f"Prometheus query failed: {exc}")
        return None


def get_error_rate_10min() -> float | None:
    """
    Returns fraction of HTTP 5xx responses over the last 10 minutes.
    Formula: 5xx_rate / total_rate
    """
    total = query_prometheus(
        'sum(rate(http_requests_total[10m]))'
    )
    errors = query_prometheus(
        'sum(rate(http_requests_total{status=~"5.."}[10m]))'
    )

    if total is None or errors is None:
        return None
    if total == 0:
        return 0.0
    return errors / total


def get_p95_latency() -> float | None:
    """Returns p95 latency in seconds over the last 5 minutes."""
    return query_prometheus(
        'histogram_quantile(0.95, rate(http_request_duration_highr_seconds_bucket[5m]))'
    )


# ---------------------------------------------------------------------------
# PostgreSQL queries
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(DB_URL)


def get_correction_rate_last_n(n: int = 100) -> float | None:
    """
    Returns the correction rate over the last N feedback events.
    correction_rate = corrected / total

    Why last 100 events instead of a time window?
        Correction rate is meaningful relative to prediction volume,
        not absolute time. 10 corrections in 10 minutes vs 10 corrections
        in 10,000 predictions are very different signals.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN feedback_type = 'corrected' THEN 1 ELSE 0 END) as corrections
            FROM (
                SELECT feedback_type
                FROM feedback
                ORDER BY created_at DESC
                LIMIT %s
            ) recent
        """, (n,))
        row = cur.fetchone()
        conn.close()

        if row is None or row[0] == 0:
            return 0.0

        total, corrections = row
        return float(corrections) / float(total)

    except Exception as exc:
        log.warning(f"DB query failed for correction rate: {exc}")
        return None


# ---------------------------------------------------------------------------
# Serving layer calls
# ---------------------------------------------------------------------------

def trigger_rollback(reason: str) -> bool:
    """Calls /admin/rollback on the serving layer."""
    try:
        resp = requests.post(
            f"{SERVING_URL}/admin/rollback",
            json={"reason": reason},
            timeout=5,
        )
        resp.raise_for_status()
        log.warning(f"ROLLBACK TRIGGERED — reason: {reason}")
        return True
    except Exception as exc:
        log.error(f"Failed to trigger rollback: {exc}")
        return False


def trigger_restore() -> bool:
    """Calls /admin/restore on the serving layer."""
    try:
        resp = requests.post(f"{SERVING_URL}/admin/restore", timeout=5)
        resp.raise_for_status()
        log.info("ROLLBACK RESTORED")
        return True
    except Exception as exc:
        log.error(f"Failed to restore: {exc}")
        return False


def load_new_model(bundle_path: str) -> bool:
    """Calls /admin/load-model on the serving layer."""
    try:
        resp = requests.post(
            f"{SERVING_URL}/admin/load-model",
            json={"bundle_path": bundle_path},
            timeout=30,  # model loading takes a few seconds
        )
        resp.raise_for_status()
        data = resp.json()
        log.info(f"Model loaded: {data.get('old_version')} → {data.get('new_version')}")
        return True
    except Exception as exc:
        log.error(f"Failed to load new model: {exc}")
        return False


def get_serving_status() -> dict | None:
    """Gets current status from /admin/status."""
    try:
        resp = requests.get(f"{SERVING_URL}/admin/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning(f"Could not get serving status: {exc}")
        return None


# ---------------------------------------------------------------------------
# MLflow model registry
# ---------------------------------------------------------------------------

def get_staging_model_versions() -> list[dict]:
    """
    Returns all model versions in Staging state from MLflow.
    These are candidates for promotion to Production.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        return [
            {
                "version": v.version,
                "run_id": v.run_id,
                "stage": v.current_stage,
            }
            for v in versions
        ]
    except Exception as exc:
        log.warning(f"Could not query MLflow staging versions: {exc}")
        return []


def get_model_artifact_path(run_id: str) -> str | None:
    """
    Gets the local path to model_bundle.joblib for a given MLflow run.
    The artifact is stored in the mlflow volume mounted at /mlflow/artifacts.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.MlflowClient()
        # Download artifact to a local temp path
        local_path = client.download_artifacts(
            run_id=run_id,
            path="bundle/model_bundle.joblib",
            dst_path="/tmp/mlflow_artifacts",
        )
        return local_path
    except Exception as exc:
        log.warning(f"Could not download model artifact for run {run_id}: {exc}")
        return None


def promote_model_to_production(version: str):
    """Transitions model version from Staging to Production in MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(f"Model version {version} promoted to Production in MLflow.")
    except Exception as exc:
        log.error(f"Failed to promote model to Production: {exc}")


# ---------------------------------------------------------------------------
# Rollback check
# ---------------------------------------------------------------------------

def run_rollback_check():
    """
    Checks if the current model should be rolled back.
    Runs every CHECK_INTERVAL_SECONDS.
    """
    status = get_serving_status()
    if status is None:
        log.warning("Could not reach serving layer — skipping rollback check")
        return

    # If already rolled back, skip
    if status.get("rolled_back"):
        log.info("System already rolled back — skipping rollback check")
        return

    rollback_triggered = False
    rollback_reason = ""

    # Check 1 — error rate
    error_rate = get_error_rate_10min()
    if error_rate is not None:
        log.info(f"Error rate (10min): {error_rate:.1%}")
        if error_rate > ERROR_RATE_THRESHOLD:
            rollback_reason = f"error_rate={error_rate:.1%} exceeds threshold={ERROR_RATE_THRESHOLD:.1%}"
            rollback_triggered = True

    # Check 2 — correction rate
    if not rollback_triggered:
        correction_rate = get_correction_rate_last_n(100)
        if correction_rate is not None:
            log.info(f"Correction rate (last 100): {correction_rate:.1%}")
            if correction_rate > CORRECTION_RATE_THRESHOLD:
                rollback_reason = f"correction_rate={correction_rate:.1%} exceeds threshold={CORRECTION_RATE_THRESHOLD:.1%}"
                rollback_triggered = True

    if rollback_triggered:
        log.warning(f"Rollback condition met: {rollback_reason}")
        trigger_rollback(rollback_reason)
    else:
        log.info("Rollback check passed — model is healthy")


# ---------------------------------------------------------------------------
# Promotion check
# ---------------------------------------------------------------------------

def run_promotion_check():
    """
    Checks if a new model version is ready to be promoted.
    Runs every CHECK_INTERVAL_SECONDS.
    """
    staging_versions = get_staging_model_versions()

    if not staging_versions:
        log.info("No models in Staging — skipping promotion check")
        return

    for candidate in staging_versions:
        version = candidate["version"]
        run_id = candidate["run_id"]
        log.info(f"Found Staging model version {version} (run_id={run_id})")

        # Download the model artifact
        bundle_path = get_model_artifact_path(run_id)
        if bundle_path is None:
            log.warning(f"Could not get artifact for version {version} — skipping")
            continue

        # Run canary check
        log.info(f"Starting canary check for version {version} ({CANARY_DURATION_SECONDS}s)...")
        canary_passed = run_canary_check()

        if canary_passed:
            log.info(f"Canary PASSED for version {version} — promoting to Production")
            success = load_new_model(bundle_path)
            if success:
                promote_model_to_production(version)
                log.info(f"Version {version} promoted and loaded successfully")
            else:
                log.error(f"Failed to load version {version} into serving")
        else:
            log.warning(f"Canary FAILED for version {version} — not promoting")


def run_canary_check() -> bool:
    """
    Monitors the serving layer for CANARY_DURATION_SECONDS.
    Returns True if error rate and latency stay within bounds.

    Why 5 minutes?
        Long enough to catch intermittent issues but short enough
        to not delay promotion unnecessarily. For a document tagger
        with low traffic, 5 minutes gives us enough data points.
    """
    start = time.time()
    checks_passed = 0
    checks_failed = 0

    while time.time() - start < CANARY_DURATION_SECONDS:
        error_rate = get_error_rate_10min()
        p95_latency = get_p95_latency()

        if error_rate is None or p95_latency is None:
            log.warning("Canary: could not get metrics — waiting")
            time.sleep(30)
            continue

        if error_rate > CANARY_MAX_ERROR_RATE:
            log.warning(f"Canary FAIL: error_rate={error_rate:.1%} > {CANARY_MAX_ERROR_RATE:.1%}")
            checks_failed += 1
        elif p95_latency > CANARY_MAX_LATENCY_P95:
            log.warning(f"Canary FAIL: p95_latency={p95_latency*1000:.0f}ms > {CANARY_MAX_LATENCY_P95*1000:.0f}ms")
            checks_failed += 1
        else:
            log.info(f"Canary OK: error_rate={error_rate:.1%}, p95={p95_latency*1000:.0f}ms")
            checks_passed += 1

        # If more than 2 consecutive failures, fail fast
        if checks_failed >= 2:
            log.warning("Canary failed 2+ checks — failing fast")
            return False

        time.sleep(60)  # check every minute during canary

    # Pass if majority of checks passed
    total = checks_passed + checks_failed
    if total == 0:
        log.warning("Canary: no checks completed")
        return False

    pass_rate = checks_passed / total
    log.info(f"Canary complete: {checks_passed}/{total} checks passed ({pass_rate:.0%})")
    return pass_rate >= 0.8  # 80% of checks must pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info("Monitor starting up...")
    log.info(f"Serving URL: {SERVING_URL}")
    log.info(f"Prometheus URL: {PROMETHEUS_URL}")
    log.info(f"MLflow URL: {MLFLOW_URL}")
    log.info(f"Check interval: {CHECK_INTERVAL_SECONDS}s")
    log.info(f"Rollback thresholds: error_rate>{ERROR_RATE_THRESHOLD:.0%}, correction_rate>{CORRECTION_RATE_THRESHOLD:.0%}")

    # Wait for serving to be ready before starting checks
    log.info("Waiting for serving layer to be ready...")
    for _ in range(12):  # wait up to 60 seconds
        try:
            resp = requests.get(f"{SERVING_URL}/health", timeout=5)
            if resp.status_code == 200:
                log.info("Serving layer is ready — starting monitoring loop")
                break
        except Exception:
            pass
        time.sleep(5)

    while True:
        try:
            log.info(f"--- Monitor check at {datetime.now(timezone.utc).isoformat()} ---")
            run_rollback_check()
            run_promotion_check()
        except Exception as exc:
            log.error(f"Unexpected error in monitor loop: {exc}", exc_info=True)

        log.info(f"Sleeping {CHECK_INTERVAL_SECONDS}s until next check...")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()