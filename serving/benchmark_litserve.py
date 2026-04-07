"""
benchmark_litserve.py — Compare FastAPI vs LitServe Under Load
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

Sends concurrent requests to both FastAPI and LitServe and compares
latency and throughput. The key test is concurrent load, where
LitServe's batching advantage becomes visible.

Run from the VM while both containers are running:
    docker run -d -p 8000:8000 ... --name fastapi smart-tagger-final ...
    docker run -d -p 8100:8100 --name litserve smart-tagger-litserve
    pip install requests --break-system-packages
    python3 benchmark_litserve.py
"""

import time
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Install requests: pip install requests --break-system-packages")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FASTAPI_URL  = "http://localhost:8000/predict"
LITSERVE_URL = "http://localhost:8100/predict"

TEST_FILE = "test.txt"

# Read test file text for LitServe JSON requests
try:
    with open(TEST_FILE, "r") as f:
        TEST_TEXT = f.read().strip()
except FileNotFoundError:
    TEST_TEXT = "Problem Set 3 - Theory of Computation 18.404. Due Friday March 14."

SEQUENTIAL_RUNS  = 20
CONCURRENT_USERS = 50
CONCURRENT_TOTAL = 100


# ---------------------------------------------------------------------------
# Request senders
# ---------------------------------------------------------------------------

def send_fastapi(file_id: int) -> float:
    """Send multipart form request to FastAPI."""
    with open(TEST_FILE, "rb") as f:
        start = time.time()
        resp = requests.post(
            FASTAPI_URL,
            files={"file": ("test.txt", f, "text/plain")},
            data={"user_id": "bench_user", "file_id": str(file_id)},
            timeout=30,
        )
        latency = (time.time() - start) * 1000
    if resp.status_code != 200:
        return -1
    return latency


def send_litserve(file_id: int) -> float:
    """Send JSON request to LitServe."""
    start = time.time()
    resp = requests.post(
        LITSERVE_URL,
        json={"text": TEST_TEXT, "user_id": "bench_user", "file_id": str(file_id)},
        timeout=30,
    )
    latency = (time.time() - start) * 1000
    if resp.status_code != 200:
        return -1
    return latency


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_sequential(send_fn, name: str, n: int = SEQUENTIAL_RUNS):
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL TEST: {name} ({n} requests)")
    print(f"{'='*60}")

    latencies = []
    for i in range(n):
        lat = send_fn(i + 1)
        if lat > 0:
            latencies.append(lat)

    if not latencies:
        print("  All requests failed!")
        return None

    warm = latencies[2:] if len(latencies) > 5 else latencies
    results = {
        "avg": round(statistics.mean(warm), 2),
        "p50": round(sorted(warm)[len(warm) // 2], 2),
        "p95": round(sorted(warm)[int(len(warm) * 0.95)], 2),
        "min": round(min(warm), 2),
        "max": round(max(warm), 2),
    }
    print(f"  Avg: {results['avg']}ms  p50: {results['p50']}ms  "
          f"p95: {results['p95']}ms  min: {results['min']}ms  max: {results['max']}ms")
    return results


def run_concurrent(send_fn, name: str,
                   num_workers: int = CONCURRENT_USERS,
                   total: int = CONCURRENT_TOTAL):
    print(f"\n{'='*60}")
    print(f"CONCURRENT TEST: {name} ({total} requests, {num_workers} workers)")
    print(f"{'='*60}")

    latencies = []
    errors = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(send_fn, i) for i in range(total)]
        for fut in as_completed(futures):
            lat = fut.result()
            if lat > 0:
                latencies.append(lat)
            else:
                errors += 1

    wall_time = (time.time() - start_all) * 1000

    if not latencies:
        print("  All requests failed!")
        return None

    rps = total / (wall_time / 1000)
    results = {
        "avg":     round(statistics.mean(latencies), 2),
        "p50":     round(sorted(latencies)[len(latencies) // 2], 2),
        "p95":     round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "min":     round(min(latencies), 2),
        "max":     round(max(latencies), 2),
        "rps":     round(rps, 1),
        "errors":  errors,
        "wall_ms": round(wall_time, 2),
    }
    print(f"  Avg: {results['avg']}ms  p50: {results['p50']}ms  "
          f"p95: {results['p95']}ms")
    print(f"  RPS: {results['rps']}  Errors: {results['errors']}  "
          f"Wall time: {results['wall_ms']}ms")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Smart File Tagger — FastAPI vs LitServe Benchmark")
    print("=" * 60)
    print(f"Test text: {TEST_TEXT[:60]}...")
    print(f"Sequential: {SEQUENTIAL_RUNS} requests")
    print(f"Concurrent: {CONCURRENT_TOTAL} requests, {CONCURRENT_USERS} workers")

    # Health checks
    for name, port, path in [
        ("FastAPI",   8000, "/health"),
        ("LitServe",  8100, "/health"),
    ]:
        try:
            r = requests.get(f"http://localhost:{port}{path}", timeout=5)
            print(f"  {name} on :{port} — {'OK' if r.status_code == 200 else 'UNHEALTHY'}")
        except Exception:
            print(f"  {name} on :{port} — NOT REACHABLE")
            print(f"  Start it first, then re-run this benchmark.")
            sys.exit(1)

    # Sequential tests
    seq_fastapi  = run_sequential(send_fastapi,  "FastAPI (ONNX, 1 worker)")
    seq_litserve = run_sequential(send_litserve, "LitServe (ONNX + batching)")

    # Concurrent tests
    con_fastapi  = run_concurrent(send_fastapi,  "FastAPI (ONNX, 1 worker)")
    con_litserve = run_concurrent(send_litserve, "LitServe (ONNX + batching)")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Seq p50':>8} {'Con p50':>8} {'Con p95':>8} {'RPS':>6}")
    print("-" * 70)
    if seq_fastapi and con_fastapi:
        print(f"{'FastAPI (ONNX, 1 worker)':<35} "
              f"{seq_fastapi['p50']:>7}ms "
              f"{con_fastapi['p50']:>7}ms "
              f"{con_fastapi['p95']:>7}ms "
              f"{con_fastapi['rps']:>6}")
    if seq_litserve and con_litserve:
        print(f"{'LitServe (ONNX + batching)':<35} "
              f"{seq_litserve['p50']:>7}ms "
              f"{con_litserve['p50']:>7}ms "
              f"{con_litserve['p95']:>7}ms "
              f"{con_litserve['rps']:>6}")

    if con_fastapi and con_litserve and con_fastapi['p50'] > 0:
        speedup = con_fastapi['p50'] / con_litserve['p50']
        rps_gain = con_litserve['rps'] / con_fastapi['rps'] if con_fastapi['rps'] > 0 else 0
        print(f"\nLitServe concurrent p50 speedup: {speedup:.2f}x")
        print(f"LitServe RPS improvement: {rps_gain:.2f}x")


if __name__ == "__main__":
    main()
