"""
benchmark_rayserve.py — Compare FastAPI vs Ray Serve Under Load
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This script sends concurrent requests to both FastAPI and Ray Serve
endpoints and compares latency. The key test is concurrent load,
where Ray Serve's batching advantage becomes visible.

Run from the VM (not inside a container) while both containers are up:
    docker run -d -p 8000:8000 --env-file .env -e USE_ONNX=true --name fastapi smart-tagger-final ...
    docker run -d -p 8100:8100 --name rayserve smart-tagger-rayserve
    python3 benchmark_rayserve.py

Requirements (on VM):
    pip install requests
"""

import time
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FASTAPI_URL  = "http://localhost:8000/predict"
RAYSERVE_URL = "http://localhost:8100/predict"

TEST_FILE = "test.txt"

# Test parameters
SEQUENTIAL_RUNS = 20
CONCURRENT_USERS = 50
CONCURRENT_TOTAL = 100  # total requests to send concurrently


def send_request(url: str, file_id: int) -> float:
    """Send one prediction request and return latency in ms."""
    with open(TEST_FILE, "rb") as f:
        start = time.time()
        resp = requests.post(
            url,
            files={"file": ("test.txt", f, "text/plain")},
            data={"user_id": "bench_user", "file_id": str(file_id)},
            timeout=30,
        )
        latency = (time.time() - start) * 1000

    if resp.status_code != 200:
        print(f"  ERROR: {resp.status_code} from {url}")
        return -1
    return latency


def run_sequential(url: str, name: str, n: int = SEQUENTIAL_RUNS):
    """Send n requests one at a time."""
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL TEST: {name} ({n} requests)")
    print(f"{'='*60}")

    latencies = []
    for i in range(n):
        lat = send_request(url, i + 1)
        if lat > 0:
            latencies.append(lat)

    if not latencies:
        print("  All requests failed!")
        return None

    # Drop first 2 as warmup
    warm = latencies[2:] if len(latencies) > 5 else latencies
    results = {
        "avg":  round(statistics.mean(warm), 2),
        "p50":  round(sorted(warm)[len(warm) // 2], 2),
        "p95":  round(sorted(warm)[int(len(warm) * 0.95)], 2),
        "min":  round(min(warm), 2),
        "max":  round(max(warm), 2),
    }
    print(f"  Avg: {results['avg']}ms  p50: {results['p50']}ms  "
          f"p95: {results['p95']}ms  min: {results['min']}ms  max: {results['max']}ms")
    return results


def run_concurrent(url: str, name: str,
                   num_workers: int = CONCURRENT_USERS,
                   total: int = CONCURRENT_TOTAL):
    """Send requests concurrently using a thread pool."""
    print(f"\n{'='*60}")
    print(f"CONCURRENT TEST: {name} ({total} requests, {num_workers} workers)")
    print(f"{'='*60}")

    latencies = []
    errors = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(send_request, url, i) for i in range(total)]
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
        "avg":       round(statistics.mean(latencies), 2),
        "p50":       round(sorted(latencies)[len(latencies) // 2], 2),
        "p95":       round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "min":       round(min(latencies), 2),
        "max":       round(max(latencies), 2),
        "rps":       round(rps, 1),
        "errors":    errors,
        "wall_ms":   round(wall_time, 2),
    }
    print(f"  Avg: {results['avg']}ms  p50: {results['p50']}ms  "
          f"p95: {results['p95']}ms")
    print(f"  RPS: {results['rps']}  Errors: {results['errors']}  "
          f"Wall time: {results['wall_ms']}ms")
    return results


def main():
    print("Smart File Tagger — FastAPI vs Ray Serve Benchmark")
    print(f"Test file: {TEST_FILE}")
    print(f"Sequential: {SEQUENTIAL_RUNS} requests")
    print(f"Concurrent: {CONCURRENT_TOTAL} requests, {CONCURRENT_USERS} workers")

    # Check both servers are up
    for name, port in [("FastAPI", 8000), ("Ray Serve", 8100)]:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            print(f"  {name} on :{port} — {'OK' if r.status_code == 200 else 'UNHEALTHY'}")
        except Exception:
            print(f"  {name} on :{port} — NOT REACHABLE")
            print(f"  Start it first, then re-run this benchmark.")
            sys.exit(1)

    # --- Sequential tests ---
    seq_fastapi  = run_sequential(FASTAPI_URL,  "FastAPI (ONNX)")
    seq_rayserve = run_sequential(RAYSERVE_URL, "Ray Serve (ONNX + batching)")

    # --- Concurrent tests ---
    con_fastapi  = run_concurrent(FASTAPI_URL,  "FastAPI (ONNX)")
    con_rayserve = run_concurrent(RAYSERVE_URL, "Ray Serve (ONNX + batching)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<35} {'Seq p50':>8} {'Con p50':>8} {'Con p95':>8} {'RPS':>6}")
    print("-" * 70)
    if seq_fastapi and con_fastapi:
        print(f"{'FastAPI (ONNX, 1 worker)':<35} "
              f"{seq_fastapi['p50']:>7}ms "
              f"{con_fastapi['p50']:>7}ms "
              f"{con_fastapi['p95']:>7}ms "
              f"{con_fastapi['rps']:>6}")
    if seq_rayserve and con_rayserve:
        print(f"{'Ray Serve (ONNX + batching)':<35} "
              f"{seq_rayserve['p50']:>7}ms "
              f"{con_rayserve['p50']:>7}ms "
              f"{con_rayserve['p95']:>7}ms "
              f"{con_rayserve['rps']:>6}")

    if con_fastapi and con_rayserve and con_fastapi['p50'] > 0:
        speedup = con_fastapi['p50'] / con_rayserve['p50']
        print(f"\nRay Serve concurrent p50 speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
