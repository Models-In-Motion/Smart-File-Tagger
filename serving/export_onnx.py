"""
export_onnx.py — Export SBERT to ONNX Format
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This script exports the SBERT all-MiniLM-L6-v2 encoder to ONNX format.
The ONNX model is then used by predictor.py in MODE 3 (USE_ONNX = True).

Why ONNX is faster on CPU:
    PyTorch runs operations dynamically with Python overhead.
    ONNX Runtime compiles the full computation graph into optimized
    C++ kernels, removes redundant operations, and fuses layers.
    On CPU this typically gives 1.5x-3x speedup for transformer encoders.

Run this ONCE on Chameleon before switching predictor to ONNX mode:
    python3 export_onnx.py

Output:
    models/sbert_onnx.onnx

Then in predictor.py set:
    USE_STUB_MODEL = False
    USE_ONNX       = True

And add to requirements.txt:
    onnxruntime==1.17.3
"""

import os
import time
import numpy as np
from pathlib import Path

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — Load SBERT
# ---------------------------------------------------------------------------
print("Step 1 — Loading SBERT (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
import torch

sbert = SentenceTransformer("all-MiniLM-L6-v2")
transformer = sbert[0].auto_model   # the underlying HuggingFace transformer
transformer.eval()
print("  SBERT loaded")

# ---------------------------------------------------------------------------
# Step 2 — Create dummy input for ONNX tracing
# ---------------------------------------------------------------------------
print("Step 2 — Creating dummy input for tracing...")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
dummy_text = "Problem Set 3 Due Friday. Show L is undecidable."
dummy_inputs = tokenizer(
    dummy_text,
    return_tensors="pt",
    max_length=128,
    padding="max_length",
    truncation=True,
)
input_ids      = dummy_inputs["input_ids"]
attention_mask = dummy_inputs["attention_mask"]
print(f"  input_ids shape: {input_ids.shape}")

# ---------------------------------------------------------------------------
# Step 3 — Export to ONNX
# ---------------------------------------------------------------------------
onnx_path = Path("models/sbert_onnx.onnx")
print(f"Step 3 — Exporting to {onnx_path}...")

torch.onnx.export(
    transformer,
    (input_ids, attention_mask),
    str(onnx_path),
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["last_hidden_state"],
    dynamic_axes = {
        "input_ids":        {0: "batch_size", 1: "sequence_length"},
        "attention_mask":   {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state":{0: "batch_size", 1: "sequence_length"},
    },
    opset_version   = 14,
    do_constant_folding = True,   # fold constant ops for speed
)
print(f"  Saved: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

# ---------------------------------------------------------------------------
# Step 4 — Benchmark PyTorch vs ONNX
# ---------------------------------------------------------------------------
print("\nStep 4 — Benchmarking PyTorch vs ONNX (20 runs each)...")

import onnxruntime as ort

# ONNX session
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    str(onnx_path),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

ort_inputs = {
    "input_ids":      input_ids.numpy().astype(np.int64),
    "attention_mask": attention_mask.numpy().astype(np.int64),
}

# Warmup
for _ in range(3):
    with torch.no_grad():
        transformer(input_ids, attention_mask)
    session.run(None, ort_inputs)

# PyTorch benchmark
pytorch_times = []
for _ in range(20):
    t = time.time()
    with torch.no_grad():
        transformer(input_ids, attention_mask)
    pytorch_times.append((time.time() - t) * 1000)

# ONNX benchmark
onnx_times = []
for _ in range(20):
    t = time.time()
    session.run(None, ort_inputs)
    onnx_times.append((time.time() - t) * 1000)

pytorch_avg = sum(pytorch_times) / len(pytorch_times)
onnx_avg    = sum(onnx_times)    / len(onnx_times)
speedup     = pytorch_avg / onnx_avg

print(f"\n  PyTorch avg latency : {pytorch_avg:.2f}ms")
print(f"  ONNX avg latency    : {onnx_avg:.2f}ms")
print(f"  Speedup             : {speedup:.2f}x")

# ---------------------------------------------------------------------------
# Step 5 — Verify outputs match
# ---------------------------------------------------------------------------
print("\nStep 5 — Verifying ONNX output matches PyTorch output...")

with torch.no_grad():
    pt_output = transformer(input_ids, attention_mask).last_hidden_state.numpy()

onnx_output = session.run(None, ort_inputs)[0]

max_diff = np.abs(pt_output - onnx_output).max()
print(f"  Max output difference: {max_diff:.6f}")
if max_diff < 1e-3:
    print("  ✅ ONNX output matches PyTorch (difference within tolerance)")
else:
    print("  ⚠️  Larger difference than expected — check opset version")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
{'='*50}
ONNX Export Complete
{'='*50}
Output file  : {onnx_path}
File size    : {onnx_path.stat().st_size / 1024 / 1024:.1f} MB
PyTorch avg  : {pytorch_avg:.2f}ms
ONNX avg     : {onnx_avg:.2f}ms
Speedup      : {speedup:.2f}x

Next steps:
  1. In predictor.py set USE_ONNX = True
  2. Add onnxruntime to requirements.txt
  3. Rebuild Docker image: docker build -t smart-tagger-serving .
  4. Run container and re-test with Locust
{'='*50}
""")