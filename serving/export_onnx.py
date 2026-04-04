"""
export_onnx.py — Export SBERT to ONNX format
Model-level optimization for the serving options table.
Run this once to generate the ONNX model file.
"""
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os

os.makedirs("models", exist_ok=True)

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Exporting to ONNX...")
# Dummy input — same shape SBERT expects
# 128 is the max token length, 384 is embedding size
dummy_input = model.tokenize(["This is a test sentence for ONNX export"])

input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

torch.onnx.export(
    model[0].auto_model,
    (input_ids, attention_mask),
    "models/sbert_onnx.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids":      {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
)
print("Saved: models/sbert_onnx.onnx")
print("Done.")