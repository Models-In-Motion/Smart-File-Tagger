import os
os.environ["USE_STUB_MODEL"] = "false"
os.environ["USE_ONNX"] = "false"
os.environ["BUNDLE_PATH"] = "serving/models/model_bundle.joblib"

import sys
sys.path.insert(0, "serving/app")

from predictor import Predictor

p = Predictor()
print("Mode:", p._mode)
print("Model version:", p.model_version)
print("Labels:", p.label_encoder.classes_.tolist())

test_cases = [
    ("This lecture covers neural networks and backpropagation", "Lecture Notes"),
    ("Problem set 3 questions on linear algebra", "Problem Set"),
    ("Final exam covering chapters 1 through 8", "Exam"),
    ("Solution to homework 2 using dynamic programming", "Solution"),
    ("Reading assignment chapter 5 probability theory", "Reading"),
    ("Final project description build a classifier", "Project"),
]

print("\nPrediction results:")
all_passed = True
for text, expected in test_cases:
    result = p.predict(text=text, user_id="test_user")
    status = "✓" if result.predicted_tag == expected else "✗"
    if result.predicted_tag != expected:
        all_passed = False
    print(f"{status} Expected: {expected:20s} Got: {result.predicted_tag:20s} Conf: {result.confidence:.3f} Latency: {result.latency_ms}ms")

print("\nAll passed:", all_passed)
print("\nNote: if using fake bundle, predictions may not match expected — that is OK.")
print("What matters is: mode=bundle_tfidf, no errors, confidence > 0, latency < 100ms")
