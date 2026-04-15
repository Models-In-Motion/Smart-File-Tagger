"""
create_fake_model.py
Creates a fake but real LightGBM model with random weights.
This is an untrained model — exactly what the professor asked for.
Run this once to generate the .pkl files, then copy them to Chameleon.
"""

import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import lightgbm as lgb

VALID_LABELS = [
    "Lecture Notes",
    "Problem Set", 
    "Exam",
    "Syllabus",
    "Reading",
    "Solution",
    "Project",
    "Recitation",
    "Lab",
    "Other",
]

print("Step 1 — Creating label encoder...")
label_encoder = LabelEncoder()
label_encoder.fit(VALID_LABELS)

print("Step 2 — Generating random training data...")
# SBERT all-MiniLM-L6-v2 produces 384-dimensional embeddings
# We create 1000 fake samples of 384 features each
X, y = make_classification(
    n_samples=1000,
    n_features=384,      # must match SBERT embedding size
    n_classes=10,        # must match number of labels
    n_informative=50,
    n_redundant=10,
    random_state=42,
)
# y values are 0-9, matching label encoder indices

print("Step 3 — Training LightGBM on random data...")
model = lgb.LGBMClassifier(
    n_estimators=10,     # very few trees — this is intentionally weak
    num_leaves=8,
    random_state=42,
    verbose=-1,
)
model.fit(X, y)

print("Step 4 — Saving model files...")
import os
os.makedirs("models", exist_ok=True)

with open("models/lgbm_classifier_v1.pkl", "wb") as f:
    pickle.dump(model, f)
print("  Saved: models/lgbm_classifier_v1.pkl")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("  Saved: models/label_encoder.pkl")

# Quick sanity check
print("\nStep 5 — Sanity check...")
with open("models/lgbm_classifier_v1.pkl", "rb") as f:
    loaded_model = pickle.load(f)

test_input = np.random.randn(1, 384)
probs = loaded_model.predict_proba(test_input)[0]
predicted_idx = probs.argmax()
predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

print(f"  Test prediction: {predicted_label} (confidence: {probs[predicted_idx]:.3f})")
print(f"  All probabilities sum to: {probs.sum():.4f}")
print("\nDone. Model files are in serving/models/")