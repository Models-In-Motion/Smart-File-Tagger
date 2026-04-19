import os, sys
os.environ["USE_STUB_MODEL"] = "false"
os.environ["USE_ONNX"] = "false"
os.environ["BUNDLE_PATH"] = "serving/models/model_bundle.joblib"
sys.path.insert(0, "serving/app")

import joblib
import pandas as pd

# Check what the model was trained on
bundle = joblib.load("serving/models/model_bundle.joblib")
print("=== MODEL INFO ===")
print("Model name:", bundle["model_name"])
print("Text col:", bundle["text_col"])
print("Label col:", bundle["label_col"])
print("Vocab size:", len(bundle["tfidf_vectorizer"].vocabulary_))

# Check eval set
df = pd.read_parquet("data/artifacts/versions/v2/eval.parquet")
print("\n=== EVAL SET INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Label col:", "label" in df.columns)
print("Text col:", "extracted_text" in df.columns)
print("Label distribution:")
print(df["label"].value_counts())
print("\nText extraction methods:")
print(df["text_extraction_method"].value_counts())
print("\nSample text (first 200 chars):")
print(df["extracted_text"].iloc[0][:200])

# Check train set for comparison
train = pd.read_parquet("data/artifacts/versions/v2/train.parquet")
print("\n=== TRAIN SET INFO ===")
print("Shape:", train.shape)
print("Label distribution:")
print(train["label"].value_counts())
print("Text extraction methods:")
print(train["text_extraction_method"].value_counts())
