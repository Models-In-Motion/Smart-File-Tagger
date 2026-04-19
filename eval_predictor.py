import os
os.environ["USE_STUB_MODEL"] = "false"
os.environ["USE_ONNX"] = "false"
os.environ["BUNDLE_PATH"] = "serving/models/model_bundle.joblib"

import sys
sys.path.insert(0, "serving/app")

import pandas as pd
from predictor import Predictor
from sklearn.metrics import classification_report

# Load eval set
df = pd.read_parquet("data/artifacts/versions/v2/eval.parquet")
print(f"Eval set: {len(df)} rows")
print(df["label"].value_counts())

# Run predictions
p = Predictor()
print(f"\nMode: {p._mode}")

y_true = []
y_pred = []

for i, row in df.iterrows():
    text = str(row["extracted_text"])
    if not text.strip():
        continue
    result = p.predict(text=text, user_id="eval")
    y_true.append(row["label"])
    y_pred.append(result.predicted_tag)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
