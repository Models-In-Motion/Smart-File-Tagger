#!/usr/bin/env python3
"""Training entrypoint for the Nextcloud document auto-tagging project.

This script supports four model variants:
- tfidf_logreg
- tfidf_lightgbm
- sbert_logreg
- sbert_mlp
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

SUPPORTED_MODELS = {
    "tfidf_logreg",
    "tfidf_lightgbm",
    "sbert_logreg",
    "sbert_mlp",
}


@dataclass
class SplitData:
    x_train: Any
    x_val: Any
    x_test: Any
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train document tagger models")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to config YAML")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Model variant to train",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow tracking URI (if omitted, uses config value)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name override",
    )
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Local directory to store model artifacts",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_and_filter_data(cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    data_cfg = cfg["data"]
    df = pd.read_parquet(data_cfg["path"])

    text_col = data_cfg["text_col"]
    label_col = data_cfg["label_col"]
    allowed_labels = data_cfg.get("allowed_labels")

    if text_col not in df.columns:
        raise ValueError(f"text_col '{text_col}' not found in data")
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in data")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].fillna("").astype(str)

    # Keep only rows with non-empty text and labels
    df = df[(df[text_col].str.strip() != "") & (df[label_col].str.strip() != "")]

    if allowed_labels:
        df = df[df[label_col].isin(allowed_labels)]

    if df.empty:
        raise ValueError("No rows left after filtering")

    return df[text_col], df[label_col]


def split_data(texts: pd.Series, labels: pd.Series, cfg: dict[str, Any]) -> tuple[SplitData, LabelEncoder]:
    split_cfg = cfg["split"]
    seed = int(split_cfg["random_state"])
    train_size = float(split_cfg["train_size"])
    val_size = float(split_cfg["val_size"])
    test_size = float(split_cfg["test_size"])

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    x_train_txt, x_rest_txt, y_train_raw, y_rest_raw = train_test_split(
        texts,
        labels,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )

    val_ratio_in_rest = val_size / (val_size + test_size)
    x_val_txt, x_test_txt, y_val_raw, y_test_raw = train_test_split(
        x_rest_txt,
        y_rest_raw,
        train_size=val_ratio_in_rest,
        random_state=seed,
        stratify=y_rest_raw,
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)

    return (
        SplitData(
            x_train=x_train_txt,
            x_val=x_val_txt,
            x_test=x_test_txt,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        ),
        encoder,
    )


def build_tfidf_features(split: SplitData, cfg: dict[str, Any]) -> tuple[SplitData, TfidfVectorizer]:
    tfidf_cfg = cfg["models"]["tfidf"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
    )

    x_train = vectorizer.fit_transform(split.x_train)
    x_val = vectorizer.transform(split.x_val)
    x_test = vectorizer.transform(split.x_test)

    return (
        SplitData(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=split.y_train,
            y_val=split.y_val,
            y_test=split.y_test,
        ),
        vectorizer,
    )


def build_sbert_features(split: SplitData, cfg: dict[str, Any]) -> tuple[SplitData, SentenceTransformer]:
    from sentence_transformers import SentenceTransformer

    sbert_cfg = cfg["models"]["sbert"]
    model = SentenceTransformer(sbert_cfg["model_name"], device=sbert_cfg["device"])

    def encode(texts: pd.Series) -> np.ndarray:
        return model.encode(
            texts.tolist(),
            batch_size=int(sbert_cfg["batch_size"]),
            show_progress_bar=False,
            normalize_embeddings=bool(sbert_cfg["normalize_embeddings"]),
            convert_to_numpy=True,
        )

    x_train = encode(split.x_train)
    x_val = encode(split.x_val)
    x_test = encode(split.x_test)

    return (
        SplitData(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=split.y_train,
            y_val=split.y_val,
            y_test=split.y_test,
        ),
        model,
    )


def build_classifier(model_name: str, cfg: dict[str, Any], num_classes: int):
    if model_name in {"tfidf_logreg", "sbert_logreg"}:
        c = cfg["models"]["logreg"]
        kwargs = {
            "max_iter": int(c["max_iter"]),
            "C": float(c["c"]),
            "solver": c["solver"],
        }
        if "multi_class" in c:
            kwargs["multi_class"] = c["multi_class"]

        try:
            return LogisticRegression(**kwargs)
        except TypeError:
            # Newer scikit-learn versions may remove or deprecate some args.
            kwargs.pop("multi_class", None)
            return LogisticRegression(**kwargs)

    if model_name == "tfidf_lightgbm":
        from lightgbm import LGBMClassifier

        c = cfg["models"]["lightgbm"]
        return LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=int(c["n_estimators"]),
            learning_rate=float(c["learning_rate"]),
            num_leaves=int(c["num_leaves"]),
            subsample=float(c["subsample"]),
            colsample_bytree=float(c["colsample_bytree"]),
            random_state=int(c["random_state"]),
            n_jobs=int(c["n_jobs"]),
        )

    if model_name == "sbert_mlp":
        c = cfg["models"]["mlp"]
        return MLPClassifier(
            hidden_layer_sizes=tuple(c["hidden_layer_sizes"]),
            activation=c["activation"],
            learning_rate_init=float(c["learning_rate_init"]),
            max_iter=int(c["max_iter"]),
            random_state=int(c["random_state"]),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(len(classes)))
    for idx, class_name in enumerate(classes):
        metrics[f"f1_{slugify(str(class_name))}"] = float(per_class[idx])

    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Prefer CLI, then env (set in docker-compose for in-network MLflow), then config.
    mlflow_uri = (
        args.mlflow_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or cfg["mlflow"].get("tracking_uri")
    )
    experiment_name = args.experiment_name or cfg["mlflow"]["experiment_name"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    texts, labels = load_and_filter_data(cfg)
    print(f"  rows: {len(texts)}", flush=True)
    split_raw, encoder = split_data(texts, labels, cfg)
    print("Building features...", flush=True)

    feature_start = time.perf_counter()
    if args.model.startswith("tfidf_"):
        split, featurizer = build_tfidf_features(split_raw, cfg)
        featurizer_kind = "tfidf"
        sbert_name = None
    else:
        split, featurizer = build_sbert_features(split_raw, cfg)
        featurizer_kind = "sbert"
        sbert_name = cfg["models"]["sbert"]["model_name"]
    feature_seconds = time.perf_counter() - feature_start
    print(f"  feature build took {feature_seconds:.1f}s", flush=True)

    classifier = build_classifier(args.model, cfg, num_classes=len(encoder.classes_))

    print("Training classifier...", flush=True)
    train_start = time.perf_counter()
    classifier.fit(split.x_train, split.y_train)
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    y_val_pred = classifier.predict(split.x_val)
    y_test_pred = classifier.predict(split.x_test)
    infer_seconds = time.perf_counter() - infer_start

    val_metrics = {f"val_{k}": v for k, v in evaluate(split.y_val, y_val_pred, encoder.classes_).items()}
    test_metrics = {f"test_{k}": v for k, v in evaluate(split.y_test, y_test_pred, encoder.classes_).items()}

    runtime_metrics = {
        "feature_build_seconds": float(feature_seconds),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(infer_seconds),
        "avg_inference_ms_per_sample": float((infer_seconds / len(split.y_test)) * 1000.0),
        "n_train": int(len(split.y_train)),
        "n_val": int(len(split.y_val)),
        "n_test": int(len(split.y_test)),
    }

    model_artifact_dir = output_dir / args.model
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_artifact_dir / "model_bundle.joblib"

    bundle = {
        "model_name": args.model,
        "classifier": classifier,
        "label_encoder": encoder,
        "featurizer_kind": featurizer_kind,
        "tfidf_vectorizer": featurizer if featurizer_kind == "tfidf" else None,
        "sbert_model_name": sbert_name,
        "text_col": cfg["data"]["text_col"],
        "label_col": cfg["data"]["label_col"],
    }
    joblib.dump(bundle, artifact_path)

    runtime_metrics["model_artifact_size_bytes"] = int(artifact_path.stat().st_size)

    params = {
        "model": args.model,
        "data_path": cfg["data"]["path"],
        "text_col": cfg["data"]["text_col"],
        "label_col": cfg["data"]["label_col"],
        "split_train_size": cfg["split"]["train_size"],
        "split_val_size": cfg["split"]["val_size"],
        "split_test_size": cfg["split"]["test_size"],
        "split_random_state": cfg["split"]["random_state"],
    }

    tags = {
        "git_sha": get_git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
    }

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"Logging to MLflow at {mlflow_uri}", flush=True)
    else:
        print("Logging to local MLflow (./mlruns); set MLFLOW_TRACKING_URI to use a server.", flush=True)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(params)
        mlflow.set_tags(tags)

        # Log model-specific hyperparameters
        model_cfg_key = "tfidf" if args.model.startswith("tfidf") else "sbert"
        for key, value in cfg["models"][model_cfg_key].items():
            mlflow.log_param(f"{model_cfg_key}_{key}", value)

        if args.model.endswith("logreg"):
            for key, value in cfg["models"]["logreg"].items():
                mlflow.log_param(f"logreg_{key}", value)
        elif args.model.endswith("lightgbm"):
            for key, value in cfg["models"]["lightgbm"].items():
                mlflow.log_param(f"lightgbm_{key}", value)
        elif args.model.endswith("mlp"):
            for key, value in cfg["models"]["mlp"].items():
                mlflow.log_param(f"mlp_{key}", value)

        all_metrics = {}
        all_metrics.update(val_metrics)
        all_metrics.update(test_metrics)
        all_metrics.update(runtime_metrics)

        for key, value in all_metrics.items():
            mlflow.log_metric(key, float(value))

        # ── Quality gates ────────────────────────────────────────────────
        # Only register the model if it meets minimum quality thresholds.
        # This prevents bad models from being deployed automatically.
        MACRO_F1_THRESHOLD = 0.60
        MIN_CLASS_F1_THRESHOLD = 0.40

        test_macro_f1 = all_metrics["test_macro_f1"]
        min_class_f1 = min(
            v for k, v in all_metrics.items()
            if k.startswith("test_f1_")
        )

        gates_passed = (
            test_macro_f1 >= MACRO_F1_THRESHOLD
            and min_class_f1 >= MIN_CLASS_F1_THRESHOLD
        )

        mlflow.log_metric("quality_gate_passed", float(gates_passed))
        mlflow.log_metric("min_class_f1", min_class_f1)

        if gates_passed:
            print(
                f"Quality gates PASSED — "
                f"macro_f1={test_macro_f1:.3f}, "
                f"min_class_f1={min_class_f1:.3f}",
                flush=True,
            )
            mlflow.sklearn.log_model(
                sk_model=classifier,
                artifact_path="model",
                registered_model_name="smart-tagger" if gates_passed else None,
            )
            # Also save the full bundle as a separate artifact for the serving layer
            mlflow.log_artifact(str(artifact_path), artifact_path="bundle")

        else:
            print(
                f"Quality gates FAILED — "
                f"macro_f1={test_macro_f1:.3f} (need {MACRO_F1_THRESHOLD}), "
                f"min_class_f1={min_class_f1:.3f} (need {MIN_CLASS_F1_THRESHOLD})",
                flush=True,
            )
            print("Model NOT registered.", flush=True)

        summary_path = model_artifact_dir / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(str(summary_path), artifact_path="metrics")

    print("Training complete")
    print(json.dumps({**val_metrics, **test_metrics, **runtime_metrics}, indent=2))


if __name__ == "__main__":
    main()
