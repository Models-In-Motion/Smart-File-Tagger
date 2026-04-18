#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export MLFLOW_TRACKING_URI=http://<CHAMELEON_VM_IP>:5000
#   ./scripts/chameleon/run_training_docker.sh tfidf_logreg

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model>"
  echo "Models: tfidf_logreg | tfidf_lightgbm | sbert_logreg | sbert_mlp"
  exit 1
fi

MODEL="$1"

case "${MODEL}" in
  tfidf_logreg|tfidf_lightgbm|sbert_logreg|sbert_mlp) ;;
  *)
    echo "Invalid model: ${MODEL}"
    exit 1
    ;;
esac

if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "Please set MLFLOW_TRACKING_URI, e.g. http://<CHAMELEON_VM_IP>:5000"
  exit 1
fi

# Resolve repo root and training directory so training can read ../data artifacts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"

# Build image once (safe to re-run).
docker build -t mlops-trainer:latest "${TRAINING_DIR}"

# Reuse Hugging Face cache so SBERT model download doesn't repeat for each run.
mkdir -p "$HOME/.cache/huggingface"

docker run --rm \
  --name "train-${MODEL}" \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -v "${REPO_ROOT}:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/training \
  mlops-trainer:latest \
  python train.py \
    --config configs/train.yaml \
    --model "${MODEL}" \
    --mlflow-uri "${MLFLOW_TRACKING_URI}" \
    --run-name "chameleon_${MODEL}_$(date +%Y%m%d_%H%M%S)"
