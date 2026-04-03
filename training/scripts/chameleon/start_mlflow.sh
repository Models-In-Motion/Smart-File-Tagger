#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/chameleon/start_mlflow.sh
# Optional:
#   MLFLOW_PORT=5000 MLFLOW_HOME=$HOME/mlflow ./scripts/chameleon/start_mlflow.sh

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_HOME="${MLFLOW_HOME:-$HOME/mlflow}"

mkdir -p "${MLFLOW_HOME}/artifacts"

docker rm -f mlflow-server >/dev/null 2>&1 || true

docker run -d \
  --name mlflow-server \
  -p "${MLFLOW_PORT}:5000" \
  -v "${MLFLOW_HOME}:/mlflow" \
  ghcr.io/mlflow/mlflow:v2.14.3 \
  mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts

echo "MLflow started on port ${MLFLOW_PORT}."
echo "Check logs with: docker logs -f mlflow-server"
