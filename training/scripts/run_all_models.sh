#!/usr/bin/env bash
set -euo pipefail

python3 train.py --config configs/train.yaml --model tfidf_logreg
python3 train.py --config configs/train.yaml --model tfidf_lightgbm
python3 train.py --config configs/train.yaml --model sbert_logreg
python3 train.py --config configs/train.yaml --model sbert_mlp
