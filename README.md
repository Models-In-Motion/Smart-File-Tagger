# Smart File Tagger

**NYU ECE-GY 9183 MLOps, Spring 2026**

| Role | Name | NetID |
|---|---|---|
| Serving Lead | Krish Jani | kj2743 |
| Training Lead | Vedant Pradhan | vsp7234 |
| Data Lead | Viral Dalal | vd2477 |

---

## What This System Does

Smart File Tagger is an ML sidecar service for Nextcloud (self-hosted cloud storage) that automatically tags uploaded documents into one of 5 categories: **Lecture Notes, Problem Set, Exam, Reading, Other**. Users can also define custom categories using example files.

When a user uploads a file to Nextcloud:
1. A webhook fires to the PHP plugin
2. The plugin fetches the file and sends it to the FastAPI serving layer
3. The serving layer extracts text, runs TF-IDF + LightGBM inference, and returns a prediction
4. If confidence ≥ 0.75 → tag is applied automatically
5. If confidence 0.50–0.74 → tag is suggested to the user
6. If confidence < 0.50 → no tag applied

User corrections are logged as feedback and trigger automatic retraining when enough accumulate.

---

## Infrastructure

- **Platform:** Chameleon Cloud (KVM), m1.xlarge (8 vCPU, 16GB RAM), Ubuntu 22.04
- **Deployment:** Docker Compose v1 (single node)
- **Floating IP:** `129.114.27.110`

---

## Service URLs and Credentials

| Service | URL | Credentials |
|---|---|---|
| Nextcloud (user-facing) | http://129.114.27.110:8080 | admin / admin |
| FastAPI serving | http://129.114.27.110:8000/docs | — |
| MLflow | http://129.114.27.110:5001 | — |
| Grafana | http://129.114.27.110:3000 | admin / admin |
| Prometheus | http://129.114.27.110:9090 | — |
| Adminer (DB UI) | http://129.114.27.110:8081 | See below |

**Adminer login:**
- System: PostgreSQL
- Server: `postgres`
- Username: `tagger`
- Password: `tagger`
- Database: `tagger`

---

## Prerequisites

- Ubuntu 22.04 instance on Chameleon Cloud (m1.xlarge recommended)
- Docker and Docker Compose v1.29 installed
- Git and Git LFS installed
- Floating IP assigned and SSH access configured

---

## Deployment From Scratch

### 1. SSH into the instance
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.27.110
```

### 2. Install dependencies
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose git git-lfs python3-pip
sudo usermod -aG docker cc
newgrp docker
```

### 3. Clone the repository
```bash
git clone https://github.com/Models-In-Motion/Smart-File-Tagger.git smart-file-tagger
cd smart-file-tagger
git lfs install
git lfs pull
```

### 4. Create data directories for persistent storage
```bash
mkdir -p data/postgres data/mlflow_artifacts data/nextcloud data/grafana
```

### 5. Start all services
```bash
docker-compose up -d postgres mlflow serving prometheus grafana nextcloud monitor retrain-cron adminer
```

### 6. Wait for services to be healthy (~60 seconds)
```bash
docker-compose ps
```

All services should show `Up` or `Up (healthy)`.

### 7. Initialize Nextcloud
```bash
bash infra/init-nextcloud.sh
bash infra/enable-nextcloud-app.sh
```

### 8. Verify serving is ready
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Should return `"status": "ok"` and `"model_loaded": true`.

---

## Architecture

```
Nextcloud (PHP plugin)
    ↓ webhook on file upload
FastAPI serving layer (port 8000)
    ↓ two-layer prediction
    ├── Layer 1: SBERT cosine similarity vs custom category prototypes
    └── Layer 2: TF-IDF vectorizer + LightGBM classifier
    ↓ results logged to
PostgreSQL
    ↓ feedback accumulates
retrain-cron (hourly)
    ↓ threshold reached
batch_pipeline.py → train.py → MLflow (Staging)
    ↓ monitor.py canary check (5 min)
Serving hot-reload → MLflow (Production)
```

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| TF-IDF + LightGBM (not SBERT) for main classifier | 5–15ms inference vs 50–150ms; F1 ~0.56 on 5-class academic doc task |
| SBERT only for custom categories | Few-shot prototype matching needs semantic similarity; TF-IDF can't generalize from 3 examples |
| Docker Compose (not Kubernetes) | 3-person team, single-node deployment |
| Fail-open design | If DB is down, predictions still work |
| DB-backed rollback state | Consistent across 4 Gunicorn workers |
| Bind mounts (not named volumes) | Data survives container kills and VM restarts |

---

## Prediction Thresholds

| Threshold | Value | Purpose |
|---|---|---|
| AUTO_APPLY_THRESHOLD | 0.75 | Confidence above which tag is applied automatically |
| SUGGEST_THRESHOLD | 0.50 | Confidence above which tag is suggested to user |
| CUSTOM_CATEGORY_THRESHOLD | 0.60 | SBERT cosine similarity for custom category match |
| ERROR_RATE_THRESHOLD | 5% | HTTP 5xx rate above which rollback triggers |
| CORRECTION_RATE_THRESHOLD | 65% | User correction rate above which rollback triggers |
| FEEDBACK_THRESHOLD | 5 | Corrections needed to trigger retraining |

---

## Automated Retraining Pipeline

1. Users upload files → predictions logged to `predictions` table
2. Users correct wrong tags → corrections logged to `feedback` table
3. `retrain-cron` runs hourly, counts corrections since last training run
4. When corrections ≥ `FEEDBACK_THRESHOLD`:
   - `batch_pipeline.py` merges corrections into training data
   - Data quality validation runs (label distribution, text quality, leakage check)
   - `train.py` retrains TF-IDF + LightGBM, evaluates on held-out set
   - Quality gates: core macro F1 ≥ 0.50, per-class F1 ≥ 0.30 (excludes Other)
   - If gates pass → model registered in MLflow as new version, promoted to Staging
5. `monitor.py` runs every 5 minutes:
   - Detects Staging model → runs 5-minute canary (error rate < 5%, p95 < 500ms)
   - If canary passes → hot-reloads model into serving, promotes to Production in MLflow

---

## Monitoring and Rollback

**Grafana dashboard:** "Smart File Tagger — Production" at http://129.114.27.110:3000
- Request rate
- Prediction label distribution
- Latency p50/p95
- Feedback events (accepted vs corrected)

**Automatic rollback triggers:**
- Error rate > 5% over last 10 minutes
- Correction rate > 65% over last 100 feedback events (minimum 10 events)

**Manual rollback:**
```bash
curl -X POST http://129.114.27.110:8000/admin/rollback \
  -H "Content-Type: application/json" \
  -d '{"reason": "manual"}'
```

**Restore after rollback:**
```bash
curl -X POST http://129.114.27.110:8000/admin/restore
```

---

## PostgreSQL Tables

| Table | Purpose |
|---|---|
| predictions | Every prediction logged with file_id, label, confidence, latency |
| feedback | User feedback (accepted, corrected, rejected) |
| custom_categories | SBERT prototype vectors for user-defined categories |
| model_status | Rollback state (consistent across Gunicorn workers) |
| retrain_log | Retraining event history |

---

## Common Commands

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs --tail=20 serving
docker-compose logs --tail=20 monitor
docker-compose logs --tail=20 retrain-cron

# Restart a service (ContainerConfig error fix)
docker-compose rm -f serving
docker-compose up -d serving

# Test prediction
echo "This lecture covers neural networks" > /tmp/test.txt
curl -s -X POST http://localhost:8000/predict \
  -F "file=@/tmp/test.txt" \
  -F "user_id=admin" \
  -F "file_id=test_001" | python3 -m json.tool

# Check recent predictions
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action FROM predictions ORDER BY created_at DESC LIMIT 5;"

# Check feedback
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, feedback_type, corrected_tag FROM feedback ORDER BY created_at DESC LIMIT 5;"
```

---

## Git Workflow

```bash
git add -A
git commit -m "Your message

Assisted by Claude Sonnet 4.5"
git push origin version-1/main
```

All LLM-assisted commits include "Assisted by Claude Sonnet 4.5" per course policy.
