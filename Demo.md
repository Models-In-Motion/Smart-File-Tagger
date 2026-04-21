# Smart File Tagger — Demo Script
## April 20, 2026 | NYU ECE-GY 9183 MLOps

---

## SETUP BEFORE RECORDING

Open these in your browser tabs:
- Tab 1: Nextcloud → http://129.114.27.110:8080 (admin/admin)
- Tab 2: Grafana → http://129.114.27.110:3000 (admin/admin)
- Tab 3: MLflow → http://129.114.27.110:5001
- Tab 4: GitHub repo → https://github.com/Models-In-Motion/Smart-File-Tagger

SSH into Chameleon in your terminal:
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.27.110
cd Smart-File-Tagger
```

---

## SECTION 1 — Show the integrated system (Joint requirement)

### 1.1 Show all services running as one unified system

**Command:**
```bash
docker-compose ps
```

**Expected output:**
```
smart-file-tagger_grafana_1        Up    0.0.0.0:3000->3000/tcp
smart-file-tagger_mlflow_1         Up    0.0.0.0:5001->5000/tcp
smart-file-tagger_monitor_1        Up
smart-file-tagger_nextcloud_1      Up    0.0.0.0:8080->80/tcp
smart-file-tagger_postgres_1       Up    0.0.0.0:5432->5432/tcp
smart-file-tagger_prometheus_1     Up    0.0.0.0:9090->9090/tcp
smart-file-tagger_retrain-cron_1   Up
smart-file-tagger_serving_1        Up    0.0.0.0:8000->8000/tcp
```

**Notes:** "We have a single Docker Compose stack with 8 services — one PostgreSQL, one MLflow, one monitoring stack. Nothing is duplicated."

### 1.2 Show repo structure

**Switch to GitHub tab** and show:
- `data/` — Viral's data pipeline
- `serving/` — Krish's serving layer
- `training/` — Vedant's training pipeline
- `nextcloud-app/` — PHP plugin
- `infra/` — prometheus, grafana configs
- `docker-compose.yml` — single source of truth
- `docs/safeguarding.md` — safeguarding plan

---

## SECTION 2 — Complementary ML feature in Nextcloud (Joint requirement)

### 2.1 Show Nextcloud is running

**Switch to Nextcloud tab** → show Files page

### 2.2 Upload a file and show automatic tagging

**Command:**
```bash
cat > /tmp/demo_lecture.txt << 'EOF'
Lecture 12: Introduction to Transformers and Attention Mechanisms
In this lecture we cover the self-attention mechanism, multi-head attention,
positional encoding, and the transformer architecture.
We also discuss BERT, GPT, and their applications in NLP tasks.
EOF

curl -s -T /tmp/demo_lecture.txt \
  http://admin:admin@129.114.27.110:8080/remote.php/dav/files/admin/lecture_12.txt
```

**Expected:** No output (silent upload)

**Then immediately check:**
```bash
sleep 5
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action, created_at FROM predictions ORDER BY created_at DESC LIMIT 3;"
```

**Expected output:**
```
 file_id | predicted_tag | confidence |   action   
---------+---------------+------------+------------
 XXX     | Lecture Notes |   0.96+    | auto_apply
```

**Switch to Nextcloud tab** → refresh → click on `lecture_12.txt` → show the `Lecture Notes` tag in the sidebar.

**Notes:** "The ML feature is live inside Nextcloud. When a file is uploaded, the Flow webhook fires, our PHP controller fetches the file via WebDAV, sends it to FastAPI, gets a prediction, and applies the tag — all automatically, no human intervention."

### 2.3 Upload different file types

**Command:**
```bash
cat > /tmp/demo_exam.txt << 'EOF'
Midterm Examination — Deep Learning
Time: 90 minutes. Answer all questions.
Question 1: Derive backpropagation for a 3-layer network.
Question 2: Explain dropout regularization.
Question 3: Compare batch normalization vs layer normalization.
EOF

curl -s -T /tmp/demo_exam.txt \
  http://admin:admin@129.114.27.110:8080/remote.php/dav/files/admin/midterm.txt

cat > /tmp/demo_pset.txt << 'EOF'
Problem Set 4 — Due Friday
Question 1: Solve the eigenvalue decomposition for matrix A.
Question 2: Implement gradient descent from scratch.
Question 3: Prove the convergence of stochastic gradient descent.
EOF

curl -s -T /tmp/demo_pset.txt \
  http://admin:admin@129.114.27.110:8080/remote.php/dav/files/admin/pset4.txt
```

**Then check:**
```bash
sleep 5
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action FROM predictions ORDER BY created_at DESC LIMIT 5;"
```

**Expected:** Exam → `Exam`, Problem set → `Problem Set`, both with high confidence.

**Switch to Nextcloud** → show tags on both files.

---

## SECTION 3 — Feedback capture (Joint requirement)

### 3.1 Show feedback being saved

**Command:**
```bash
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "demo_001",
    "user_id": "admin",
    "predicted_tag": "Lecture Notes",
    "confidence": 0.96,
    "action_taken": "auto_apply",
    "feedback_type": "accepted",
    "model_version": "tfidf_lightgbm_bundle"
  }' | python3 -m json.tool
```

**Expected:**
```json
{
    "success": true,
    "message": "Feedback saved"
}
```

**Command (show correction):**
```bash
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "demo_002",
    "user_id": "admin",
    "predicted_tag": "Lecture Notes",
    "confidence": 0.72,
    "action_taken": "suggest",
    "feedback_type": "corrected",
    "corrected_tag": "Reading",
    "model_version": "tfidf_lightgbm_bundle"
  }' | python3 -m json.tool
```

**Then show it's in PostgreSQL:**
```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, feedback_type, corrected_tag, created_at FROM feedback ORDER BY created_at DESC LIMIT 5;"
```

**Notes:** "Every user interaction is captured — accepts, rejects, and corrections. Corrections include the user's correct label, which feeds back into retraining."

---

## SECTION 4 — Retraining pipeline (Joint requirement)

### 4.1 Show retrain-cron is running

**Command:**
```bash
docker logs smart-file-tagger_retrain-cron_1 --tail=10
```

**Expected:**
```
[retrain-cron] Checking feedback trigger at ...
[retrain_trigger] Unchecked corrections: X
[retrain_trigger] Below threshold, no retraining needed.
[retrain-cron] Sleeping 1h.
```

**Notes:** "The retraining cron job runs every hour. It checks if 50+ corrections have accumulated. If yes, it automatically triggers train.py, runs quality gates, and registers the model in MLflow."

### 4.2 Show training config and quality gates

**Command:**
```bash
cat training/configs/train_docker.yaml
```

**Notes:** "Training uses LightGBM with TF-IDF features. Quality gates require macro F1 ≥ 0.60 and per-class F1 ≥ 0.40 for core labels before a model is registered."

---

## SECTION 5 — Monitoring (Krish — Serving role requirement)

### 5.1 Panel 1 — Request Rate

**Generate traffic first:**
```bash
for i in $(seq 1 15); do
  echo "Lecture $i covers neural networks and deep learning" > /tmp/t.txt
  curl -s -X POST http://localhost:8000/predict \
    -F "file=@/tmp/t.txt" \
    -F "user_id=admin" \
    -F "file_id=load_$i" > /dev/null
done
echo "Done"
```

**Switch to Grafana tab** → show Panel 1 (Request rate).

**Notes:** "This panel shows real-time request rate per second across all API endpoints. You can see the spike when we just sent 15 requests. The serving layer handles these with 4 Gunicorn workers — we benchmarked p50 latency at under 25ms on this m1.xlarge instance."

---

### 5.2 Panel 2 — Prediction Label Distribution

**Generate diverse predictions:**
```bash
# Lecture Notes
for i in 1 2 3; do
  echo "Lecture $i: neural networks backpropagation algorithms deep learning" > /tmp/t.txt
  curl -s -T /tmp/t.txt http://admin:admin@129.114.27.110:8080/remote.php/dav/files/admin/lec_$i.txt
done

# Problem Set
for i in 1 2; do
  echo "Problem set $i homework questions matrix eigenvalues linear algebra" > /tmp/t.txt
  curl -s -T /tmp/t.txt http://admin:admin@129.114.27.110:8080/remote.php/dav/files/admin/pset_$i.txt
done

# Exam
echo "Midterm exam questions chapters backpropagation neural networks" > /tmp/t.txt
curl -s -T /tmp/t.txt http://admin:admin@129.114.27.110:8080/remote.php/dav/files/admin/exam_demo.txt

sleep 10
```

**Switch to Grafana** → show Panel 2 (Prediction label distribution).

**Notes:** "This panel shows the distribution of predicted labels over time. Right now we see Lecture Notes, Problem Set, and Exam being predicted. This is important for fairness monitoring — if one label starts dominating (like the model always predicting Lecture Notes), that's a signal the model is drifting toward the majority class. Our per-class F1 quality gates prevent this during training, and this panel catches it in production."

---

### 5.3 Panel 3 — Latency p50 and p95

**No extra commands needed — traffic from previous steps is enough.**

**Switch to Grafana** → show Panel 3 (Latency p50 and p95).

**Notes:** "This panel shows p50 and p95 latency in milliseconds. The green line is p50 — median latency — running at around 15-25ms. The yellow line is p95 — 95th percentile — which stays under 80ms. This is the TF-IDF model advantage: it's much faster than our earlier SBERT-based approach which had p50 around 190ms. Our monitor.py automatically rolls back the model if p95 exceeds 500ms during a canary check."

---

### 5.4 Panel 4 — Feedback Events

**Get recent file IDs:**
```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag FROM predictions ORDER BY created_at DESC LIMIT 6;"
```

**Send feedback using those IDs (replace IDs with actual values):**
```bash
# 3 accepted — replace FILE_ID_1, FILE_ID_2, FILE_ID_3 with real IDs from above
for id in FILE_ID_1 FILE_ID_2 FILE_ID_3; do
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"file_id\":\"$id\",\"user_id\":\"admin\",\"predicted_tag\":\"Lecture Notes\",\"confidence\":0.96,\"action_taken\":\"auto_apply\",\"feedback_type\":\"accepted\",\"model_version\":\"tfidf_lightgbm_bundle\"}" > /dev/null
done

# 1 correction — replace FILE_ID_4 with real ID
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"file_id":"FILE_ID_4","user_id":"admin","predicted_tag":"Lecture Notes","confidence":0.72,"action_taken":"auto_apply","feedback_type":"corrected","corrected_tag":"Reading","model_version":"tfidf_lightgbm_bundle"}' \
  | python3 -m json.tool
```

**Switch to Grafana** → show Panel 4 (Feedback events).

**Notes:** "This panel tracks user feedback — accepted predictions in green, corrections in yellow. The correction rate is what drives our automated rollback. Right now correction rate is low and healthy. If it crosses 40% — meaning users are correcting more than 40% of predictions — monitor.py automatically rolls back the model to prevent bad predictions from propagating."

---

### 5.5 Show automated rollback

**Command:**
```bash
# Show current healthy state
curl -s http://localhost:8000/admin/status | python3 -m json.tool
```

**Expected:**
```json
{
    "model_version": "tfidf_lightgbm_bundle",
    "model_mode": "bundle_tfidf",
    "rolled_back": false,
    "rollback_reason": null
}
```

**Trigger rollback:**
```bash
curl -s -X POST http://localhost:8000/admin/rollback \
  -H "Content-Type: application/json" \
  -d '{"reason": "correction_rate=50% exceeds threshold=40%"}' \
  | python3 -m json.tool
```

**Expected:**
```json
{
    "success": true,
    "rolled_back": true,
    "reason": "correction_rate=50% exceeds threshold=40%"
}
```

**Show predictions return null during rollback:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@/tmp/demo_lecture.txt" \
  -F "user_id=admin" \
  -F "file_id=rollback_test" | python3 -m json.tool
```

**Expected:**
```json
{
    "predicted_tag": null,
    "action": "no_tag"
}
```

**Notes:** "During rollback, predictions return null — no tag is applied. This is the fail-safe: users keep using Nextcloud normally, files just don't get tagged until the model is restored. This is better than serving bad predictions."

**Restore:**
```bash
curl -s -X POST http://localhost:8000/admin/restore | python3 -m json.tool
```

**Notes:** "Once the team investigates and fixes the issue — or a new better model passes quality gates and gets promoted — we restore normal operation. The rollback state is stored in PostgreSQL so it persists across container restarts and is consistent across all 4 Gunicorn workers."

---

### 5.6 Show monitor logs

**Command:**
```bash
docker logs smart-file-tagger_monitor_1 --tail=20
```

**Expected:**
```
Monitor check at 2026-04-21T...
Error rate (10min): 0.0%
Correction rate (last 100): 0.0%
Rollback check passed — model is healthy
No models in Staging — skipping promotion check
Sleeping 300s until next check...
```

**Notes:** "The monitor runs every 5 minutes automatically — no human needed. It checks error rate from Prometheus and correction rate from PostgreSQL, then decides whether to roll back or promote."

---

## SECTION 6 — MLflow tracking (Vedant — Training role requirement)

### 6.1 Show MLflow UI

**Switch to MLflow tab** → `http://129.114.27.110:5001`

Show:
- Experiment `nextcloud-doc-tagger-training`
- Training runs with per-class F1 metrics
- Models tab → `smart-tagger` registered model

**Command to show from terminal:**
```bash
curl -s http://localhost:5001/api/2.0/mlflow/experiments/list | \
  python3 -c "import sys,json; d=json.load(sys.stdin); [print(e['name']) for e in d['experiments']]"
```

**Notes:** "Every training run is tracked in MLflow with per-class F1 scores. Models only get registered if they pass quality gates — macro F1 ≥ 0.60 and every core label F1 ≥ 0.40."

---

## SECTION 7 — Data quality (Viral — Data role requirement)

### 7.1 Show data validation

**Command:**
```bash
ls data/artifacts/versions/
```

**Show the train/eval splits exist:**
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/artifacts/versions/v6/train_llm.parquet')
print('Train shape:', df.shape)
print('Labels:', df['llm_label_merged'].value_counts().to_dict())
"
```

### 7.2 Show drift monitoring

**Command:**
```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT predicted_tag, COUNT(*) as count FROM predictions GROUP BY predicted_tag ORDER BY count DESC;"
```

**Notes:** "Viral's drift monitoring checks prediction distribution hourly and writes to a drift_metrics table. If distribution shifts significantly from training data, it flags for retraining."

---

## SECTION 8 — Custom categories (Bonus feature)

### 8.1 Register a custom category

**Command:**
```bash
curl -s -X POST http://localhost:8000/register-category \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "admin",
    "category_name": "Lab Report",
    "example_texts": [
      "Lab report experiment 1. Materials and methods. Results show significant findings. Conclusion based on experimental data.",
      "Lab report experiment 2. Hypothesis testing. Statistical analysis of results. Discussion of experimental errors.",
      "Lab report experiment 3. Procedure and observations. Data collection and analysis. Conclusion and future work.",
      "Lab report experiment 4. Introduction and background. Experimental setup. Results and discussion.",
      "Lab report experiment 5. Abstract. Methods and materials. Results. Conclusion and references."
    ]
  }' | python3 -m json.tool
```

**Expected:**
```json
{
    "success": true,
    "message": "Category 'Lab Report' created with 5 examples"
}
```

### 8.2 Test the custom category

**Command:**
```bash
cat > /tmp/lab_report.txt << 'EOF'
Lab Report: Enzyme Kinetics Experiment
Materials and Methods: We used Bradford assay to measure protein concentration.
Results: The Michaelis constant Km was determined to be 2.3 mM.
Discussion: Our results confirm the expected enzymatic behavior.
Conclusion: The experiment successfully demonstrated enzyme kinetics.
EOF

curl -s -X POST http://localhost:8000/predict \
  -F "file=@/tmp/lab_report.txt" \
  -F "user_id=admin" \
  -F "file_id=lab_demo" | python3 -m json.tool
```

**Expected:**
```json
{
    "predicted_tag": "Lab Report",
    "confidence": 0.98+,
    "category_type": "custom",
    "action": "auto_apply"
}
```

**Notes:** "Users can create custom categories with just 3-5 examples. The system uses SBERT embeddings and cosine similarity — no retraining needed. The custom category takes priority over the base model."

---

## SECTION 9 — Safeguarding plan

**Switch to GitHub tab** → show `docs/safeguarding.md`

**Notes:** "We implemented concrete mechanisms for all 6 principles:
- **Fairness:** Per-class F1 gates prevent label bias
- **Explainability:** Confidence scores and top-3 predictions on every response
- **Transparency:** MLflow tracks every model version, training run, and metrics
- **Privacy:** File content is never stored — only the predicted label
- **Accountability:** Every prediction logged with model version ID
- **Robustness:** Fail-open design, automated rollback, quality gates"

---

## SECTION 10 — End-to-end automation summary

**Command (show everything is automated):**
```bash
docker-compose ps | grep -E "monitor|retrain|cron"
```

**Notes:** "The full pipeline runs without human intervention:
1. Files uploaded → automatically tagged in Nextcloud
2. User feedback → stored in PostgreSQL
3. When 50+ corrections accumulate → retraining triggers automatically
4. New model passes quality gates → registered in MLflow
5. Monitor promotes to production after canary check
6. If error rate > 5% or correction rate > 40% → automatic rollback"

---

## CLEANUP COMMANDS (run after demo if needed)

```bash
# Clear test data
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "DELETE FROM feedback WHERE user_id='admin' AND created_at > NOW() - INTERVAL '1 hour';"

# Restore if rolled back
curl -s -X POST http://localhost:8000/admin/restore

# Check all services still healthy
docker-compose ps
```