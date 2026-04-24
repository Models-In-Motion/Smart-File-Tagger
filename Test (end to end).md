# Smart File Tagger — Demo Notes
### April 24, 2026

---

## SETUP (before demo)

```bash
mkdir -p ~/Documents/mlops-demo
python3 ~/generate_demo_files.py
```

```bash
cd ~/smart-file-tagger
docker-compose ps
```

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "TRUNCATE TABLE predictions; TRUNCATE TABLE feedback;"
```

Open these tabs:
- Nextcloud:  http://129.114.27.110:8080  (admin / admin)
- Grafana:    http://129.114.27.110:3000  (admin / admin)
- MLflow:     http://129.114.27.110:5001
- FastAPI:    http://129.114.27.110:8000/docs

---

## SCENE 1 — Auto-Apply Tag

The system automatically classifies a file and applies a tag without any user
action when confidence is above 75%.

**Upload:** `~/Documents/mlops-demo/auto_apply/lecture_neural_networks.txt`

Wait 2–3 seconds, click the file, check the Tags section in the right sidebar —
**Lecture Notes** should be applied automatically.

To see the prediction log:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action, latency_ms FROM predictions ORDER BY created_at DESC LIMIT 3;"
```

Notes: `action = auto_apply`, confidence >= 0.75. Latency should be 5–15ms — that's the TF-IDF model.

---

## SCENE 2 — Suggested Tag (Accept)

For borderline predictions between 50–75% confidence, the system surfaces a
suggestion and lets the user confirm rather than auto-applying.

**Upload:** `~/Documents/mlops-demo/suggest/ambiguous_notes_or_reading.txt`

A suggestion banner appears bottom-right. Click **Accept**.

To see the prediction:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action FROM predictions ORDER BY created_at DESC LIMIT 3;"
```

To see the feedback entry:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, feedback_type, corrected_tag FROM feedback ORDER BY created_at DESC LIMIT 3;"
```

Notes: `action = suggest` in predictions, `feedback_type = accepted` in feedback.

---

## SCENE 3 — Manual Tag Correction

Users can always override the system. The correction is logged as feedback
and feeds into retraining.

**Upload:** `~/Documents/mlops-demo/manual_correction/quiz_questions.txt`

The system may auto-apply the wrong tag (e.g. Other or Problem Set). Click
the file to open the sidebar, find the **Manual Tag** panel bottom-right,
select **Exam** from the dropdown, click **Apply Tag**.

To see the correction logged:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, corrected_tag, feedback_type FROM feedback ORDER BY created_at DESC LIMIT 5;"
```

Notes: `feedback_type = corrected`, `corrected_tag = Exam`.

---

## SCENE 4 — Custom Category

Users can define their own categories beyond the 5 built-in labels. The system
uses SBERT sentence embeddings and cosine similarity against a prototype vector
built from the user's example texts. Custom categories are checked first,
before the TF-IDF classifier runs.

**Open:** http://129.114.27.110:8000/docs → `POST /register-category` → Try it out

```json
{
  "user_id": "admin",
  "category_name": "Research Proposal",
  "examples": [
    "This paper proposes a novel method for improving GAN training stability using Wasserstein distance",
    "We propose experiments to evaluate transformer architectures on low-resource translation tasks",
    "Our research investigates the application of reinforcement learning to robotic manipulation"
  ]
}
```

Execute — should return 200.

**Upload:** `~/Documents/mlops-demo/custom_category/research_proposal_gans.txt`

To see the custom category stored:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT user_id, category_name, example_count FROM custom_categories;"
```

To see the prediction with custom category:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action, category_type FROM predictions ORDER BY created_at DESC LIMIT 3;"
```

Notes: `category_type = custom` confirms it matched via SBERT, not TF-IDF.

---

## SCENE 5 — Feedback Loop + Automatic Retraining

The system monitors corrections. When enough accumulate since the last training
run, it automatically merges them with the original dataset and retrains.

### Step 1 — Clear retrain_log so the counter starts from zero

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "TRUNCATE TABLE retrain_log;"
```

### Step 2 — Upload and correct all 5 files

Upload each file from `~/Documents/mlops-demo/feedback_loop/` to Nextcloud
one by one. After each upload, click the file, use the **Manual Tag** panel
to apply the correction below:

| File | Correct to |
|---|---|
| borderline_1.txt | Lecture Notes |
| borderline_2.txt | Reading |
| borderline_3.txt | Problem Set |
| borderline_4.txt | Exam |
| borderline_5.txt | Reading |

### Step 3 — Confirm corrections are in the DB

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT feedback_type, corrected_tag, created_at FROM feedback ORDER BY created_at DESC LIMIT 5;"
```

Should show 5 rows, all `feedback_type = corrected`.

### Step 4 — Manually trigger the cron (don't wait 1 hour)

```bash
docker exec smart-file-tagger_retrain-cron_1 python3 /app/retrain_trigger.py
```

Output to expect:
- `Unchecked corrections: 5`
- `Threshold reached (5 >= 5) — triggering retraining`
- `Built 5 corrected-feedback training rows`
- `Train rows after corrected feedback append: 15967 -> 15972`
- Data quality validation report — all 5 labels should show OK
- `Training classifier...` — takes ~2 minutes, don't kill it
- F1 scores per class printed at end
- MLflow run registered

### Step 5 — Confirm new dataset was built

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT triggered_at, reason FROM retrain_log ORDER BY triggered_at DESC LIMIT 3;"
```

### Step 6 — Confirm feedback breakdown

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type;"
```

### If something goes wrong in Scene 5

**Trigger shows 0 corrections:**
The retrain_log likely has a recent entry blocking the count. Run Step 1 again and retry Step 4.

**Training fails or output cuts off:**
```bash
docker exec smart-file-tagger_retrain-cron_1 python3 /app/retrain_trigger.py 2>&1 | tail -40
```

**retrain-cron container not found:**
```bash
docker-compose ps
docker-compose up -d retrain-cron
```

**Manual tag panel didn't appear:**
Click a different file first, then come back — the JS polls every 1 second
starting 3 seconds after page load.

---

## SCENE 6 — MLflow

Every training run is tracked automatically — data version, hyperparameters,
metrics, and the registered model artifact.

**URL:** http://129.114.27.110:5001

- Experiments → `nextcloud-doc-tagger-training` → click latest run
- To see: `macro_f1`, per-class F1 scores, `train_samples`, `eval_samples`
- Models → `smart-tagger` → version 2 in **Production**

Notes: Model only reaches Production after passing quality gates in `train.py`
(macro F1 >= 0.60, per-class >= 0.40) AND passing the 5-minute canary check
in `monitor.py` (error rate < 5%, p95 < 500ms). The full automated promotion
flow works — demonstrated earlier in this session.

---

## SCENE 7 — Grafana

**URL:** http://129.114.27.110:3000 (admin/admin)

Dashboards → **Smart File Tagger — Production**

- **Panel 1 — Request Rate:** spikes from the uploads done during the demo
- **Panel 2 — Label Distribution:** breakdown of which labels were predicted
- **Panel 3 — Latency p50/p95:** TF-IDF inference is 5–15ms by design
- **Panel 4 — Feedback Events:** accepted vs corrected counts

Notes: Correction rate is monitored continuously. If it exceeds 65% over the
last 100 feedbacks, `monitor.py` auto-rolls back to the previous model. The
rollback state is stored in PostgreSQL, not in memory, so it's consistent
across all 4 Gunicorn workers.

---

## SCENE 8 — Rollback (if asked)

```bash
curl -s -X POST http://localhost:8000/admin/rollback \
  -H "Content-Type: application/json" \
  -d '{"reason": "demo rollback"}' | python3 -m json.tool
```

```bash
curl -s http://localhost:8000/admin/status | python3 -m json.tool
```

```bash
curl -s -X POST http://localhost:8000/admin/restore | python3 -m json.tool
```

To see rollback state in DB:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT key, value, updated_at FROM model_status;"
```

---

## QUICK REFERENCE — All DB Queries

All recent predictions:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, predicted_tag, confidence, action, latency_ms FROM predictions ORDER BY created_at DESC LIMIT 10;"
```

All feedback:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT file_id, feedback_type, predicted_tag, corrected_tag, created_at FROM feedback ORDER BY created_at DESC LIMIT 10;"
```

Correction rate:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT COUNT(*) FILTER (WHERE feedback_type='corrected') AS corrections, COUNT(*) AS total, ROUND(100.0 * COUNT(*) FILTER (WHERE feedback_type='corrected') / NULLIF(COUNT(*),0), 1) AS pct FROM feedback;"
```

Custom categories:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT user_id, category_name, example_count FROM custom_categories;"
```

Retrain log:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT triggered_at, reason FROM retrain_log ORDER BY triggered_at DESC LIMIT 5;"
```

Rollback state:

```bash
docker exec smart-file-tagger_postgres_1 \
  psql -U tagger -d tagger -c \
  "SELECT key, value, updated_at FROM model_status;"
```
