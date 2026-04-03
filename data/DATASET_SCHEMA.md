# Dataset Schema — Smart File Tagger

**Owner:** Viral Dalal (vd2477) — Data Lead  
**Course:** ECE-GY 9183 MLOps, Spring 2026  
**Last updated:** 2026-04-03

---

## ⚠️ Read This Before Touching the Dataset

This document is the single source of truth for the dataset schema. Every team
member's code must conform to it:

- **Viral** — `download_sample.py`, `preprocess.py`, `split.py`, `feedback_export.py` must write these exact column names and types
- **Vedant** — `train.py`, `retrain_job.py`, `evaluate.py` must read these exact column names
- **Krish** — the six label strings below must exactly match what `/predict` returns in `predicted_tag`

If you need to add or change a column, open a PR and update this file at the
same time. Do not silently change column names in code.

---

## Source Dataset

**MIT OpenCourseWare (MIT OCW)**  
URL: https://ocw.mit.edu  
License: Creative Commons BY-NC-SA 4.0  
Coverage: ~2,500+ courses with downloadable PDF materials  

Label inference is done purely from URL path structure — no manual annotation
required. Folder names inside each course provide structurally-guaranteed labels.

---

## The Valid Label Values

These are the only valid values for the `label` column. Exact casing is mandatory.
Every system — Parquet dataset, training script, serving `/predict` response,
TagController.php, and the Accept/Reject UI — must use these exact strings.

```
Lecture Notes
Problem Set
Exam
Syllabus
Reading
Solution
Project
Recitation
Lab
Other
```

### Why these 10 and not 6

MIT OCW courses contain richer material than 6 categories cover. Forcing
everything into 6 buckets poisons training data — a recitation note labeled as
"Lecture Notes" and a project description labeled as "Problem Set" both introduce
noise that degrades model accuracy.

| Added label | Covers |
|---|---|
| `Project` | Project descriptions, rubrics, guidelines (`/pages/projects/`) |
| `Recitation` | Recitation notes, worked examples, review sheets (`/pages/recitations/`) |
| `Lab` | Lab writeups, instructions, data collection tasks (`/pages/labs/`) |
| `Other` | Files that match no rule — see section below |

### The `Other` label

Files whose URL path and filename match no inference rule are assigned
`label = "Other"` and `label_source = "no_rule_matched"` instead of being
silently dropped. Reasons:

1. **Vedant's choice, not Viral's** — whether to include or exclude `Other`
   rows in training is a modeling decision, not a data decision. Viral's job
   is to surface the data; Vedant decides what to train on.
2. **Silent drops are invisible bugs** — if 40% of PDFs are being discarded,
   the manifest should show that, not hide it.
3. **Future categories** — a large `Other` cluster may turn out to be a valid
   new category worth adding. You can't discover that if the rows don't exist.

Vedant can filter Other out in one line if he chooses:

```python
df_train = df[df["label"] != "Other"]
# or equivalently
df_train = df[df["label_source"] != "no_rule_matched"]
```

If you see any variant of any label string (`lecture_notes`, `EXAM`,
`problem set`, `ProblemSet`) in any file, fix it immediately. A casing mismatch
silently breaks the integration between serving and the Nextcloud app.

---

## Label Inference Rules

Labels are inferred from the PDF's URL path + filename. Rules are checked in
order — first match wins.

Labels are inferred from the PDF's URL path + filename. Rules are checked in
order — first match wins. Files that match no rule are assigned `Other` with
`label_source = "no_rule_matched"` — they are never silently dropped.

| URL / filename contains | Assigned label |
|---|---|
| `lecture-notes`, `lecture_notes`, `/lectures/` | `Lecture Notes` |
| `assignments`, `problem-sets`, `psets`, `homework` | `Problem Set` |
| `exams`, `quiz`, `midterm`, `final-exam` | `Exam` |
| `syllabus` | `Syllabus` |
| `readings`, `reading-material` | `Reading` |
| `solutions`, `_sol`, `-sol`, `answer-key` | `Solution` |
| `projects`, `/project/`, `project-description` | `Project` |
| `recitations`, `/recitation/` | `Recitation` |
| `labs`, `/lab/`, `lab-materials` | `Lab` |
| *(no rule matched)* | `Other` |

---

## Full Schema

### Absolutely Necessary Columns

These columns must be present in every row. A missing value in any of these
columns means the row must be dropped before training.

| Column | Type | Example | Why It Is Needed |
|---|---|---|---|
| `doc_id` | string | `"a3f9c12b8e41"` | Stable unique identifier for every document. The feedback loop logs user corrections against this ID. Without it you cannot join user feedback back to training data during retraining. Generated as MD5 of `source_url`. |
| `extracted_text` | string | `"Problem Set 3 — Due Friday..."` | The model input. SBERT encodes this into a 384-dim embedding. LightGBM trains on those embeddings. If this is empty the document must be dropped — do not store zero-text rows. |
| `label` | string | `"Problem Set"` | The training target for LightGBM. Must be one of the ten exact values listed above. |
| `label_source` | string | `"folder_structure"` | How the label was inferred. One of three values: `folder_structure` (from URL path like `/pages/assignments/`), `filename_pattern` (from filename like `_pset3.pdf`), or `no_rule_matched` (assigned `Other`). Vedant uses this to filter to only high-confidence labels for the initial training run. |
| `course_id` | string | `"6.006"` | Critical for preventing data leakage. The batch pipeline must split by `course_id`, not by individual document. Two Problem Sets from the same course have near-identical SBERT embeddings — if they land in both train and eval, accuracy metrics are meaningless. |
| `source_url` | string | `"https://ocw.mit.edu/..."` | Full URL the PDF was downloaded from. Required for reproducibility — if a label looks wrong you trace it back to the original file. Also used in Viral's manifest and dataset versioning. |
| `source` | string | `"mit_ocw"` | Which dataset this row came from. Always `mit_ocw` for everything scraped by this pipeline. Will be `user_feedback` for rows added during the retraining loop. Without this field the baseline dataset and user corrections are indistinguishable in PostgreSQL, which breaks retraining. |
| `ingestion_timestamp` | string (ISO 8601) | `"2026-04-03T10:22:00Z"` | When this document was scraped. The batch pipeline uses this for temporal splits — eval data must never contain documents ingested after the training cutoff date. Required for dataset versioning. |
| `dataset_version` | string | `"v1.0"` | Which version of the dataset this row belongs to. MLflow logs this alongside each training run. Without it, training runs are not reproducible — you cannot tell which data produced which model. |

---

### Good to Have Columns

These columns should be populated where available. Use `null` if the value
cannot be scraped — do not skip the row or fabricate a value.

| Column | Type | Example | Why It Is Needed |
|---|---|---|---|
| `department` | string | `"Electrical Engineering and Computer Science"` | Enables realistic custom category demos (e.g. user creates a "Theory of Computation" category from MIT 18.404 materials). Useful for stratified analysis of model accuracy by department. Scrapeable from course home page. |
| `course_title` | string | `"Introduction to Algorithms"` | Human readable. Makes debugging the dataset far easier. Also used in the data generator to produce realistic filenames and synthetic request payloads. |
| `semester` | string | `"Spring 2020"` | Useful for detecting if model performance varies across time periods. Weak signal for temporal evaluation — older semesters in train, newer in eval. |
| `filename` | string | `"pset3.pdf"` | Needed for debugging. If extracted text looks wrong, you need to know which file to inspect. Also used by Krish's `extractor.py` as a fallback signal for mime type detection when HTTP Content-Type headers are wrong. |
| `char_count` | int64 | `3240` | Filtering signal. Documents with fewer than ~200 characters almost certainly failed text extraction (scanned images, cover pages). Vedant must filter these before training. |
| `word_count` | int64 | `512` | More meaningful than `char_count` for NLP tasks. Also used as an additional LightGBM feature alongside the SBERT embedding — word count is a weak but real signal (Syllabi tend to be short, Lecture Notes long). |
| `file_size_bytes` | int64 | `184320` | Available for free from the HTTP `Content-Length` header before downloading. Weak label signal. Can be used as an additional LightGBM feature. |
| `text_extraction_method` | string | `"pdfminer"` | Either `pdfminer` or `ocr`. If some PDFs are scanned images and Tesseract is used as a fallback, Vedant needs to know — OCR text quality is lower and may need different preprocessing before SBERT encoding. |
| `instructor` | string | `"Erik Demaine"` | Scrapeable from most course pages. Not a model feature but useful for dataset documentation and the custom category demo. Set to `null` if not found on the page. |

---

## What is Explicitly NOT in the Schema

| Field | Why it is excluded |
|---|---|
| `embedding` | Never store SBERT embeddings in Parquet. At 384 floats × 50,000 documents that is 75MB of floats inside the dataset. Embeddings are computed at training time and discarded. If SBERT model version ever changes, stored embeddings are immediately stale. |
| `file_content_b64` | Raw PDF bytes do not belong in Parquet. They belong in object storage. The Parquet stores extracted text only. |
| `split` | Do not bake train/val/test split into the raw dataset. Viral's batch pipeline assigns splits dynamically. If split is hardcoded in the Parquet you cannot re-split when the dataset grows without rewriting the entire file. |
| `label_confidence_score` | Not computable from URL patterns — only a binary high/medium signal is available, which is already captured in `label_source`. Do not invent a fake float. |

---

## Python Schema Reference

Copy this into the top of any script that reads or writes the dataset as a
reference and for runtime validation.

```python
REQUIRED_COLUMNS = {
    "doc_id":               "string",
    "extracted_text":       "string",
    "label":                "string",
    "label_source":         "string",   # "folder_structure" or "filename_pattern"
    "course_id":            "string",
    "source_url":           "string",
    "source":               "string",   # "mit_ocw" or "user_feedback"
    "ingestion_timestamp":  "string",   # ISO 8601 UTC
    "dataset_version":      "string",
}

OPTIONAL_COLUMNS = {
    "department":              "string",
    "course_title":            "string",
    "semester":                "string",
    "filename":                "string",
    "char_count":              "int64",
    "word_count":              "int64",
    "file_size_bytes":         "int64",
    "text_extraction_method":  "string",  # "pdfminer" or "ocr"
    "instructor":              "string",  # nullable
}

VALID_LABELS = {
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
}

VALID_LABEL_SOURCES = {
    "folder_structure",    # inferred from URL path   — high confidence
    "filename_pattern",    # inferred from filename   — medium confidence
    "no_rule_matched",     # no rule matched, label is always "Other"
}

VALID_SOURCES = {"mit_ocw", "user_feedback"}
```

---

## Validation Snippet

Run this after every pipeline run before handing the Parquet to Vedant.

```python
import pandas as pd

def validate_dataset(path: str) -> None:
    df = pd.read_parquet(path)
    errors = []

    # Check required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Check no nulls in required columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            errors.append(f"Null values found in required column: {col}")

    # Check label values
    if "label" in df.columns:
        bad = set(df["label"].unique()) - VALID_LABELS
        if bad:
            errors.append(f"Invalid label values: {bad}")

    # Check label_source values
    if "label_source" in df.columns:
        bad_src = set(df["label_source"].unique()) - VALID_LABEL_SOURCES
        if bad_src:
            errors.append(f"Invalid label_source values: {bad_src}")

    # Check Other rows always have label_source == no_rule_matched
    if "label" in df.columns and "label_source" in df.columns:
        bad_other = df[
            (df["label"] == "Other") &
            (df["label_source"] != "no_rule_matched")
        ]
        if len(bad_other) > 0:
            errors.append(
                f"{len(bad_other)} rows have label=Other "
                f"but label_source != no_rule_matched"
            )

    # Check no empty extracted_text
    if "extracted_text" in df.columns:
        empty = (df["extracted_text"].str.strip() == "").sum()
        if empty > 0:
            errors.append(f"{empty} rows have empty extracted_text — drop them")

    if errors:
        for e in errors:
            print(f"❌  {e}")
        raise ValueError("Dataset validation failed — fix errors before training")
    else:
        label_counts = df["label"].value_counts().to_dict()
        other_pct = round(label_counts.get("Other", 0) / len(df) * 100, 1)
        print(f"✅  Validation passed: {len(df)} rows")
        print(f"    Label distribution: {label_counts}")
        print(f"    Other: {other_pct}% of dataset")
        if other_pct > 30:
            print(
                f"⚠️   Warning: Other is {other_pct}% of dataset. "
                f"Consider reviewing label rules — some valid categories "
                f"may be unmatched."
            )

validate_dataset("sample_data/sample_labeled.parquet")
```

---

## Sample Row (for schema verification)

```json
{
  "doc_id":                 "a3f9c12b8e41",
  "extracted_text":         "Problem Set 3 — Theory of Computation (18.404)\nDue: Friday, March 14 at 11:59pm\n\n1. Show that the language L = { <M> | M is a TM that accepts at least two strings } is undecidable...",
  "label":                  "Problem Set",
  "label_source":           "folder_structure",
  "course_id":              "18.404",
  "source_url":             "https://ocw.mit.edu/courses/18-404j-theory-of-computation-fall-2020/pages/assignments/pset3.pdf",
  "source":                 "mit_ocw",
  "ingestion_timestamp":    "2026-04-03T10:22:00Z",
  "dataset_version":        "v1.0",
  "department":             "Mathematics",
  "course_title":           "Theory of Computation",
  "semester":               "Fall 2020",
  "filename":               "pset3.pdf",
  "char_count":             3240,
  "word_count":             512,
  "file_size_bytes":        184320,
  "text_extraction_method": "pdfminer",
  "instructor":             "Michael Sipser"
}
```

---

## Changelog

| Version | Date | Author | Change |
|---|---|---|---|
| v1.0 | 2026-04-03 | vd2477 | Initial schema definition |
| v1.1 | 2026-04-03 | vd2477 | Expanded labels from 6 to 10 — added Project, Recitation, Lab, Other. Added no_rule_matched as a third label_source value. Updated inference rules table, Python schema reference, and validation snippet. |
