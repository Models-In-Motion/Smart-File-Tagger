# Dataset Schema — OCW Document Tagging

**Version:** v2  
**Last updated:** April 2026  
**Owner (data):** Viral  
**Owner (training):** Vedant  

---

## 1. Label Set

The model is trained on **7 labels**. Three labels originally considered (`Recitation`, `Lab`, `Syllabus`) were dropped after data collection confirmed MIT OCW does not publish those document types as downloadable PDFs at scale.

| Label | Description | Example files |
|---|---|---|
| `Lecture Notes` | Slides, written notes, or transcripts from lectures | `lec1.pdf`, `notes_week3.pdf` |
| `Problem Set` | Homework and problem set handouts | `pset2.pdf`, `hw5.pdf` |
| `Exam` | Midterms, finals, quizzes (questions only) | `midterm.pdf`, `quiz1.pdf` |
| `Solution` | Answer keys for problem sets, exams, or quizzes | `pset2_sol.pdf`, `exam_solutions.pdf` |
| `Reading` | Assigned readings, reference material, open textbooks | `reading_ch4.pdf`, `reference.pdf` |
| `Project` | Project descriptions, reports, or guidelines | `project_spec.pdf`, `final_project.pdf` |
| `Other` | Documents that don't fit any of the above | catch-all fallback |

### Dropped labels and reason

| Label | Rows in v2 dataset | Reason for dropping |
|---|---|---|
| `Syllabus` | 4 | Too few examples — not trainable |
| `Recitation` | 0 | MIT OCW does not publish recitation PDFs |
| `Lab` | 0 | MIT OCW does not publish lab handouts as PDFs |

---

## 2. Parquet Schema

### Canonical dataset — `artifacts/ocw_dataset.parquet`

| Column | Type | Nullable | Description |
|---|---|---|---|
| `doc_id` | string | No | SHA1 hash of source_url — stable unique identifier |
| `extracted_text` | string | No | Full text extracted from the PDF or VTT file |
| `label` | string | No | One of the 7 valid labels above |
| `label_source` | string | No | How the label was derived (see Section 3) |
| `course_id` | string | No | MIT course number e.g. `6.006`, `18.02` |
| `source_url` | string | No | Full OCW URL to the source file |
| `source` | string | No | Always `mit_ocw` for scraped data |
| `ingestion_timestamp` | string | No | UTC ISO-8601 timestamp of ETL run |
| `dataset_version` | string | No | e.g. `v1.0` |
| `department` | string | Yes | Department name derived from course number |
| `course_title` | string | Yes | Full course title from OCW metadata |
| `semester` | string | Yes | e.g. `Fall 2020` |
| `filename` | string | Yes | Original filename e.g. `pset3.pdf` |
| `char_count` | int | Yes | Character count of extracted_text |
| `word_count` | int | Yes | Word count of extracted_text |
| `file_size_bytes` | int | Yes | Size of source PDF in bytes |
| `text_extraction_method` | string | Yes | `pypdf`, `pdftotext`, or `strings` |
| `instructor` | string | Yes | Instructor name(s) from OCW metadata |

### Training splits — `artifacts/versions/v2/`

| File | Rows | Courses | Notes |
|---|---|---|---|
| `train.parquet` | 24,174 | 168 | Excludes `Other` label (dropped by batch pipeline) |
| `eval.parquet` | 7,752 | 44 | No course overlap with train — leakage-safe |
| `split_metadata.json` | — | — | Label distributions, course lists, split config |

**Course overlap between train and eval: 0** (split is course-level, not document-level)

---

## 3. Label Sources

| `label_source` value | Meaning |
|---|---|
| `folder_structure` | Label derived from OCW `learning_resource_types` in `data.json`, or from the folder path in legacy courses |
| `filename_pattern` | Label derived from filename stem pattern (e.g. ends with `sol`, `prob`, `sum`) |
| `no_rule_matched` | No rule fired — document becomes `Other` and is dropped before training |

---

## 4. Data Pipeline Summary

```
scrape_ocw.py           →  data/courses/{slug}/          (264 course folders)
build_ocw_dataset.py    →  artifacts/ocw_dataset.parquet  (7,057 rows, 238 courses)
synthetic_expansion.py  →  artifacts/ocw_dataset_expanded.parquet  (42,337 rows, 5x augmentation)
data_generator.py       →  artifacts/production_feedback.jsonl     (2,000 feedback events)
batch_pipeline.py       →  artifacts/versions/v2/train.parquet + eval.parquet
```

---

## 5. Label Distribution (v2)

### Raw dataset (`ocw_dataset.parquet`)

| Label | Count | % |
|---|---|---|
| Lecture Notes | 2,806 | 39.8% |
| Other | 1,735 | 24.6% |
| Problem Set | 1,443 | 20.4% |
| Exam | 379 | 5.4% |
| Reading | 307 | 4.4% |
| Project | 263 | 3.7% |
| Solution | 120 | 1.7% |
| Syllabus | 4 | 0.1% |
| **Total** | **7,057** | |

### Train split (post-augmentation, v2)

| Label | Train | Eval |
|---|---|---|
| Lecture Notes | 13,320 | 3,515 |
| Other | 7,710 | 2,700 |
| Problem Set | 6,577 | 2,076 |
| Exam | 1,866 | 408 |
| Reading | 1,308 | 534 |
| Project | 1,464 | 114 |
| Solution | 606 | 114 |

---

## 6. What Vedant Needs to Update (`train.yaml`)

- Set `num_labels: 7`
- Set label list to: `["Lecture Notes", "Problem Set", "Exam", "Solution", "Reading", "Project", "Other"]`
- `Other` is a real class — the model must be trained on it and can predict it at inference time
- Remove `Recitation`, `Lab`, `Syllabus` from any label encoding
- Input column: `extracted_text`
- Target column: `label`
- Train file: `artifacts/versions/v2/train.parquet`
- Eval file: `artifacts/versions/v2/eval.parquet`

---

## 7. Known Limitations

1. **Class imbalance** — Lecture Notes dominates (~40%). Recommend weighted loss or oversampling for minority classes (Solution, Project).
2. **Other is a real trained class** — 24.6% of raw data is `Other` and it is included in train/eval. The model can predict `Other` at inference time as a catch-all for documents that don't confidently match any specific category.
3. **Syllabus dropped** — only 4 examples found across 238 courses. Not trainable.
4. **Text backend** — extracted with `strings` locally (fallback). Docker/Chameleon uses `pypdf` which produces cleaner text. Retrain on Chameleon output for best quality.
5. **No Recitation or Lab examples** — MIT OCW does not expose these as PDFs in the formats we ingest.
