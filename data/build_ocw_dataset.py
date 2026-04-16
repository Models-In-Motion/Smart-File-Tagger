#!/usr/bin/env python3
"""
Build an OCW dataset for ML Systems project work.

Supports both:
1) Modern OCW offline dumps (course/data.json + resources/*/data.json + static_resources/*)
2) Legacy OCW dumps with nested `contents/` folders.

Output schema (required columns):
    doc_id, extracted_text, label, label_source, course_id,
    source_url, source, ingestion_timestamp, dataset_version

Optional columns (null if unavailable):
    department, course_title, semester, filename,
    char_count, word_count, file_size_bytes,
    text_extraction_method, instructor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Internal label → canonical label (team-agreed valid values)
# ---------------------------------------------------------------------------
INTERNAL_TO_LABEL = {
    "Lecture Notes":        "Lecture Notes",
    "Lecture Transcript":   "Lecture Notes",
    "Recitation Transcript":"Recitation",
    "Problem Set":          "Problem Set",
    "Problem Set Solution": "Solution",
    "Exam":                 "Exam",
    "Exam Solution":        "Solution",
    "Reading / Reference":  "Reading",
    "Instructor Insights":  "Other",
    "Syllabus":             "Syllabus",
    "Assignment":           "Problem Set",
    "Project":              "Project",
    "Quiz":                 "Exam",
    "Quiz Solution":        "Solution",
}

VALID_LABELS = {
    "Lecture Notes", "Problem Set", "Exam", "Syllabus",
    "Reading", "Solution", "Project", "Recitation", "Lab", "Other",
}

VALID_LABEL_SOURCES = {"folder_structure", "filename_pattern", "no_rule_matched"}
VALID_SOURCES       = {"mit_ocw", "user_feedback"}

# learning_resource_types from data.json → internal label
LRT_MAP = {
    "Problem Sets":          "Problem Set",
    "Problem Set Solutions": "Problem Set Solution",
    "Exams":                 "Exam",
    "Exam Solutions":        "Exam Solution",
    "Lecture Notes":         "Lecture Notes",
    "Lecture Videos":        "Lecture Transcript",
    "Recitation Videos":     "Recitation Transcript",
    "Readings":              "Reading / Reference",
    "Instructor Insights":   "Instructor Insights",
    "Assignments":           "Assignment",
    "Projects":              "Project",
    "Quizzes":               "Quiz",
    "Quiz Solutions":        "Quiz Solution",
    "Open Textbooks":        "Reading / Reference",
}

# MIT course number prefix → department name
DEPT_MAP = {
    "1":  "Civil and Environmental Engineering",
    "2":  "Mechanical Engineering",
    "3":  "Materials Science and Engineering",
    "4":  "Architecture",
    "5":  "Chemistry",
    "6":  "Electrical Engineering and Computer Science",
    "7":  "Biology",
    "8":  "Physics",
    "9":  "Brain and Cognitive Sciences",
    "10": "Chemical Engineering",
    "11": "Urban Studies and Planning",
    "12": "Earth, Atmospheric and Planetary Sciences",
    "14": "Economics",
    "15": "Management",
    "16": "Aeronautics and Astronautics",
    "17": "Political Science",
    "18": "Mathematics",
    "20": "Biological Engineering",
    "21": "Humanities",
    "22": "Nuclear Science and Engineering",
    "24": "Philosophy",
}

OCW_BASE = "https://ocw.mit.edu"

TIME_LINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}")
TAG_RE       = re.compile(r"<[^>]+>")
SPACES_RE    = re.compile(r"\s+")
NONWORD_RE   = re.compile(r"\W+")
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
HASH_NAME_RE  = re.compile(r"^[a-f0-9]{32}")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

@dataclass
class TextExtractor:
    backend: str

    @staticmethod
    def autodetect(prefer_pypdf: bool = True) -> "TextExtractor":
        if prefer_pypdf:
            try:
                import pypdf  # noqa: F401
                return TextExtractor("pypdf")
            except Exception:
                pass
        if _command_exists("pdftotext"):
            return TextExtractor("pdftotext")
        if _command_exists("strings"):
            return TextExtractor("strings")
        return TextExtractor("none")

    def extract_pdf(self, path: Path) -> str:
        if self.backend == "pypdf":
            return _extract_pdf_pypdf(path)
        if self.backend == "pdftotext":
            return _extract_pdf_pdftotext(path)
        if self.backend == "strings":
            return _extract_pdf_strings(path)
        return ""


def _command_exists(cmd: str) -> bool:
    try:
        p = subprocess.run(["/bin/zsh", "-lc", f"command -v {cmd}"], capture_output=True, text=True)
        return p.returncode == 0 and bool(p.stdout.strip())
    except Exception:
        return False


def _extract_pdf_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def _extract_pdf_pdftotext(path: Path) -> str:
    try:
        proc = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True)
        return (proc.stdout or "").strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _extract_pdf_strings(path: Path) -> str:
    try:
        proc = subprocess.run(["strings", "-n", "6", str(path)], capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        bad = ("%PDF", "endobj", "xref", "obj", "stream", "endstream",
               "startxref", "/Type", "/Length", "FlateDecode")
        kept = []
        for line in (proc.stdout or "").splitlines():
            s = line.strip()
            if len(s) < 20 or any(t in s for t in bad):
                continue
            alpha = sum(ch.isalpha() for ch in s)
            if alpha < 10 or alpha / max(len(s), 1) < 0.35:
                continue
            kept.append(s)
        return "\n".join(kept).strip()
    except Exception:
        return ""


def extract_vtt_text(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s or s.upper().startswith("WEBVTT") or TIME_LINE_RE.match(s) or s.isdigit():
            continue
        s = TAG_RE.sub(" ", s).replace("&nbsp;", " ").replace("&amp;", "&")
        kept.append(s)
    return SPACES_RE.sub(" ", " ".join(kept)).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def join_nonempty(values: Iterable[str]) -> str:
    return " | ".join(v for v in values if v)


def normalize_text(text: str) -> str:
    return SPACES_RE.sub(" ", NONWORD_RE.sub(" ", text.lower().strip()))


def make_doc_id(source_url: str) -> str:
    return hashlib.md5(source_url.encode("utf-8", errors="ignore")).hexdigest()[:12]


def dept_from_course_id(course_id: Optional[str]) -> Optional[str]:
    if not course_id:
        return None
    prefix = re.match(r"^(\d+)", course_id)
    if prefix:
        return DEPT_MAP.get(prefix.group(1))
    return None


def map_lrt_to_internal(lr_types: list[str]) -> Optional[str]:
    for t in lr_types:
        if t in LRT_MAP:
            return LRT_MAP[t]
    return None


def internal_to_canonical(internal: Optional[str]) -> str:
    if internal is None:
        return "Other"
    return INTERNAL_TO_LABEL.get(internal, "Other")


def rel_course_file_to_local(course_dir: Path, ocw_path: str) -> Optional[Path]:
    if not ocw_path:
        return None
    basename = Path(ocw_path).name
    if not basename:
        return None
    for sub in ("static_resources", "resources", ""):
        candidate = course_dir / sub / basename if sub else course_dir / basename
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Course metadata
# ---------------------------------------------------------------------------

def parse_course_metadata(course_dir: Path) -> dict:
    data_path = course_dir / "data.json"
    if not data_path.exists():
        return {"course_dir": course_dir.name}
    try:
        d = json.loads(data_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {"course_dir": course_dir.name}

    instructors = []
    for person in d.get("instructors") or []:
        first = person.get("first_name", "")
        last  = person.get("last_name", "")
        name  = f"{first} {last}".strip()
        if name:
            instructors.append(name)

    course_id = d.get("primary_course_number")
    term = d.get("term")
    year = d.get("year")
    semester = f"{term} {year}".strip() if (term or year) else None

    return {
        "course_dir":   course_dir.name,
        "course_id":    course_id,
        "course_title": d.get("course_title"),
        "semester":     semester,
        "instructor":   join_nonempty(instructors) or None,
        "department":   dept_from_course_id(course_id),
    }


# ---------------------------------------------------------------------------
# Modern format processing
# ---------------------------------------------------------------------------

def modern_course_records(
    course_dir: Path,
    extractor: TextExtractor,
    min_text_chars: int,
    ingestion_timestamp: str,
    dataset_version: str,
) -> list[dict]:
    meta = parse_course_metadata(course_dir)
    resources_dir = course_dir / "resources"
    if not resources_dir.exists():
        return []

    seen_paths: set[str] = set()
    records: list[dict] = []

    for data_path in resources_dir.rglob("data.json"):
        try:
            d = json.loads(data_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        lr_types = d.get("learning_resource_types") or []
        if not isinstance(lr_types, list):
            lr_types = [str(lr_types)]

        internal = map_lrt_to_internal([str(x) for x in lr_types])

        if not internal and "syllabus" in str(data_path).lower():
            internal = "Syllabus"

        label        = internal_to_canonical(internal)
        label_source = "folder_structure" if internal else "no_rule_matched"

        source_file: Optional[str] = None
        source_path: Optional[Path] = None
        is_vtt = False

        for field in ("file", "transcript_file"):
            f = d.get(field)
            if f and str(f).lower().endswith(".pdf"):
                local = rel_course_file_to_local(course_dir, str(f))
                if local and local.exists():
                    source_file = str(f)
                    source_path = local
                    break

        if source_path is None:
            cf = d.get("captions_file")
            if cf and str(cf).lower().endswith(".vtt"):
                local = rel_course_file_to_local(course_dir, str(cf))
                if local and local.exists():
                    source_file = str(cf)
                    source_path = local
                    is_vtt = True

        if source_path is None or str(source_path) in seen_paths:
            continue
        seen_paths.add(str(source_path))

        text = (extract_vtt_text(source_path) if is_vtt else extractor.extract_pdf(source_path)).strip()
        if len(text) < min_text_chars:
            continue

        source_url = (OCW_BASE + source_file) if source_file else (OCW_BASE + "/courses/" + course_dir.name + "/" + str(source_path.relative_to(course_dir)))
        doc_id = make_doc_id(source_url)

        records.append({
            # required
            "doc_id":               doc_id,
            "extracted_text":       text,
            "label":                label,
            "label_source":         label_source,
            "course_id":            meta.get("course_id"),
            "source_url":           source_url,
            "source":               "mit_ocw",
            "ingestion_timestamp":  ingestion_timestamp,
            "dataset_version":      dataset_version,
            # optional
            "department":           meta.get("department"),
            "course_title":         meta.get("course_title"),
            "semester":             meta.get("semester"),
            "filename":             source_path.name,
            "char_count":           len(text),
            "word_count":           len(text.split()),
            "file_size_bytes":      source_path.stat().st_size,
            "text_extraction_method": "vtt" if is_vtt else extractor.backend,
            "instructor":           meta.get("instructor"),
        })

    return records


# ---------------------------------------------------------------------------
# Legacy format processing
# ---------------------------------------------------------------------------

def _legacy_label(parts: list[str], stem: str) -> tuple[Optional[str], str]:
    """Returns (internal_label, label_source)."""
    if "syllabus" in parts:
        return "Syllabus", "folder_structure"
    if "instructor-insights" in parts:
        return "Instructor Insights", "folder_structure"
    if "related-resources" in parts:
        return "Reading / Reference", "folder_structure"
    if any(p.startswith("lecture-") for p in parts):
        return "Lecture Notes", "folder_structure"
    if any(p.startswith("problem-solving") for p in parts):
        return "Problem Set", "folder_structure"
    if any("exam" in p for p in parts) and any("review" in p for p in parts):
        return "Exam", "folder_structure"
    if any(p.startswith("exam") for p in parts):
        return ("Problem Set Solution" if stem.endswith("sol") else "Exam"), "folder_structure"
    if stem.endswith("prob"):
        return "Problem Set", "filename_pattern"
    if stem.endswith("sol"):
        return "Problem Set Solution", "filename_pattern"
    if stem.endswith("sum"):
        return "Lecture Notes", "filename_pattern"
    return None, "no_rule_matched"


def derive_legacy_label(pdf_path: Path, course_root: Path) -> tuple[Optional[str], str]:
    rel   = pdf_path.relative_to(course_root)
    parts = [p.lower() for p in rel.parts]
    stem  = pdf_path.stem.lower()
    fname = pdf_path.name.lower()

    if YOUTUBE_ID_RE.match(stem):
        parent = pdf_path.parent.name.lower()
        return _legacy_label(parts[:-1] + [parent], stem=parent)
    if HASH_NAME_RE.match(fname):
        return "Reading / Reference", "filename_pattern"
    return _legacy_label(parts, stem=stem)


def legacy_course_records(
    course_dir: Path,
    extractor: TextExtractor,
    min_text_chars: int,
    ingestion_timestamp: str,
    dataset_version: str,
) -> list[dict]:
    contents_dirs = list(course_dir.rglob("contents"))
    if not contents_dirs:
        return []

    meta = parse_course_metadata(course_dir)
    records: list[dict] = []

    for contents in contents_dirs:
        for pdf in contents.rglob("*.pdf"):
            internal, label_source = derive_legacy_label(pdf, contents)
            label = internal_to_canonical(internal)

            text = (extractor.extract_pdf(pdf) or "").strip()
            if len(text) < min_text_chars:
                continue

            rel_path   = pdf.relative_to(contents)
            source_url = f"{OCW_BASE}/courses/{course_dir.name}/{rel_path}"
            doc_id     = make_doc_id(source_url)

            records.append({
                "doc_id":               doc_id,
                "extracted_text":       text,
                "label":                label,
                "label_source":         label_source,
                "course_id":            meta.get("course_id"),
                "source_url":           source_url,
                "source":               "mit_ocw",
                "ingestion_timestamp":  ingestion_timestamp,
                "dataset_version":      dataset_version,
                "department":           meta.get("department"),
                "course_title":         meta.get("course_title"),
                "semester":             meta.get("semester"),
                "filename":             pdf.name,
                "char_count":           len(text),
                "word_count":           len(text.split()),
                "file_size_bytes":      pdf.stat().st_size,
                "text_extraction_method": extractor.backend,
                "instructor":           meta.get("instructor"),
            })

    return records


# ---------------------------------------------------------------------------
# Dataset validation (from team schema spec)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "doc_id", "extracted_text", "label", "label_source",
    "course_id", "source_url", "source", "ingestion_timestamp", "dataset_version",
}


def validate_dataset(df: pd.DataFrame) -> None:
    errors = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            errors.append(f"Null values in required column: {col}")

    if "label" in df.columns:
        bad = set(df["label"].unique()) - VALID_LABELS
        if bad:
            errors.append(f"Invalid label values: {bad}")

    if "label_source" in df.columns:
        bad = set(df["label_source"].unique()) - VALID_LABEL_SOURCES
        if bad:
            errors.append(f"Invalid label_source values: {bad}")

    if "label" in df.columns and "label_source" in df.columns:
        bad_other = df[(df["label"] == "Other") & (df["label_source"] != "no_rule_matched")]
        if len(bad_other):
            errors.append(f"{len(bad_other)} rows: label=Other but label_source != no_rule_matched")

    if "extracted_text" in df.columns:
        empty = (df["extracted_text"].str.strip() == "").sum()
        if empty:
            errors.append(f"{empty} rows have empty extracted_text")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        raise ValueError("Dataset validation failed")

    label_counts = df["label"].value_counts().to_dict()
    other_pct = round(label_counts.get("Other", 0) / len(df) * 100, 1)
    print(f"[OK] Validation passed: {len(df)} rows, {df['course_id'].nunique()} courses")
    print(f"     Labels: {label_counts}")
    if other_pct > 30:
        print(f"[WARN] 'Other' is {other_pct}% of dataset — review label rules")


# ---------------------------------------------------------------------------
# Find course dirs
# ---------------------------------------------------------------------------

def find_course_dirs(root: Path) -> list[Path]:
    candidates = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "data.json").exists() and (p / "resources").exists():
            candidates.append(p)
        elif list(p.rglob("contents")):
            candidates.append(p)
    return candidates


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dataset(
    root: Path,
    output_parquet: Path,
    summary_json: Optional[Path],
    min_text_chars: int,
    dedupe_by_text: bool,
    dataset_version: str,
    extractor: TextExtractor,
) -> None:
    course_dirs = find_course_dirs(root)
    if not course_dirs:
        raise RuntimeError(f"No OCW course folders found in {root}")

    print(f"[INFO] Found {len(course_dirs)} course directories")
    print(f"[INFO] Text backend: {extractor.backend}")

    ingestion_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records: list[dict] = []

    for course_dir in course_dirs:
        modern = modern_course_records(course_dir, extractor, min_text_chars, ingestion_timestamp, dataset_version)
        use = modern if modern else legacy_course_records(course_dir, extractor, min_text_chars, ingestion_timestamp, dataset_version)
        print(f"[INFO] {course_dir.name}: {len(use)} records")
        records.extend(use)

    if not records:
        raise RuntimeError("No records extracted")

    df = pd.DataFrame(records)

    before = len(df)
    if dedupe_by_text:
        text_hashes = df["extracted_text"].apply(lambda t: hashlib.sha1(normalize_text(t).encode()).hexdigest())
        df = df[~text_hashes.duplicated(keep="first")].copy()
    after = len(df)

    validate_dataset(df)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    summary = {
        "num_rows":            int(len(df)),
        "num_courses":         int(df["course_id"].nunique()),
        "dataset_version":     dataset_version,
        "ingestion_timestamp": ingestion_timestamp,
        "text_backend":        extractor.backend,
        "min_text_chars":      int(min_text_chars),
        "rows_before_dedupe":  int(before),
        "rows_after_dedupe":   int(after),
        "label_counts":        df["label"].value_counts().to_dict(),
        "label_source_counts": df["label_source"].value_counts().to_dict(),
        "course_counts":       df["course_id"].value_counts().to_dict(),
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[INFO] Dataset build complete")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCW dataset (team schema v2)")
    parser.add_argument("--root",            type=Path,  default=Path("."))
    parser.add_argument("--output",          type=Path,  default=Path("artifacts/ocw_dataset.parquet"))
    parser.add_argument("--summary-output",  type=Path,  default=Path("artifacts/ocw_summary.json"))
    parser.add_argument("--min-text-chars",  type=int,   default=200)
    parser.add_argument("--no-dedupe",       action="store_true")
    parser.add_argument("--dataset-version", type=str,   default="v1.0")
    parser.add_argument("--text-backend",    type=str,   default="auto",
                        choices=["auto", "pypdf", "pdftotext", "strings", "none"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    extractor = TextExtractor.autodetect() if args.text_backend == "auto" else TextExtractor(args.text_backend)

    try:
        build_dataset(
            root=args.root,
            output_parquet=args.output,
            summary_json=args.summary_output,
            min_text_chars=args.min_text_chars,
            dedupe_by_text=not args.no_dedupe,
            dataset_version=args.dataset_version,
            extractor=extractor,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
