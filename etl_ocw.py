"""
MIT OCW ETL Pipeline
--------------------
Builds a labeled Parquet dataset from one or more extracted OCW course folders.

This script supports two folder layouts:
1) Legacy OCW layout with nested `contents/` folders (e.g. 18-06sc-fall-2011)
2) Modern OCW layout with `resources/**/data.json` + `static_resources/*.pdf`

Output schema is intentionally unchanged:
    file_id, course, label, text, source_path

Usage:
    python etl_ocw.py --input_dirs <dir1> [<dir2> ...] --output dataset.parquet
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

YOUTUBE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')
HASH_NAME_RE  = re.compile(r'^[a-f0-9]{32}')

RESOURCE_TYPE_TO_LABEL = {
    "Lecture Notes": "Lecture Notes",
    "Problem Sets": "Problem Set",
    "Problem Set Solutions": "Problem Set Solution",
    "Exams": "Exam",
    "Exam Solutions": "Exam Solution",
    "Readings": "Reading / Reference",
    "Instructor Insights": "Instructor Insights",
}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _extract_pdf_text_pdfminer(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        text = extract_text(str(path))
        return text.strip() if text else ""
    except Exception:
        return ""


def _extract_pdf_text_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts)
        return text.strip()
    except Exception:
        return ""


def _extract_pdf_text_pdftotext(path: Path) -> str:
    if not shutil.which("pdftotext"):
        return ""
    try:
        proc = subprocess.run(
            ["pdftotext", str(path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _extract_pdf_text_strings(path: Path) -> str:
    if not shutil.which("strings"):
        return ""
    try:
        proc = subprocess.run(
            ["strings", "-n", "8", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def derive_label(pdf_path: Path, course_root: Path) -> str | None:
    """
    Return a string label for the given PDF based on its path relative
    to course_root.  Returns None if the file should be skipped.
    """
    rel = pdf_path.relative_to(course_root)
    parts = [p.lower() for p in rel.parts]
    stem  = pdf_path.stem.lower()
    fname = pdf_path.name.lower()

    if YOUTUBE_ID_RE.match(stem):
        parent = pdf_path.parent.name.lower()
        return _label_from_parts(parts[:-1] + [parent], stem=parent)

    if HASH_NAME_RE.match(fname):
        return "Reading / Reference"

    return _label_from_parts(parts, stem=stem)


def _label_from_parts(parts: list[str], stem: str) -> str | None:
    if "syllabus" in parts:
        return "Syllabus"
    if "instructor-insights" in parts:
        return "Instructor Insights"
    if "related-resources" in parts:
        return "Related Resources"
    if any(p.startswith("lecture-") for p in parts):
        return "Lecture Notes"
    if any(p.startswith("problem-solving") for p in parts):
        return "Problem Solving Session"
    if any("exam" in p for p in parts) and any("review" in p for p in parts):
        return "Exam Review"
    if any(p.startswith("exam") for p in parts):
        if stem.endswith("sol"):
            return "Exam Solution"
        return "Exam"
    if stem.endswith("prob"):
        return "Problem Set"
    if stem.endswith("sol"):
        return "Problem Set Solution"
    if stem.endswith("sum"):
        return "Summary"

    return None  # skip unclassifiable


def extract_pdf_text(path: Path) -> str:
    for fn in (
        _extract_pdf_text_pdfminer,
        _extract_pdf_text_pypdf,
        _extract_pdf_text_pdftotext,
        _extract_pdf_text_strings,
    ):
        text = fn(path)
        if text:
            return text
    return ""


def _resolve_local_pdf(course_dir: Path, source_path: str) -> Path | None:
    if not source_path:
        return None
    if not source_path.lower().endswith(".pdf"):
        return None

    name = Path(source_path).name
    candidates = [
        course_dir / "static_resources" / name,
        course_dir / "resources" / name,
        course_dir / name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _derive_modern_label(resource_json_path: Path, data: dict) -> str | None:
    raw_types = data.get("learning_resource_types") or []
    if not isinstance(raw_types, list):
        raw_types = [raw_types]

    for t in raw_types:
        label = RESOURCE_TYPE_TO_LABEL.get(str(t))
        if label:
            return label

    title = str(data.get("title") or "").lower()
    path_l = str(resource_json_path).lower()
    if "syllabus" in title or "syllabus" in path_l:
        return "Syllabus"
    return None


def process_modern_course_folder(folder: Path) -> list[dict]:
    resources_dir = folder / "resources"
    if not resources_dir.exists():
        return []

    records = []
    for data_json in resources_dir.rglob("data.json"):
        data = _read_json(data_json)
        if not data:
            continue

        label = _derive_modern_label(data_json, data)
        if label is None:
            continue

        local_pdf = _resolve_local_pdf(folder, str(data.get("file") or ""))
        if local_pdf is None:
            continue

        text = extract_pdf_text(local_pdf)
        if not text:
            continue

        records.append({
            "file_id": str(local_pdf.relative_to(folder)),
            "course": folder.name,
            "label": label,
            "text": text,
            "source_path": str(local_pdf),
        })
    return records


def process_legacy_course_folder(folder: Path) -> list[dict]:
    """
    Find `contents/` sub-directories and process labeled PDFs inside them.
    """
    contents_dirs = list(folder.rglob("contents"))
    if not contents_dirs:
        print(f"[WARN] No 'contents/' dir found in {folder}, using folder root as fallback")
        contents_dirs = [folder]

    records = []
    for contents in contents_dirs:
        course_name = contents.parent.name
        for pdf in contents.rglob("*.pdf"):
            label = derive_label(pdf, contents)
            if label is None:
                continue
            text = extract_pdf_text(pdf)
            if not text:
                continue
            records.append({
                "file_id": str(pdf.relative_to(contents)),
                "course": course_name,
                "label": label,
                "text": text,
                "source_path": str(pdf),
            })
    return records


def process_course_folder(folder: Path) -> list[dict]:
    # Modern marker
    if (folder / "resources").exists() and (folder / "data.json").exists():
        modern_records = process_modern_course_folder(folder)
        if modern_records:
            return modern_records
    return process_legacy_course_folder(folder)


def main():
    parser = argparse.ArgumentParser(description="MIT OCW ETL pipeline")
    parser.add_argument(
        "--input_dirs", nargs="+", required=True,
        help="One or more paths to extracted OCW course folders"
    )
    parser.add_argument(
        "--output", default="dataset.parquet",
        help="Output Parquet file path (default: dataset.parquet)"
    )
    args = parser.parse_args()

    all_records = []
    for d in args.input_dirs:
        folder = Path(d)
        if not folder.exists():
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue
        print(f"[INFO] Processing: {folder}")
        records = process_course_folder(folder)
        print(f"       → {len(records)} labeled documents extracted")
        all_records.extend(records)

    if not all_records:
        print("[ERROR] No records extracted. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    print(f"\n[INFO] Total documents: {len(df)}")
    print(df["label"].value_counts().to_string())

    output_path = Path(args.output)
    df.to_parquet(output_path, index=False)
    print(f"\n[INFO] Saved to {output_path}")


if __name__ == "__main__":
    main()
