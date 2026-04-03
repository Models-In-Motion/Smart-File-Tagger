#!/usr/bin/env python3
"""
online_features.py

Real-time feature computation path for the document auto-tagging system.
Takes a single PDF file, extracts text and computes features, and outputs
the JSON payload that the serving layer expects.

Usage:
    python online_features.py path/to/document.pdf
    python online_features.py path/to/document.pdf --pretty
    python online_features.py path/to/document.pdf --output out.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Text extraction (mirrors build_ocw_dataset.py — single-file version)
# ---------------------------------------------------------------------------

def _extract_pypdf(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return text, "pypdf"
    except Exception:
        return "", "pypdf_failed"


def _extract_pdftotext(path: Path) -> tuple[str, str]:
    try:
        proc = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip(), "pdftotext"
    except Exception:
        pass
    return "", "pdftotext_failed"


def extract_text(path: Path) -> tuple[str, str]:
    """Try pypdf first, fall back to pdftotext. Returns (text, backend_used)."""
    try:
        import pypdf  # noqa: F401
        text, backend = _extract_pypdf(path)
        if text:
            return text, backend
    except ImportError:
        pass

    # fallback
    text, backend = _extract_pdftotext(path)
    return text, backend


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(pdf_path: str) -> dict:
    path = Path(pdf_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    extracted_text, text_extraction_method = extract_text(path)

    char_count = len(extracted_text)
    word_count = len(extracted_text.split()) if extracted_text else 0
    file_size_bytes = path.stat().st_size

    return {
        "filename": path.name,
        "file_size_bytes": file_size_bytes,
        "text_extraction_method": text_extraction_method,
        "extracted_text": extracted_text,
        "char_count": char_count,
        "word_count": word_count,
        "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from a single PDF for real-time inference."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print JSON output (default: compact)")
    parser.add_argument("--output", help="Write JSON to this file instead of stdout")
    args = parser.parse_args()

    try:
        features = compute_features(args.pdf)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    indent = 2 if args.pretty else None
    output = json.dumps(features, indent=indent, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output)
        # Print summary to stderr so stdout stays clean JSON
        print(f"[INFO] Written to {args.output}", file=sys.stderr)
        print(f"[INFO] chars={features['char_count']}  words={features['word_count']}  "
              f"backend={features['text_extraction_method']}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
