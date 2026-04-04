"""
extractor.py — Text Extraction
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This file has one job: take a raw file (PDF, image, or plain text)
and return the text inside it as a plain Python string.

Why this is its own file:
    main.py receives the file. predictor.py needs text, not a file.
    extractor.py is the bridge between those two. Keeping it separate
    means if we ever want to swap pdfminer for a different library,
    we only change this one file and nothing else breaks.
"""

import io
import logging
from pathlib import Path

import pdfminer.high_level
from PIL import Image
import pytesseract

log = logging.getLogger(__name__)

# Maximum characters we keep from any document.
# SBERT has a 512-token limit anyway (~2000 chars), so storing
# more than this just wastes memory without improving predictions.
MAX_CHARS = 4000


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Extract plain text from a file.

    Args:
        file_bytes : the raw bytes of the uploaded file
        filename   : original filename, e.g. "pset3.pdf"
                     used to detect file type when mime type is unreliable

    Returns:
        A tuple of (extracted_text, method_used)
        extracted_text : plain text string, empty string if extraction failed
        method_used    : "pdfminer", "ocr", or "plaintext"

    Why we return the method:
        The predictor and the feedback log both want to know HOW we got the
        text. OCR text is noisier than pdfminer text, which may affect
        confidence scores later.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_from_pdf(file_bytes)

    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
        return _extract_from_image(file_bytes)

    elif suffix in {".txt", ".md", ".csv"}:
        return _extract_from_plaintext(file_bytes)

    else:
        # Unknown type — try PDF extraction first as a best guess,
        # fall back to treating it as plain text
        log.warning(f"Unknown file type '{suffix}', attempting PDF extraction")
        text, method = _extract_from_pdf(file_bytes)
        if text:
            return text, method
        return _extract_from_plaintext(file_bytes)


# ---------------------------------------------------------------------------
# Private helpers — one function per file type
# ---------------------------------------------------------------------------

def _extract_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    """
    Use pdfminer to extract text from a PDF.

    pdfminer works well on PDFs that were created digitally (e.g. exported
    from LaTeX or Word). It does NOT work on scanned PDFs — those are
    essentially images embedded in a PDF wrapper, and pdfminer will return
    an empty string for them.

    If pdfminer returns empty, we fall back to OCR automatically.
    """
    try:
        # pdfminer expects a file-like object, not raw bytes
        # io.BytesIO wraps the bytes so pdfminer can read them
        pdf_file = io.BytesIO(file_bytes)
        text = pdfminer.high_level.extract_text(pdf_file)

        if text and text.strip():
            clean = _clean_text(text)
            log.info(f"pdfminer extracted {len(clean)} chars")
            return clean, "pdfminer"

        # pdfminer returned nothing — this is probably a scanned PDF
        # Convert the PDF pages to images and run OCR on them
        log.info("pdfminer returned empty text, trying OCR fallback")
        return _ocr_pdf(file_bytes)

    except Exception as exc:
        log.warning(f"pdfminer failed: {exc}, trying OCR fallback")
        return _ocr_pdf(file_bytes)


def _ocr_pdf(file_bytes: bytes) -> tuple[str, str]:
    """
    Convert PDF pages to images and run Tesseract OCR on them.

    We only do this as a fallback because OCR is:
    1. Much slower than pdfminer
    2. Less accurate (especially on math notation, which is common in MIT OCW)

    We only process the first 3 pages to keep latency reasonable.
    For a Problem Set, the first 3 pages almost always contain enough
    text for a good classification.

    Requires: pdf2image library + poppler system dependency
    If these are not installed, OCR of scanned PDFs silently returns empty.
    """
    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(file_bytes, first_page=1, last_page=3)
        all_text = []
        for page in pages:
            page_text = pytesseract.image_to_string(page)
            all_text.append(page_text)
        combined = _clean_text(" ".join(all_text))
        log.info(f"OCR extracted {len(combined)} chars from {len(pages)} pages")
        return combined, "ocr"

    except ImportError:
        log.warning("pdf2image not installed — cannot OCR scanned PDFs")
        return "", "ocr_unavailable"
    except Exception as exc:
        log.warning(f"OCR failed: {exc}")
        return "", "ocr_failed"


def _extract_from_image(file_bytes: bytes) -> tuple[str, str]:
    """
    Run Tesseract OCR directly on an image file.
    Used when someone uploads a .jpg or .png scan of a document.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        clean = _clean_text(text)
        log.info(f"OCR extracted {len(clean)} chars from image")
        return clean, "ocr"
    except Exception as exc:
        log.warning(f"Image OCR failed: {exc}")
        return "", "ocr_failed"


def _extract_from_plaintext(file_bytes: bytes) -> tuple[str, str]:
    """
    Decode raw bytes as UTF-8 text.
    Falls back to latin-1 if UTF-8 decoding fails — latin-1 never throws
    because it maps every byte to a character.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    clean = _clean_text(text)
    log.info(f"Plaintext extracted {len(clean)} chars")
    return clean, "plaintext"


def _clean_text(text: str) -> str:
    """
    Remove junk whitespace and truncate to MAX_CHARS.

    Why we truncate:
        SBERT has a 512-token limit. Anything beyond that gets cut off
        silently by the model anyway. Truncating here means we're not
        wasting time encoding characters that will be ignored.
    """
    import re
    # Collapse multiple spaces, tabs, newlines into single space
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text[:MAX_CHARS]