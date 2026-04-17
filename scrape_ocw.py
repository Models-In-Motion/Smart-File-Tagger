#!/usr/bin/env python3
"""
scrape_ocw.py — Production MIT OCW course discovery and ZIP downloader.

Step 1 — Discover all course slugs from the OCW sitemap index.
Step 2 — For each course, visit /download/, parse the .zip link with
          BeautifulSoup, download the archive, extract it, then delete the zip.
Step 3 — Resume-safe: a checkpoint file tracks completed / failed courses so
          the script can be restarted after interruption without re-downloading.

Usage examples:
    python scrape_ocw.py                          # download everything (200+ courses)
    python scrape_ocw.py --max-courses 3          # quick smoke-test
    python scrape_ocw.py --departments 6 18 8     # only EECS / Math / Physics
    python scrape_ocw.py --seed 42 --max-courses 50

After downloading, point build_ocw_dataset.py at the output directory:
    python build_ocw_dataset.py --root data/courses ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCW_BASE        = "https://ocw.mit.edu"
SITEMAP_URL     = f"{OCW_BASE}/sitemap.xml"
USER_AGENT      = "MIT-OCW-ETL/1.0 (ECE-GY-9183 MLOps Project; educational use)"
MIN_SLEEP       = 2.0
DEFAULT_SLEEP   = 3.0
REQUEST_TIMEOUT = 30          # seconds — page fetches
DOWNLOAD_TIMEOUT = 300        # seconds — ZIP downloads (some are large)


# Matches /courses/{slug}/ in any URL
COURSE_SLUG_RE = re.compile(r"/courses/([a-z0-9][^/]+)/?")
# First numeric segment of a slug → department number (e.g. "6-006-…" → "6")
DEPT_RE = re.compile(r"^(\d+)")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def fetch_text(session: requests.Session, url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_bytes_to_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> None:
    """Stream-download url → out_path (1 MB chunks)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Checkpoint / discovery JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1 — Discover courses from sitemap
# ---------------------------------------------------------------------------

def slug_department(slug: str) -> str:
    """Return the department number prefix from a course slug ('6-006-…' → '6')."""
    m = DEPT_RE.match(slug)
    return m.group(1) if m else ""


def parse_sitemap_slugs(xml_text: str) -> list[str]:
    """
    Parse the OCW sitemap index XML and return a sorted, deduplicated list
    of course slugs.

    The top-level sitemap is a <sitemapindex> whose <loc> entries look like:
        https://ocw.mit.edu/courses/{slug}/sitemap.xml
    We extract {slug} from every such URL.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse sitemap XML: {exc}") from exc

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Try namespaced lookup first, fall back to bare tag (both appear in practice)
    locs = [el.text for el in root.findall(".//sm:loc", ns)] or \
           [el.text for el in root.findall(".//loc")]

    slugs: list[str] = []
    seen: set[str] = set()
    for loc in locs:
        if not loc:
            continue
        m = COURSE_SLUG_RE.search(loc)
        if not m:
            continue
        slug = m.group(1)
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)

    return sorted(slugs)


def discover_courses(
    session: requests.Session,
    sleep: float,
    discovery_path: Path,
    departments: Optional[list[str]] = None,
) -> list[str]:
    """
    Fetch the OCW sitemap, extract all course slugs, optionally filter by
    department, persist to discovery_path, and return the slug list.
    """
    print(f"[INFO] Fetching sitemap: {SITEMAP_URL}")
    xml_text = fetch_text(session, SITEMAP_URL)
    time.sleep(sleep)

    all_slugs = parse_sitemap_slugs(xml_text)
    print(f"[INFO] Sitemap contains {len(all_slugs)} course slugs")

    if departments:
        dept_set = set(departments)
        filtered = [s for s in all_slugs if slug_department(s) in dept_set]
        print(f"[INFO] Department filter {departments}: {len(filtered)} courses kept")
        all_slugs = filtered

    discovery = {
        "count": len(all_slugs),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "departments_filter": departments or [],
        "slugs": all_slugs,
    }
    save_json(discovery_path, discovery)
    print(f"[INFO] Saved {len(all_slugs)} slugs → {discovery_path}")
    return all_slugs


# ---------------------------------------------------------------------------
# Step 2 — Find the ZIP download link for a course
# ---------------------------------------------------------------------------

def get_zip_url(session: requests.Session, slug: str) -> Optional[str]:
    """
    Visit https://ocw.mit.edu/courses/{slug}/download/ and use BeautifulSoup
    to locate the first <a href="…*.zip"> link.

    Returns the absolute ZIP URL, or None if no link is found.
    Raises requests.HTTPError on 4xx/5xx so the caller can handle 404 cleanly.
    """
    download_page = f"{OCW_BASE}/courses/{slug}/download/"
    html = fetch_text(session, download_page)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        lower = href.lower()
        if lower.endswith(".zip") or ".zip?" in lower:
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return f"{OCW_BASE}{href}"
            # Relative URL — resolve against the download page
            return f"{OCW_BASE}/courses/{slug}/{href}"

    return None


# ---------------------------------------------------------------------------
# Step 3 — Extract ZIP with single-root-folder stripping
# ---------------------------------------------------------------------------

def extract_zip_to_dir(zip_path: Path, out_dir: Path) -> None:
    """
    Extract zip_path into out_dir, stripping any single top-level folder so
    that course files always land directly in out_dir/.

    build_ocw_dataset.py expects:
        out_dir/data.json          ← modern format
        out_dir/resources/
        out_dir/static_resources/
    or
        out_dir/.../contents/      ← legacy format

    Without stripping, the ZIP's internal root folder (e.g. "18.06-spring-2010/")
    would create out_dir/18.06-spring-2010/data.json, which the ETL wouldn't find.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("ZIP archive is empty")

        # Detect whether ALL entries share exactly one top-level folder
        top_level_names = {n.split("/")[0] for n in names if n.strip("/")}
        single_root = (
            len(top_level_names) == 1
            and any(n.endswith("/") and n.count("/") == 1 for n in names)
        )

        if single_root:
            prefix = top_level_names.pop() + "/"
            for member in zf.infolist():
                rel = member.filename[len(prefix):]
                if not rel:          # the root dir entry itself — skip
                    continue
                target = out_dir / rel
                if member.filename.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src:
                        target.write_bytes(src.read())
        else:
            # Files already at ZIP root — extract directly
            zf.extractall(out_dir)


# ---------------------------------------------------------------------------
# Main scraper loop
# ---------------------------------------------------------------------------

def run_scraper(args: argparse.Namespace) -> int:
    if args.sleep < MIN_SLEEP:
        print(f"[ERROR] --sleep must be >= {MIN_SLEEP:.1f} s (OCW rate-limit policy)",
              file=sys.stderr)
        return 1

    output_dir      = Path(args.output_dir)
    data_dir        = output_dir.parent          # data/  (parent of data/courses/)
    zips_dir        = data_dir / "zips"
    discovery_path  = data_dir / "discovered_courses.json"
    checkpoint_path = data_dir / "scrape_checkpoint.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt      = load_json(checkpoint_path)
    completed: set[str]  = set(ckpt.get("completed", []))
    failed:    dict[str, str] = ckpt.get("failed", {})

    def save_checkpoint() -> None:
        save_json(checkpoint_path, {
            "completed":    sorted(completed),
            "failed":       failed,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    # ── Step 1: Discover (or load cached) slugs ───────────────────────────────
    existing = load_json(discovery_path)
    if existing.get("slugs"):
        slugs: list[str] = existing["slugs"]
        print(f"[INFO] Reusing discovery file ({len(slugs)} slugs): {discovery_path}")
        # Re-apply department filter if caller wants a subset of what was cached
        if args.departments:
            dept_set = set(args.departments)
            slugs = [s for s in slugs if slug_department(s) in dept_set]
            print(f"[INFO] Department filter {args.departments}: {len(slugs)} courses")
    else:
        slugs = discover_courses(
            session, args.sleep, discovery_path,
            departments=args.departments or None,
        )

    if not slugs:
        print("[ERROR] No courses to download — check sitemap or --departments filter",
              file=sys.stderr)
        return 1

    # ── Shuffle for variety, then cap ─────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(slugs)

    if args.max_courses > 0:
        slugs = slugs[: args.max_courses]
        print(f"[INFO] Capped at {len(slugs)} courses (--max-courses {args.max_courses})")

    # ── Skip already-completed ────────────────────────────────────────────────
    if args.skip_existing:
        before = len(slugs)
        slugs = [
            s for s in slugs
            if s not in completed and not (output_dir / s).is_dir()
        ]
        skipped = before - len(slugs)
        if skipped:
            print(f"[INFO] Skipping {skipped} already-completed courses")

    total = len(slugs)
    print(f"\n[INFO] Courses to download : {total}")
    print(f"[INFO] Output directory    : {output_dir}")
    print(f"[INFO] Checkpoint          : {checkpoint_path}")
    print(f"[INFO] Sleep between reqs  : {args.sleep:.1f} s\n")

    success_count = 0
    fail_count    = 0

    for idx, slug in enumerate(slugs, start=1):
        print(f"[{idx}/{total}] Downloading {slug} ...")

        # ── 2a. Get ZIP URL from /download/ page ─────────────────────────────
        zip_url: Optional[str] = None
        try:
            zip_url = get_zip_url(session, slug)
            time.sleep(args.sleep)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            msg  = f"HTTP {code} fetching download page"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            save_checkpoint()
            time.sleep(args.sleep)
            continue
        except requests.exceptions.Timeout:
            msg = "Timeout fetching download page"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            save_checkpoint()
            time.sleep(args.sleep)
            continue
        except Exception as exc:
            msg = f"Error fetching download page: {exc}"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            save_checkpoint()
            time.sleep(args.sleep)
            continue

        if zip_url is None:
            msg = "No .zip link found on download page"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            save_checkpoint()
            continue

        print(f"  ZIP  : {zip_url}")

        # ── 2b. Download the ZIP ──────────────────────────────────────────────
        zip_path = zips_dir / f"{slug}.zip"
        try:
            fetch_bytes_to_file(session, zip_url, zip_path)
            time.sleep(args.sleep)
        except requests.exceptions.Timeout:
            msg = "Timeout downloading ZIP"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            zip_path.unlink(missing_ok=True)
            save_checkpoint()
            time.sleep(args.sleep)
            continue
        except Exception as exc:
            msg = f"Download failed: {exc}"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            zip_path.unlink(missing_ok=True)
            save_checkpoint()
            time.sleep(args.sleep)
            continue

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if size_mb == 0:
            msg = "Downloaded ZIP is 0 bytes"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            zip_path.unlink(missing_ok=True)
            save_checkpoint()
            continue

        print(f"  Size : {size_mb:.1f} MB")

        # ── 2c. Extract ZIP → data/courses/{slug}/ ────────────────────────────
        course_dir = output_dir / slug
        try:
            extract_zip_to_dir(zip_path, course_dir)
            print(f"  Out  : {course_dir}")
        except zipfile.BadZipFile:
            msg = "BadZipFile — archive may be corrupted or truncated"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            zip_path.unlink(missing_ok=True)
            save_checkpoint()
            continue
        except Exception as exc:
            msg = f"Extraction failed: {exc}"
            print(f"  [SKIP] {msg}")
            failed[slug] = msg
            fail_count  += 1
            zip_path.unlink(missing_ok=True)
            save_checkpoint()
            continue

        # ── 2d. Delete ZIP to save disk space ────────────────────────────────
        zip_path.unlink(missing_ok=True)

        completed.add(slug)
        failed.pop(slug, None)   # clear any previous failure entry for this slug
        success_count += 1
        save_checkpoint()
        print(f"  [OK]")

    # ── Final summary ─────────────────────────────────────────────────────────
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"[DONE] Completed : {success_count}")
    print(f"[DONE] Failed    : {fail_count}")
    print(f"[DONE] Attempted : {total}")
    print(f"[DONE] Output    : {output_dir}")
    if failed:
        sample = list(failed.items())[:20]
        print(f"\n[WARN] Failed courses ({len(failed)}) — first 20:")
        for s, reason in sample:
            print(f"  {s}: {reason}")
        if len(failed) > 20:
            print(f"  … and {len(failed) - 20} more — see {checkpoint_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and download MIT OCW course ZIPs at scale. "
            "Extracted folders are compatible with build_ocw_dataset.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="data/courses",
        help="Root directory where extracted course folders are written",
    )
    parser.add_argument(
        "--max-courses",
        type=int,
        default=0,
        help="Maximum number of courses to download (0 = no limit)",
    )
    parser.add_argument(
        "--departments",
        nargs="*",
        default=[],
        metavar="DEPT",
        help=(
            "Filter by MIT department number prefix, e.g. --departments 6 18 8 5. "
            "Default: all departments."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Seconds between HTTP requests (minimum {MIN_SLEEP})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling course order (for reproducibility)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip courses that already have a folder in --output-dir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_scraper(parse_args()))
