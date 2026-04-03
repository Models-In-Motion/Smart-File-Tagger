#!/usr/bin/env python3
"""
Polite OCW archive downloader.

Given course page URLs, this script tries to find downloadable archive links
(typically .zip) from each course's /download/ page and downloads them with a
mandatory wait between requests.

This is intended for ETL bootstrapping and should be run responsibly.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


ZIP_LINK_RE = re.compile(r'href=["\']([^"\']+\.zip(?:\?[^"\']*)?)["\']', re.IGNORECASE)
GENERIC_LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MLSysOps-ETL/1.0 (+course-project)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def normalize_course_url(url: str) -> str:
    u = url.strip()
    if not u:
        return u
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    return u.rstrip("/") + "/"


def download_page_url(course_url: str) -> str:
    # Most OCW course pages expose a dedicated /download/ page.
    return urllib.parse.urljoin(course_url, "download/")


def find_zip_links(html: str, base_url: str) -> list[str]:
    links = [urllib.parse.urljoin(base_url, m.group(1)) for m in ZIP_LINK_RE.finditer(html)]
    # Keep unique order.
    seen = set()
    out = []
    for x in links:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def find_fallback_download_links(html: str, base_url: str) -> list[str]:
    out = []
    seen = set()
    for m in GENERIC_LINK_RE.finditer(html):
        href = m.group(1)
        link = urllib.parse.urljoin(base_url, href)
        low = link.lower()
        if "/download" in low and (low.endswith(".zip") or ".zip?" in low):
            if link not in seen:
                seen.add(link)
                out.append(link)
    return out


def safe_name_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = "course.zip"
    if not name.lower().endswith(".zip"):
        name = name + ".zip"
    return name


def download_file(url: str, out_path: Path, timeout: int = 120) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MLSysOps-ETL/1.0 (+course-project)",
            "Accept": "application/zip,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out_path.write_bytes(resp.read())


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    # Extract into folder named after zip stem.
    stem = zip_path.stem
    out_dir = target_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def parse_course_urls(args_urls: list[str], urls_file: Path | None) -> list[str]:
    urls: list[str] = []
    urls.extend(args_urls)
    if urls_file:
        for ln in urls_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)

    # normalize and dedupe
    out: list[str] = []
    seen = set()
    for u in urls:
        nu = normalize_course_url(u)
        if not nu:
            continue
        if nu in seen:
            continue
        seen.add(nu)
        out.append(nu)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OCW course archives (.zip) politely")
    parser.add_argument("--urls", nargs="*", default=[], help="Course URLs (e.g., https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/)")
    parser.add_argument("--urls-file", type=Path, default=None, help="Text file containing one course URL per line")
    parser.add_argument("--out-dir", type=Path, default=Path("ocw_downloads"), help="Where zip files are stored")
    parser.add_argument("--extract-dir", type=Path, default=Path("ocw_courses"), help="Where archives are extracted")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Wait between web requests (must be >=2 for course policy)" )
    parser.add_argument("--max-zips-per-course", type=int, default=1, help="Cap number of zips to download per course URL")
    parser.add_argument("--no-extract", action="store_true", help="Only download, do not extract")
    args = parser.parse_args()

    if args.sleep_seconds < 2.0:
        print("[ERROR] --sleep-seconds must be >= 2.0", file=sys.stderr)
        return 1

    course_urls = parse_course_urls(args.urls, args.urls_file)
    if not course_urls:
        print("[ERROR] No course URLs provided", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing {len(course_urls)} course URLs")
    print(f"[INFO] Sleep between requests: {args.sleep_seconds:.1f}s")

    downloaded = 0
    extracted = 0

    for idx, course_url in enumerate(course_urls, start=1):
        print(f"\n[INFO] ({idx}/{len(course_urls)}) course: {course_url}")
        dl_page = download_page_url(course_url)
        try:
            html = fetch_text(dl_page)
        except Exception as exc:
            print(f"[WARN] Failed to fetch {dl_page}: {exc}")
            time.sleep(args.sleep_seconds)
            continue

        zip_links = find_zip_links(html, dl_page)
        if not zip_links:
            zip_links = find_fallback_download_links(html, dl_page)

        if not zip_links:
            print("[WARN] No zip links found on download page")
            time.sleep(args.sleep_seconds)
            continue

        for zurl in zip_links[: args.max_zips_per_course]:
            name = safe_name_from_url(zurl)
            out_zip = args.out_dir / name
            if out_zip.exists() and out_zip.stat().st_size > 0:
                print(f"[INFO] Already downloaded: {out_zip}")
            else:
                print(f"[INFO] Downloading: {zurl}")
                try:
                    download_file(zurl, out_zip)
                    downloaded += 1
                    print(f"[INFO] Saved: {out_zip}")
                except Exception as exc:
                    print(f"[WARN] Download failed: {exc}")
                    time.sleep(args.sleep_seconds)
                    continue

            if not args.no_extract:
                try:
                    out_dir = extract_zip(out_zip, args.extract_dir)
                    extracted += 1
                    print(f"[INFO] Extracted: {out_dir}")
                except Exception as exc:
                    print(f"[WARN] Extraction failed for {out_zip}: {exc}")

            time.sleep(args.sleep_seconds)

    print("\n[INFO] Done")
    print(f"[INFO] Downloaded archives: {downloaded}")
    print(f"[INFO] Extracted archives: {extracted}")
    print(f"[INFO] ZIP directory: {args.out_dir}")
    print(f"[INFO] Extract directory: {args.extract_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
