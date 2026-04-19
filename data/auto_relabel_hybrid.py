#!/usr/bin/env python3
"""Hybrid relabeling for OCW PDFs with conservative confidence tiers.

Priority:
1) LRT single-label match from OCW resources metadata (high confidence)
2) Filename/url single-cue match (medium confidence)
3) Text single-cue match on unresolved rows only (medium confidence)
4) Keep original label when unresolved/conflicting
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

TARGET_LABELS = {
    "Lecture Notes",
    "Problem Set",
    "Exam",
    "Reading",
    "Project",
    "Solution",
    "Other",
}

LRT_TO_LABEL = {
    "lecture notes": "Lecture Notes",
    "lectures and recitations": "Lecture Notes",
    "video lectures": "Lecture Notes",
    "lecture videos": "Lecture Notes",
    "recitations": "Lecture Notes",
    "class sessions": "Lecture Notes",
    "problem sets": "Problem Set",
    "problem set": "Problem Set",
    "homework": "Problem Set",
    "assignments": "Problem Set",
    "laboratory assignments": "Problem Set",
    "labs": "Problem Set",
    "exams": "Exam",
    "quizzes": "Exam",
    "tests": "Exam",
    "midterm exam": "Exam",
    "final exam": "Exam",
    "readings": "Reading",
    "additional readings": "Reading",
    "reference materials": "Reading",
    "textbooks": "Reading",
    "projects": "Project",
    "project": "Project",
    "final project": "Project",
    "design project": "Project",
    "problem set solutions": "Solution",
    "exam solutions": "Solution",
    "quiz solutions": "Solution",
    "solutions": "Solution",
}

FILENAME_PATTERNS = [
    (r"\b(solution|solutions|soln|answer\s*key|answers?)\b", "Solution"),
    (r"\b(problem[\-_ ]?set|pset\d*|homework|hw\d*|assignment|assign|lab\d*|lab)\b", "Problem Set"),
    (r"\b(exam|midterm|final|quiz|test)\b", "Exam"),
    (r"\b(lecture|lec\d*|rec\d*|recitation|tut\d*|tutorial|notes?|slides?)\b", "Lecture Notes"),
    (r"\b(project|proposal|milestone|capstone|design|memo)\b", "Project"),
    (r"\b(reading|paper|article|chapter|bibliography|reference|guide)\b", "Reading"),
]

# Strong text cues only; intentionally stricter than filename rules.
TEXT_PATTERNS = [
    (r"\b(problem set\s*\d|homework\s*\d|assignment\s*\d|lab\s*\d)\b", "Problem Set"),
    (r"\b(lecture notes?|recitation|tutorial|class notes)\b", "Lecture Notes"),
    (r"\b(project proposal|final project|term project|project memo|capstone project)\b", "Project"),
    (r"\b(required reading|recommended reading|reading list|bibliography)\b", "Reading"),
]


@dataclass
class ResourceEntry:
    basename: str
    lrt_labels: list[str]


def norm_text(s: str) -> str:
    s = unquote(str(s or "")).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def infer_course_slug(row: pd.Series) -> str | None:
    source_url = row.get("source_url")
    if isinstance(source_url, str) and source_url:
        m = re.search(r"/courses/([^/]+)/", source_url)
        if m:
            return m.group(1)

    source_path = row.get("source_path")
    if isinstance(source_path, str):
        parts = Path(source_path).parts
        if "courses" in parts:
            i = parts.index("courses")
            if i + 1 < len(parts):
                return parts[i + 1]

    for col in ("course", "course_slug", "course_id"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def infer_basename(row: pd.Series) -> str:
    if isinstance(row.get("filename"), str) and row.get("filename"):
        return Path(str(row["filename"])).name.lower()
    if isinstance(row.get("source_path"), str) and row.get("source_path"):
        return Path(str(row["source_path"])).name.lower()
    source_url = str(row.get("source_url", "") or "")
    if source_url:
        return Path(urlparse(source_url).path).name.lower()
    return ""


def map_lrt(raw_lrt: list[str]) -> list[str]:
    labels: set[str] = set()
    for item in raw_lrt:
        label = LRT_TO_LABEL.get(norm_text(item))
        if label:
            labels.add(label)
    return sorted(labels)


def iter_resources(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("resources"), list):
            return [x for x in payload["resources"] if isinstance(x, dict)]
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def basename_from_resource(resource: dict[str, Any]) -> str | None:
    f = resource.get("file")
    if isinstance(f, str) and f.lower().endswith(".pdf"):
        return Path(f).name.lower()
    u = resource.get("url")
    if isinstance(u, str) and u:
        p = urlparse(u)
        name = Path(p.path).name.lower()
        if name.endswith(".pdf"):
            return name
    return None


def build_lrt_index(courses_root: Path) -> dict[tuple[str, str], list[ResourceEntry]]:
    idx: dict[tuple[str, str], list[ResourceEntry]] = defaultdict(list)
    for j in courses_root.rglob("resources/**/data.json"):
        try:
            payload = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue

        parts = j.parts
        if "courses" not in parts:
            continue
        ci = parts.index("courses")
        if ci + 1 >= len(parts):
            continue
        course_slug = parts[ci + 1]

        for res in iter_resources(payload):
            b = basename_from_resource(res)
            if not b:
                continue
            raw_lrt = res.get("learning_resource_types") or []
            if not isinstance(raw_lrt, list):
                raw_lrt = []
            idx[(course_slug, b)].append(
                ResourceEntry(basename=b, lrt_labels=map_lrt([str(x) for x in raw_lrt]))
            )
    return idx


def cues_from_patterns(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    labels = sorted({label for pat, label in patterns if re.search(pat, text)})
    return labels


def filename_single_cue_label(row: pd.Series) -> tuple[str | None, list[str]]:
    filename = infer_basename(row)
    source_url = str(row.get("source_url", "") or "")
    url_base = Path(urlparse(source_url).path).name if source_url else ""
    txt = norm_text(f"{filename} {url_base}")
    labels = cues_from_patterns(txt, FILENAME_PATTERNS)
    return (labels[0], labels) if len(labels) == 1 else (None, labels)


def text_single_cue_label(row: pd.Series, max_chars: int = 1600) -> tuple[str | None, list[str]]:
    txt = str(row.get("extracted_text", "") or "")[:max_chars]
    txt = norm_text(txt)
    labels = cues_from_patterns(txt, TEXT_PATTERNS)
    return (labels[0], labels) if len(labels) == 1 else (None, labels)


def has_solution_filename_hint(row: pd.Series) -> bool:
    """
    Detect explicit solution-like filename tokens.
    Conservative on purpose to avoid false positives like 'dissolution'.
    """
    name = infer_basename(row)
    if not name:
        return False
    return bool(
        re.search(
            r"(solution|solutions|soln|answer[_ -]?key|answers?|pset\d*sol|hw\d*sol|quiz\d*sol)",
            name,
        )
    )


def choose_label(row: pd.Series, lrt_idx: dict[tuple[str, str], list[ResourceEntry]]) -> dict[str, Any]:
    prev = str(row.get("label", "Other") or "Other")
    course = infer_course_slug(row)
    basename = infer_basename(row)

    lrt_reason = "no_lrt_match"
    lrt_candidates: list[str] = []
    matches: list[ResourceEntry] = []
    if course and basename:
        matches = lrt_idx.get((course, basename), [])
        if matches:
            lrt_candidates = sorted(set(y for x in matches for y in x.lrt_labels))
            if len(lrt_candidates) == 1:
                lrt_reason = "lrt_high_conf"
            elif len(lrt_candidates) > 1:
                lrt_reason = "lrt_conflict"
            else:
                lrt_reason = "lrt_missing_type"

    filename_label, filename_candidates = filename_single_cue_label(row)
    text_label, text_candidates = text_single_cue_label(row)

    # 1) Highest confidence: LRT single label
    if lrt_reason == "lrt_high_conf" and lrt_candidates[0] in {"Exam", "Problem Set"} and has_solution_filename_hint(row):
        new, reason, conf = "Solution", "filename_solution_override", "medium"
    elif lrt_reason == "lrt_high_conf":
        new, reason, conf = lrt_candidates[0], "lrt_high_conf", "high"

    # 2) If LRT conflicted but filename/text agrees with one LRT option, resolve conflict safely
    elif lrt_reason == "lrt_conflict" and filename_label and filename_label in lrt_candidates:
        new, reason, conf = filename_label, "lrt_conflict_resolved_filename", "medium"
    elif lrt_reason == "lrt_conflict" and text_label and text_label in lrt_candidates:
        new, reason, conf = text_label, "lrt_conflict_resolved_text", "medium"

    # 3) Filename cue
    elif filename_label:
        new, reason, conf = filename_label, "filename_single_cue", "medium"

    # 4) Text cue only for unresolved/Other-like cases
    elif text_label and prev == "Other" and text_label in {"Problem Set", "Lecture Notes", "Reading", "Project"}:
        new, reason, conf = text_label, "text_single_cue_from_other", "medium"
    elif text_label and lrt_reason in {"no_lrt_match", "lrt_missing_type"} and prev == "Other" and text_label in {"Problem Set", "Lecture Notes", "Reading", "Project"}:
        new, reason, conf = text_label, "text_single_cue_unresolved", "medium"

    elif lrt_reason == "lrt_conflict":
        new, reason, conf = prev, "keep_lrt_conflict", "low"
    else:
        new, reason, conf = prev, "keep_unresolved", "low"

    if new not in TARGET_LABELS:
        new, reason, conf = "Other", "coerce_other", "low"

    return {
        "label_prev": prev,
        "label_new": new,
        "relabel_reason": reason,
        "relabel_confidence": conf,
        "course_slug_inferred": course or "",
        "basename_inferred": basename,
        "lrt_candidates": ",".join(lrt_candidates),
        "filename_candidates": ",".join(filename_candidates),
        "text_candidates": ",".join(text_candidates),
        "lrt_match_count": len(matches),
    }


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    changed = int((df["label_prev"] != df["label_new"]).sum())
    unresolved = int((df["relabel_reason"] == "keep_unresolved").sum())
    transitions = df.groupby(["label_prev", "label_new"]).size().sort_values(ascending=False).head(25).to_dict()
    return {
        "rows_total": int(len(df)),
        "rows_changed": changed,
        "rows_changed_pct": round(changed / max(len(df), 1) * 100.0, 2),
        "rows_unresolved": unresolved,
        "rows_unresolved_pct": round(unresolved / max(len(df), 1) * 100.0, 2),
        "reason_counts": dict(Counter(df["relabel_reason"])),
        "confidence_counts": dict(Counter(df["relabel_confidence"])),
        "label_prev_counts": df["label_prev"].value_counts().to_dict(),
        "label_new_counts": df["label_new"].value_counts().to_dict(),
        "transition_top25": {f"{a} -> {b}": int(c) for (a, b), c in transitions.items()},
    }


def print_manual_sample(df: pd.DataFrame, n: int) -> None:
    changed = df[df["label_prev"] != df["label_new"]]
    if changed.empty:
        print("\nManual sample: no changed rows.")
        return
    cols = [
        "course_slug_inferred",
        "filename",
        "source_url",
        "label_prev",
        "label_new",
        "relabel_reason",
        "relabel_confidence",
        "lrt_candidates",
        "filename_candidates",
        "text_candidates",
    ]
    cols = [c for c in cols if c in changed.columns]
    sample = changed.sample(min(n, len(changed)), random_state=42)
    print(f"\nManual sample ({len(sample)} changed rows):")
    with pd.option_context("display.max_colwidth", 140, "display.width", 260):
        print(sample[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--courses-root", default="data/courses")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--unresolved-csv", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--sample-size", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    required = {"label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {sorted(missing)}")

    if "source_path" not in df.columns and "filename" not in df.columns:
        raise ValueError("Input parquet must contain source_path or filename")

    lrt_idx = build_lrt_index(Path(args.courses_root))
    print(f"[INFO] LRT index keys: {len(lrt_idx)}")

    meta = [choose_label(row, lrt_idx) for _, row in df.iterrows()]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(meta)], axis=1)
    out["label"] = out["label_new"]

    summary = summarize(out)

    out_path = Path(args.output)
    rep_path = Path(args.report_json)
    audit_path = Path(args.audit_csv)
    unr_path = Path(args.unresolved_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    unr_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(out_path, index=False)
    out[out["label_prev"] != out["label_new"]].to_csv(audit_path, index=False)
    out[out["relabel_reason"] == "keep_unresolved"].to_csv(unr_path, index=False)
    rep_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[INFO] Relabel complete")
    print(json.dumps(summary, indent=2))
    print_manual_sample(out, args.sample_size)


if __name__ == "__main__":
    main()
