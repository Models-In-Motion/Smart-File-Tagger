#!/usr/bin/env python3
"""
synthetic_expansion.py

Expands the OCW dataset with synthetic training samples using label-preserving
text augmentation techniques. Required when dataset is under 5GB.

Best practices followed:
- All augmentations are label-preserving (document type does not change)
- Full lineage tracking: source="synthetic", augmentation_method recorded
- Reproducible via --seed
- Synthetic rows are clearly distinguishable from real data (source field)
- Recomputed derived fields (char_count, word_count, doc_id)

Augmentation techniques (one per synthetic copy):
  1. sentence_dropout   — drop 20-30% of sentences randomly
  2. sentence_shuffle   — shuffle middle 60% of sentences (keep first/last)
  3. word_dropout       — randomly remove 5-10% of words
  4. truncation         — take a random 60-80% contiguous window of sentences
  5. combined           — sentence_dropout + word_dropout together

Usage:
    python synthetic_expansion.py
    python synthetic_expansion.py --multiplier 5 --seed 42
"""

import argparse
import hashlib
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

AUGMENTATION_METHODS = [
    "sentence_dropout",
    "sentence_shuffle",
    "word_dropout",
    "truncation",
    "combined",
]


# ---------------------------------------------------------------------------
# Text augmentation primitives
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation + newline heuristics."""
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def join_sentences(sentences: list[str]) -> str:
    return " ".join(sentences)


def sentence_dropout(text: str, rng: random.Random, drop_rate: float | None = None) -> str:
    """Remove drop_rate fraction of sentences at random (default 20-30%)."""
    if drop_rate is None:
        drop_rate = rng.uniform(0.20, 0.30)
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return text  # too short to drop from
    kept = [s for s in sentences if rng.random() > drop_rate]
    if not kept:
        kept = sentences[:1]  # always keep at least one
    return join_sentences(kept)


def sentence_shuffle(text: str, rng: random.Random) -> str:
    """Shuffle the middle 60% of sentences; keep first and last in place."""
    sentences = split_sentences(text)
    if len(sentences) < 4:
        return text
    n = len(sentences)
    start = max(1, n // 5)
    end = min(n - 1, n - n // 5)
    middle = sentences[start:end]
    rng.shuffle(middle)
    result = sentences[:start] + middle + sentences[end:]
    return join_sentences(result)


def word_dropout(text: str, rng: random.Random, drop_rate: float | None = None) -> str:
    """Randomly remove drop_rate fraction of words (default 5-10%)."""
    if drop_rate is None:
        drop_rate = rng.uniform(0.05, 0.10)
    words = text.split()
    if len(words) < 20:
        return text
    kept = [w for w in words if rng.random() > drop_rate]
    if not kept:
        kept = words[:5]
    return " ".join(kept)


def truncation(text: str, rng: random.Random) -> str:
    """Take a random contiguous 60-80% window of sentences."""
    sentences = split_sentences(text)
    if len(sentences) < 5:
        return text
    keep_frac = rng.uniform(0.60, 0.80)
    keep_n = max(3, int(len(sentences) * keep_frac))
    max_start = len(sentences) - keep_n
    start = rng.randint(0, max(0, max_start))
    return join_sentences(sentences[start:start + keep_n])


def combined(text: str, rng: random.Random) -> str:
    """Sentence dropout followed by word dropout — harder augmentation."""
    text = sentence_dropout(text, rng, drop_rate=rng.uniform(0.15, 0.25))
    text = word_dropout(text, rng, drop_rate=rng.uniform(0.05, 0.08))
    return text


AUGMENT_FN = {
    "sentence_dropout": sentence_dropout,
    "sentence_shuffle": sentence_shuffle,
    "word_dropout": word_dropout,
    "truncation": truncation,
    "combined": combined,
}


# ---------------------------------------------------------------------------
# Synthetic row builder
# ---------------------------------------------------------------------------

def make_synthetic_row(original: dict, method: str, copy_idx: int, rng: random.Random) -> dict:
    """Apply augmentation and build a new row with full lineage info."""
    aug_fn = AUGMENT_FN[method]
    new_text = aug_fn(original["extracted_text"], rng)

    # Ensure augmentation actually changed the text
    if new_text == original["extracted_text"]:
        new_text = sentence_dropout(new_text, rng, drop_rate=0.20)

    row = dict(original)
    row["extracted_text"] = new_text
    # Derive from lineage so two different augmentations cannot collide on short text hashes.
    row["doc_id"] = hashlib.md5(
        f"{original['doc_id']}|{method}|{copy_idx}".encode()
    ).hexdigest()[:12]
    row["char_count"] = len(new_text)
    row["word_count"] = len(new_text.split())
    row["source"] = "synthetic"
    row["source_url"] = original["source_url"] + f"#synthetic_{copy_idx}"
    row["text_extraction_method"] = f"synthetic_{method}"
    row["ingestion_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand OCW dataset with synthetic label-preserving augmentations."
    )
    parser.add_argument("--input", default="artifacts/ocw_dataset.parquet")
    parser.add_argument("--output", default="artifacts/ocw_dataset_expanded.parquet")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Number of synthetic copies per original row (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df)} original rows from {args.input}")

    # Only augment rows with real, extractable text (skip very short ones)
    eligible = df[df["char_count"] >= 300].copy()
    skipped = len(df) - len(eligible)
    if skipped:
        print(f"[INFO] Skipping {skipped} rows with char_count < 300 (too short to augment)")

    methods = AUGMENTATION_METHODS[:args.multiplier]  # cycle if multiplier > 5
    if args.multiplier > len(AUGMENTATION_METHODS):
        # Repeat methods cyclically for multiplier > 5
        methods = [
            AUGMENTATION_METHODS[i % len(AUGMENTATION_METHODS)]
            for i in range(args.multiplier)
        ]

    synthetic_rows = []
    for _, row in eligible.iterrows():
        original = row.to_dict()
        for copy_idx, method in enumerate(methods, start=1):
            synthetic_rows.append(make_synthetic_row(original, method, copy_idx, rng))

    synthetic_df = pd.DataFrame(synthetic_rows)

    # Combine original + synthetic
    expanded_df = pd.concat([df, synthetic_df], ignore_index=True)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_df.to_parquet(out_path, index=False)

    # Summary
    print(f"\n=== Synthetic Expansion Summary ===")
    print(f"Original rows:   {len(df)}")
    print(f"Synthetic rows:  {len(synthetic_df)}")
    print(f"Total rows:      {len(expanded_df)}")
    print(f"Multiplier:      {args.multiplier}x  |  Methods: {methods}")
    print(f"\nLabel distribution (expanded):")
    for label, count in expanded_df["label"].value_counts().items():
        orig_count = df["label"].value_counts().get(label, 0)
        print(f"  {label:<20} {count:>5}  (original: {orig_count})")
    print(f"\nSource breakdown:")
    for src, count in expanded_df["source"].value_counts().items():
        print(f"  {src:<20} {count:>5}")
    print(f"\n[OK] Written to {args.output}")


if __name__ == "__main__":
    main()
