#!/usr/bin/env python3
"""
qdrant_demo.py

Bonus deliverable: demonstrates using Qdrant as a vector database to store
and search SBERT document embeddings for the "user-defined custom categories"
feature of the Smart File Tagger system.

Why Qdrant over PostgreSQL (pgvector):
  - Native ANN (approximate nearest neighbor) indexing via HNSW — sub-millisecond
    similarity search even at 100k+ vectors, without full table scans.
  - Payload filtering lets us scope searches to a user's own prototypes without
    a JOIN across a user table — critical for per-user custom category isolation.
  - Built-in collection versioning and snapshotting aligns with our data
    versioning strategy, unlike generic blob storage in PostgreSQL.

Design:
  - Fixed baseline labels (Lecture Notes, Exam, etc.) → stored once at startup
    from the OCW dataset, used for production classification.
  - User-defined custom categories → averaged SBERT embeddings of 3-5 example
    files provided by the user, stored as prototype vectors per user_id.

Usage:
    # Start Qdrant first:
    docker compose up -d qdrant

    # Run demo:
    docker compose run --rm qdrant-demo
    # or locally:
    python qdrant_demo.py --input artifacts/ocw_dataset.parquet
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "ocw_prototypes"
VECTOR_DIM = 384        # all-MiniLM-L6-v2 output dimension
TEXT_CHARS = 512        # truncate input to first N chars for speed
SAMPLE_SIZE = 50        # docs to load into Qdrant for the demo
PROTOTYPE_K = 3        # example docs per custom category prototype
TOP_K = 5              # nearest neighbors to retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def wait_for_qdrant(client: QdrantClient, retries: int = 10, delay: float = 2.0) -> None:
    for i in range(retries):
        try:
            client.get_collections()
            return
        except Exception:
            if i < retries - 1:
                print(f"[INFO] Waiting for Qdrant... ({i+1}/{retries})")
                time.sleep(delay)
    print("[ERROR] Could not connect to Qdrant after retries.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Dropped existing collection '{COLLECTION_NAME}'")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[INFO] Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}, cosine)")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_documents(
    client: QdrantClient,
    model: SentenceTransformer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Encode and upsert SAMPLE_SIZE documents into Qdrant."""
    # Sample evenly across labels for a representative demo
    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), SAMPLE_SIZE // df["label"].nunique() + 1), random_state=42))
        .head(SAMPLE_SIZE)
        .reset_index(drop=True)
    )

    texts = [row["extracted_text"][:TEXT_CHARS] for _, row in sampled.iterrows()]
    print(f"[INFO] Encoding {len(sampled)} documents with SBERT...")
    vectors = encode(model, texts)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "doc_id":     row["doc_id"],
                "label":      row["label"],
                "course_id":  row["course_id"],
                "department": row["department"],
                "filename":   row["filename"],
                "source":     row["source"],
            },
        )
        for i, (_, row) in enumerate(sampled.iterrows())
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[INFO] Upserted {len(points)} vectors into '{COLLECTION_NAME}'")
    return sampled


# ---------------------------------------------------------------------------
# Demo 1: nearest-neighbor search
# ---------------------------------------------------------------------------

def demo_nearest_neighbor(
    client: QdrantClient,
    model: SentenceTransformer,
    sampled: pd.DataFrame,
) -> None:
    query_row = sampled.iloc[0]
    query_text = query_row["extracted_text"][:TEXT_CHARS]
    query_vec = encode(model, [query_text])[0].tolist()

    print(f"\n{'='*60}")
    print(f"DEMO 1: Nearest-neighbor search")
    print(f"Query document: '{query_row['filename']}'")
    print(f"True label:     {query_row['label']}")
    print(f"Top-{TOP_K} nearest neighbors:")

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=TOP_K + 1,  # +1 because the query doc itself will be #1
    )

    for rank, hit in enumerate(results, start=1):
        p = hit.payload
        marker = " ← query doc" if p["doc_id"] == query_row["doc_id"] else ""
        print(f"  {rank}. [{p['label']:<15}] score={hit.score:.4f}  {p['filename']}{marker}")


# ---------------------------------------------------------------------------
# Demo 2: user-defined custom category via prototype averaging
# ---------------------------------------------------------------------------

def demo_custom_category(
    client: QdrantClient,
    model: SentenceTransformer,
    sampled: pd.DataFrame,
) -> None:
    # Simulate a user defining a custom "Problem Set" category from 3 examples
    category_label = "Problem Set"
    examples = sampled[sampled["label"] == category_label].head(PROTOTYPE_K)

    if len(examples) < PROTOTYPE_K:
        print(f"\n[SKIP] Not enough '{category_label}' docs in sample for prototype demo")
        return

    print(f"\n{'='*60}")
    print(f"DEMO 2: User-defined custom category — '{category_label}'")
    print(f"User provides {PROTOTYPE_K} example files:")
    for _, row in examples.iterrows():
        print(f"  - {row['filename']}")

    # Encode examples and average to create prototype vector
    example_texts = [row["extracted_text"][:TEXT_CHARS] for _, row in examples.iterrows()]
    example_vecs = encode(model, example_texts)
    prototype_vec = example_vecs.mean(axis=0)
    prototype_vec = prototype_vec / np.linalg.norm(prototype_vec)  # renormalize

    # Store prototype as a named vector (simulate per-user storage)
    prototype_id = 9999
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=prototype_id,
            vector=prototype_vec.tolist(),
            payload={
                "doc_id":    "user_prototype",
                "label":     f"custom::{category_label}",
                "course_id": "user_defined",
                "department": "user_defined",
                "filename":  f"prototype_{category_label.replace(' ', '_')}",
                "source":    "user_prototype",
            },
        )]
    )

    # Search for closest documents to this prototype
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=prototype_vec.tolist(),
        limit=TOP_K,
    )

    print(f"\nDocuments closest to '{category_label}' prototype:")
    correct = 0
    for rank, hit in enumerate(results, start=1):
        p = hit.payload
        match = "✓" if p["label"] == category_label else "✗"
        if p["label"] == category_label:
            correct += 1
        print(f"  {rank}. {match} [{p['label']:<15}] score={hit.score:.4f}  {p['filename']}")

    print(f"\nPrototype retrieval precision@{TOP_K}: {correct}/{TOP_K} correct label matches")
    print(f"→ Qdrant enables per-user prototype storage + sub-ms ANN search.")
    print(f"  PostgreSQL would require a full pgvector table scan across all users' vectors.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant vector DB demo for document prototype search.")
    parser.add_argument("--input", default="artifacts/ocw_dataset.parquet")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    args = parser.parse_args()

    print(f"[INFO] Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}")
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    wait_for_qdrant(client)

    print(f"[INFO] Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df)} rows from {args.input}")

    setup_collection(client)
    sampled = ingest_documents(client, model, df)

    demo_nearest_neighbor(client, model, sampled)
    demo_custom_category(client, model, sampled)

    print(f"\n{'='*60}")
    print(f"[OK] Qdrant demo complete.")
    print(f"     Collection '{COLLECTION_NAME}' has {client.get_collection(COLLECTION_NAME).points_count} vectors.")
    print(f"\nWhy this improves the design:")
    print(f"  1. HNSW index: sub-ms ANN search vs O(n) pgvector scan")
    print(f"  2. Payload filtering: isolate per-user prototypes without SQL JOINs")
    print(f"  3. Native snapshot/restore: versioned prototype store aligned with")
    print(f"     our dataset versioning strategy (artifacts/versions/v1/)")


if __name__ == "__main__":
    main()
