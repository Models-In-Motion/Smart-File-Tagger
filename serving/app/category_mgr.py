"""
category_mgr.py — Custom Category Management
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

This file handles Layer 2 of the system: user-defined custom categories.

How custom categories work (plain English):
    1. A user says "I want a category called Grant Applications"
    2. They pick 3-5 example files they already have
    3. We run each file through SBERT to get an embedding (384 numbers)
    4. We average those embeddings into one "prototype vector"
    5. We store that prototype vector in PostgreSQL for this user
    6. When a new file comes in, we compare it against all prototype vectors
       using cosine similarity — a score from 0 (completely different) to 1
       (identical meaning)
    7. If the new file is similar enough to a prototype, that's the label

Why cosine similarity?
    Two documents can have very different lengths but the same topic.
    Cosine similarity compares the *direction* of two vectors, not their
    magnitude. So a short abstract and a full paper about the same topic
    will still be recognized as similar.

Where are prototype vectors stored?
    In PostgreSQL as a JSON array. Simple, works, no vector database needed
    for our scale. If we ever have millions of categories we'd switch to
    pgvector or Qdrant, but that's premature for this project.
"""

import json
import logging
import re
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import get_db_url

log = logging.getLogger(__name__)

# Minimum cosine similarity to consider a custom category a match.
# Below this threshold, the custom category is ignored and the fixed
# baseline classifier takes over.
CUSTOM_CATEGORY_THRESHOLD = 0.60

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "will",
    "your", "are", "not", "but", "all", "can", "been", "their", "which",
    "when", "were", "they", "class", "classes", "school", "course", "endobj", "stream", "filter", "length", "flatedecode", "startxref", "trailer", "xref", "author", "title", "creator", "producer", "courseleaf",
}


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def _get_connection():
    import psycopg2
    db_url = get_db_url()
    if db_url is None:
        raise RuntimeError("DB_URL environment variable is not set.")
    return psycopg2.connect(db_url)


def preprocess_for_sbert(text: str) -> str:
    """
    Light preprocessing before SBERT encoding.
    We avoid aggressive filtering because pdfminer produces continuous
    text without newlines — splitting heuristics tend to cut meaningful
    phrases and hurt similarity scores more than they help.
    Just return the text as-is, capped at 500 chars.
    """
    if not text:
        return ""
    return text[:500]


def _extract_keywords(example_texts: list[str], limit: int = 10) -> list[str]:
    """
    Pull frequent non-stopword tokens from user examples.
    """
    token_counts: Counter[str] = Counter()
    for text in example_texts:
        normalized = preprocess_for_sbert(text).lower()
        words = re.findall(r"[a-z]{5,}", normalized)
        token_counts.update(word for word in words if word not in STOPWORDS)

    return [word for word, _ in token_counts.most_common(limit)]


def ensure_categories_table_exists():
    """
    Creates the custom_categories table if it doesn't exist.
    Called once at server startup.
    """
    sql = """
        CREATE TABLE IF NOT EXISTS custom_categories (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT    NOT NULL,
            category_name   TEXT    NOT NULL,
            prototype_vector TEXT   NOT NULL,  -- JSON array of 384 floats
            keywords        JSONB   NOT NULL DEFAULT '[]'::jsonb,
            example_count   INT     NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            -- Each user can only have one category with a given name
            UNIQUE (user_id, category_name)
        );

        CREATE INDEX IF NOT EXISTS idx_categories_user_id
            ON custom_categories (user_id);

        ALTER TABLE custom_categories
            ADD COLUMN IF NOT EXISTS keywords JSONB NOT NULL DEFAULT '[]'::jsonb;
    """
    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        conn.close()
        log.info("Custom categories table ready")
    except Exception as exc:
        log.error(f"Could not create categories table: {exc}")


# ---------------------------------------------------------------------------
# Register a new custom category
# ---------------------------------------------------------------------------

def register_category(
    user_id:        str,
    category_name:  str,
    example_texts:  list[str],
    sbert_model,
) -> dict:
    """
    Creates a new custom category from a set of example documents.

    Args:
        user_id       : Nextcloud user ID
        category_name : user-chosen name, e.g. "Grant Applications"
        example_texts : list of extracted texts from 3-5 example files
        sbert_model   : the loaded SentenceTransformer model (passed in
                        so we don't load it twice — it's already in memory
                        inside the Predictor class)

    Returns:
        Dict with success status and message.

    Steps:
        1. Encode each example text with SBERT → list of embeddings
        2. Average all embeddings into one prototype vector
        3. Store the prototype vector in PostgreSQL
    """
    if len(example_texts) < 3:
        return {
            "success": False,
            "message": f"Need at least 3 example files, got {len(example_texts)}"
        }

    if len(example_texts) > 10:
        return {
            "success": False,
            "message": "Maximum 10 example files allowed"
        }

    # Step 1 — encode all examples
    log.info(f"Encoding {len(example_texts)} examples for category '{category_name}'")
    processed_examples = [preprocess_for_sbert(text) for text in example_texts]
    embeddings = sbert_model.encode(
        processed_examples,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # embeddings shape: (n_examples, 384)

    # Step 2 — average into prototype vector
    prototype = np.mean(embeddings, axis=0).tolist()
    # prototype is now a Python list of 384 floats
    keywords = _extract_keywords(example_texts, limit=10)

    # Step 3 — store in PostgreSQL
    sql = """
        INSERT INTO custom_categories (user_id, category_name, prototype_vector, keywords, example_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, category_name)
        DO UPDATE SET
            prototype_vector = EXCLUDED.prototype_vector,
            keywords         = EXCLUDED.keywords,
            example_count    = EXCLUDED.example_count,
            created_at       = NOW()
    """
    # ON CONFLICT ... DO UPDATE means: if the user already has a category
    # with this name, overwrite it instead of throwing an error.
    # Useful when a user wants to improve their category by adding more examples.

    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    user_id,
                    category_name,
                    json.dumps(prototype),
                    json.dumps(keywords),
                    len(example_texts),
                ))
        conn.close()
        log.info(f"Category '{category_name}' registered for user {user_id}")
        return {
            "success": True,
            "message": f"Category '{category_name}' created with {len(example_texts)} examples"
        }

    except Exception as exc:
        log.error(f"Failed to register category: {exc}")
        return {"success": False, "message": f"Database error: {exc}"}


# ---------------------------------------------------------------------------
# Find best matching custom category for a new document
# ---------------------------------------------------------------------------

def find_best_custom_category(
    user_id:    str,
    text:       str,
    sbert_model,
) -> tuple[str | None, float]:
    """
    Checks if a document matches any of the user's custom categories.

    Args:
        user_id     : Nextcloud user ID
        text        : extracted text from the new document
        sbert_model : the loaded SentenceTransformer model

    Returns:
        (category_name, similarity_score) if a match is found above threshold
        (None, 0.0) if no custom category matches well enough

    How it works:
        1. Load all prototype vectors for this user from PostgreSQL
        2. Encode the new document with SBERT
        3. Compute cosine similarity between new doc and each prototype
        4. Return the best match if it's above CUSTOM_CATEGORY_THRESHOLD
    """
    categories = _load_user_categories(user_id)
    if not categories:
        return None, 0.0

    # Encode the new document
    processed_text = preprocess_for_sbert(text)
    doc_embedding = sbert_model.encode(
        processed_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).reshape(1, -1)
    # shape: (1, 384)
    doc_word_set = set(re.findall(r"[a-z]{5,}", text.lower()))

    best_name  = None
    best_score = 0.0

    for category in categories:
        prototype = np.array(
            json.loads(category["prototype_vector"])
        ).reshape(1, -1)
        # shape: (1, 384)

        # cosine_similarity returns a (1,1) array — get the scalar value
        sbert_score = float(cosine_similarity(doc_embedding, prototype)[0][0])
        stored_keywords = category.get("keywords") or []
        if isinstance(stored_keywords, str):
            try:
                stored_keywords = json.loads(stored_keywords)
            except Exception:
                stored_keywords = []
        keyword_hits = sum(
            1 for keyword in stored_keywords
            if isinstance(keyword, str) and keyword.lower() in doc_word_set
        )
        boost = min(0.15, keyword_hits * 0.03)
        score = sbert_score + boost
        log.info(f"Category '{category['category_name']}': sbert={sbert_score:.3f} keywords={keyword_hits} boost={boost:.3f} final={score:.3f}")

        if score > best_score:
            best_score = score
            best_name  = category["category_name"]

    if best_score >= CUSTOM_CATEGORY_THRESHOLD:
        log.info(
            f"Custom category match: '{best_name}' "
            f"score={best_score:.3f} user={user_id}"
        )
        return best_name, best_score
    else:
        return None, 0.0


# ---------------------------------------------------------------------------
# List and delete categories
# ---------------------------------------------------------------------------

def list_user_categories(user_id: str) -> list[dict]:
    """
    Returns all custom categories for a user.
    Does NOT return the prototype vectors (too large, not useful to display).
    """
    categories = _load_user_categories(user_id)
    return [
        {
            "category_name": c["category_name"],
            "example_count": c["example_count"],
            "created_at":    str(c["created_at"]),
        }
        for c in categories
    ]


def delete_category(user_id: str, category_name: str) -> bool:
    """
    Deletes a custom category. Returns True if deleted, False on error.
    """
    sql = """
        DELETE FROM custom_categories
        WHERE user_id = %s AND category_name = %s
    """
    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, category_name))
        conn.close()
        log.info(f"Category '{category_name}' deleted for user {user_id}")
        return True
    except Exception as exc:
        log.error(f"Failed to delete category: {exc}")
        return False


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _load_user_categories(user_id: str) -> list[dict]:
    """
    Loads all custom categories (including prototype vectors) for a user.
    """
    sql = """
        SELECT category_name, prototype_vector, keywords, example_count, created_at
        FROM custom_categories
        WHERE user_id = %s
        ORDER BY created_at DESC
    """
    try:
        from psycopg2.extras import RealDictCursor
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as exc:
        log.error(f"Failed to load categories for user {user_id}: {exc}")
        return []