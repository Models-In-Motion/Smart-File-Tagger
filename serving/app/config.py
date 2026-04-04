"""
config.py — Central Configuration
Krish Jani (kj2743) — Serving Lead
ECE-GY 9183 MLOps, Spring 2026

All environment variables are read here and ONLY here.
Every other file imports from config.py instead of calling os.getenv directly.

Why centralize config?
    If DB_URL changes to DATABASE_URL, you change it in one place here,
    not in 4 different files. Also makes it easy to see all configuration
    in one spot.

Where do environment variables come from?
    Local development : .env file in the serving/ folder
    Docker            : passed via docker-compose.yml environment section
    Chameleon         : set in the VM or passed to docker run
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists (local development)
# In production (Docker/Chameleon), env vars are set directly
# load_dotenv does nothing if the file doesn't exist — safe to always call
load_dotenv()


def get_db_url() -> str | None:
    """
    PostgreSQL connection string.
    Format: postgresql://username:password@host:port/database_name
    Example: postgresql://krish:secret@localhost:5432/smarttagger
    """
    return os.getenv("DB_URL")


def get_model_path() -> str:
    """Path to the LightGBM model .pkl file."""
    return os.getenv("MODEL_PATH", "/models/lgbm_classifier_v1.pkl")


def get_label_encoder_path() -> str:
    """Path to the scikit-learn label encoder .pkl file."""
    return os.getenv("LABEL_ENCODER_PATH", "/models/label_encoder.pkl")


def get_nextcloud_url() -> str | None:
    """
    Base URL of the Nextcloud instance.
    Used by serving layer to write tags back via Nextcloud REST API.
    Example: http://nextcloud:80
    """
    return os.getenv("NEXTCLOUD_URL")


def get_nextcloud_admin_password() -> str | None:
    """Admin password for Nextcloud REST API calls."""
    return os.getenv("NEXTCLOUD_ADMIN_PASSWORD")


def is_debug_mode() -> bool:
    """
    When True, FastAPI shows detailed error messages.
    Must be False in production.
    """
    return os.getenv("DEBUG", "false").lower() == "true"