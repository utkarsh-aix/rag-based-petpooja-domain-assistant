"""
config/settings.py
------------------
Central configuration for the RAG chatbot.
All environment variables are loaded from a .env file at project root.
Import this module instead of hardcoding values in individual files.

Usage:
    from config.settings import settings
    print(settings.LLM_MODEL)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))


@dataclass(frozen=True)
class Settings:
    # ── API Keys ────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    VECTOR_STORE_PATH: str = "vector_store/faiss_index"
    DATA_DIR: str = "data"

    # ── Embedding model ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── LLM settings ─────────────────────────────────────────────────────────
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: int = 0

    # ── Retrieval settings ───────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 4

    # ── Text-splitting settings ───────────────────────────────────────────────
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150


# Single shared instance – import this everywhere
settings = Settings()
