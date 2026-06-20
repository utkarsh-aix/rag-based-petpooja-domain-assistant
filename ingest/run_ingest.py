"""
ingest/run_ingest.py
--------------------
Entry point for the document ingestion pipeline.

Run from the project root (either form works):
    python -m ingest.run_ingest
    python ingest/run_ingest.py
"""

import logging
import sys
from pathlib import Path

# ── Ensure project root is on sys.path so absolute imports always work ─────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import settings
from ingest.embed_store import create_vector_store
from ingest.load_data import load_documents
from ingest.split_text import split_documents

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    """
    Execute the full ingestion pipeline:

    1. Load documents from the configured data directory.
    2. Split them into chunks.
    3. Embed chunks and persist the FAISS vector store.
    """
    logger.info("Starting ingestion pipeline")

    logger.info("Step 1/3 — Loading documents from '%s'", settings.DATA_DIR)
    documents = load_documents(settings.DATA_DIR)
    logger.info("Loaded %d document chunk(s) in total", len(documents))

    if not documents:
        logger.error("No documents loaded — aborting ingestion.")
        sys.exit(1)

    logger.info("Step 2/3 — Splitting documents into chunks")
    chunks = split_documents(documents)
    logger.info("Created %d chunk(s)", len(chunks))

    logger.info(
        "Step 3/3 — Creating embeddings and saving FAISS index to '%s'",
        settings.VECTOR_STORE_PATH,
    )
    create_vector_store(chunks, settings.VECTOR_STORE_PATH)

    logger.info("Ingestion completed successfully ✓")


if __name__ == "__main__":
    run_ingestion()
