"""
ingest/load_data.py
-------------------
Loads documents from a directory tree, supporting .md, .txt, .pdf, and .docx.

Each loaded document is enriched with metadata:
  - source_file : filename only (e.g. "faqs.md")
  - source_path : full relative path
  - file_type   : extension without the dot (e.g. "pdf")
  - category    : immediate parent folder name (e.g. "support", "legal")

One bad file never aborts the whole ingest — errors are logged and skipped.
"""

import logging
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)

# ── Dispatch table: file extension → loader class ─────────────────────────────
LOADER_MAP = {
    ".md":   TextLoader,
    ".txt":  TextLoader,
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
}

# Kwargs forwarded only to text-based loaders that accept an encoding arg
_TEXT_LOADERS = {TextLoader}


def load_documents(data_dir: str) -> list:
    """
    Recursively load all supported documents from *data_dir*.

    Supported extensions: .md, .txt, .pdf, .docx

    For every loaded ``Document`` the following metadata keys are set:

    * ``source_file``  – basename of the file (e.g. ``"faqs.md"``)
    * ``source_path``  – full path as a string
    * ``file_type``    – lowercase extension without dot (e.g. ``"pdf"``)
    * ``category``     – name of the immediate parent directory
      (e.g. ``"support"``, ``"legal"``); falls back to ``"general"``
      when the file sits directly in *data_dir*.

    Files that raise an exception during loading are skipped; a warning is
    emitted via the ``ingest.load_data`` logger so the pipeline continues.

    Args:
        data_dir: Path to the root data directory (relative or absolute).

    Returns:
        List of :class:`langchain_core.documents.Document` objects.

    Example::

        docs = load_documents("data")
        print(f"Loaded {len(docs)} document chunks")
    """
    base_path = Path(data_dir).resolve()

    if not base_path.exists():
        logger.error("Data directory does not exist: %s", base_path)
        return []

    documents: list = []
    # Track per-extension counts for the summary log
    counts: dict[str, int] = defaultdict(int)

    for ext, loader_cls in LOADER_MAP.items():
        for file_path in sorted(base_path.rglob(f"*{ext}")):
            try:
                # Build loader – only text loaders accept an encoding kwarg
                if loader_cls in _TEXT_LOADERS:
                    loader = loader_cls(str(file_path), encoding="utf-8")
                else:
                    loader = loader_cls(str(file_path))

                docs = loader.load()

                # Derive category from parent folder name
                parent = file_path.parent
                if parent == base_path:
                    category = "general"
                else:
                    category = parent.name

                # Attach rich metadata to every chunk/page
                for doc in docs:
                    doc.metadata.update(
                        {
                            "source_file": file_path.name,
                            "source_path": str(file_path),
                            "file_type": ext.lstrip("."),
                            "category": category,
                        }
                    )

                documents.extend(docs)
                counts[ext] += len(docs)
                logger.debug("Loaded %d chunk(s) from %s", len(docs), file_path.name)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping %s — could not load: %s", file_path.name, exc
                )

    # ── Summary log ───────────────────────────────────────────────────────────
    if counts:
        summary = ", ".join(
            f"{ext} → {n} chunk(s)" for ext, n in sorted(counts.items())
        )
        logger.info("Documents loaded by type: %s", summary)
    else:
        logger.warning("No documents found in: %s", base_path)

    return documents
