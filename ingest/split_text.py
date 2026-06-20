"""
ingest/split_text.py
--------------------
Splits loaded Document objects into smaller chunks for embedding.

Chunk size and overlap default to the values in config.settings but can be
overridden per call, making this function easy to test or tune at runtime.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)


def split_documents(
    documents: list,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list:
    """
    Split a list of Documents into smaller chunks.

    In addition to the original document metadata, every returned chunk
    receives a ``chunk_index`` key that records its zero-based position
    within the full list of chunks produced from the entire document set.

    After splitting, summary statistics are emitted at INFO level:
    total chunks created and average chunk length in characters.

    Args:
        documents:     List of :class:`langchain_core.documents.Document` objects.
        chunk_size:    Maximum character length of each chunk
                       (default: ``settings.CHUNK_SIZE``).
        chunk_overlap: Number of overlapping characters between adjacent chunks
                       (default: ``settings.CHUNK_OVERLAP``).

    Returns:
        List of chunk Documents with ``chunk_index`` metadata attached.

    Example::

        chunks = split_documents(docs)
        # Each chunk.metadata now has {"chunk_index": 0, ...original fields...}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    # ── Attach positional metadata ─────────────────────────────────────────────
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    # ── Summary log ───────────────────────────────────────────────────────────
    if chunks:
        avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
        logger.info(
            "Splitting complete — %d chunk(s) created, avg length %d chars "
            "(chunk_size=%d, chunk_overlap=%d)",
            len(chunks),
            avg_len,
            chunk_size,
            chunk_overlap,
        )
    else:
        logger.warning("No chunks produced — input document list may be empty.")

    return chunks
