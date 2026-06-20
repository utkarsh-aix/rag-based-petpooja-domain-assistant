"""
ingest/embed_store.py
---------------------
Creates, updates, and inspects the FAISS vector store used by the retriever.

Functions:
    create_vector_store      – Build a new FAISS index from scratch and persist it.
    update_vector_store      – Incrementally add new documents to an existing index
                               (creates one if it does not yet exist).
    get_vector_store_stats   – Return a dict with diagnostic info about the stored index.
"""

import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Internal helper ───────────────────────────────────────────────────────────

def _load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Instantiate the embedding model (cached by LangChain internally)."""
    return HuggingFaceEmbeddings(model_name=model_name)


def _load_existing_index(store_path: str, embeddings: HuggingFaceEmbeddings) -> FAISS | None:
    """
    Try to load a FAISS index from *store_path*.

    Returns the index on success, or ``None`` if no index exists yet.
    Raises ``RuntimeError`` if the path exists but loading fails.
    """
    index_file = os.path.join(store_path, "index.faiss")
    if not os.path.exists(index_file):
        return None

    try:
        vector_store = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded existing FAISS index from '%s'", store_path)
        return vector_store
    except Exception as exc:
        raise RuntimeError(
            f"Found index files at '{store_path}' but failed to load them: {exc}"
        ) from exc


def _save_index(vector_store: FAISS, store_path: str) -> None:
    """Persist *vector_store* to *store_path*, creating directories as needed."""
    try:
        os.makedirs(store_path, exist_ok=True)
        vector_store.save_local(store_path)
        logger.info("FAISS index saved to '%s'", store_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save FAISS index to '{store_path}': {exc}"
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def create_vector_store(
    documents: list,
    store_path: str = settings.VECTOR_STORE_PATH,
    embedding_model: str = settings.EMBEDDING_MODEL,
) -> FAISS:
    """
    Build a new FAISS vector store from *documents* and persist it to disk.

    .. warning::
        This **overwrites** any existing index at *store_path*.
        Use :func:`update_vector_store` to add to an existing index.

    Args:
        documents:       List of :class:`langchain_core.documents.Document` objects.
        store_path:      Directory where the index files will be saved
                         (default: ``settings.VECTOR_STORE_PATH``).
        embedding_model: HuggingFace model name for embeddings
                         (default: ``settings.EMBEDDING_MODEL``).

    Returns:
        The in-memory :class:`FAISS` vector store instance.

    Raises:
        ValueError:   If *documents* is empty.
        RuntimeError: If saving the index to disk fails.
    """
    if not documents:
        raise ValueError("Cannot create a vector store from an empty document list.")

    logger.info(
        "Creating new FAISS index from %d document(s) using model '%s'",
        len(documents),
        embedding_model,
    )

    try:
        embeddings = _load_embeddings(embedding_model)
        vector_store = FAISS.from_documents(documents, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Failed to build FAISS index: {exc}") from exc

    _save_index(vector_store, store_path)
    return vector_store


def update_vector_store(
    new_documents: list,
    store_path: str = settings.VECTOR_STORE_PATH,
    embedding_model: str = settings.EMBEDDING_MODEL,
) -> FAISS:
    """
    Add *new_documents* to an existing FAISS index, or create one if absent.

    The workflow is:
    1. Attempt to load the index at *store_path*.
    2. If found, call ``vector_store.add_documents(new_documents)``.
    3. If not found, create a fresh index via :func:`create_vector_store`.
    4. Persist the updated index back to disk.

    Args:
        new_documents:   Documents to add.
        store_path:      Path to the existing (or new) FAISS index directory
                         (default: ``settings.VECTOR_STORE_PATH``).
        embedding_model: HuggingFace model name for embeddings
                         (default: ``settings.EMBEDDING_MODEL``).

    Returns:
        The updated (or newly created) :class:`FAISS` vector store instance.

    Raises:
        ValueError:   If *new_documents* is empty.
        RuntimeError: If loading or saving the index fails.
    """
    if not new_documents:
        raise ValueError("new_documents must not be empty.")

    embeddings = _load_embeddings(embedding_model)
    vector_store = _load_existing_index(store_path, embeddings)

    if vector_store is not None:
        logger.info(
            "Updating existing index — adding %d new document(s)", len(new_documents)
        )
        try:
            vector_store.add_documents(new_documents)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to add documents to existing FAISS index: {exc}"
            ) from exc
    else:
        logger.info(
            "No existing index found at '%s' — creating a new one", store_path
        )
        vector_store = create_vector_store(new_documents, store_path, embedding_model)
        return vector_store  # already saved inside create_vector_store

    _save_index(vector_store, store_path)
    return vector_store


def get_vector_store_stats(store_path: str = settings.VECTOR_STORE_PATH) -> dict:
    """
    Return diagnostic information about the persisted FAISS index.

    The returned dict always contains an ``"exists"`` key. When the index is
    present, it also includes ``"vector_count"`` and ``"store_path"``.

    Args:
        store_path: Directory of the FAISS index
                    (default: ``settings.VECTOR_STORE_PATH``).

    Returns:
        Dictionary with the following keys:

        * ``"exists"`` (bool)       – Whether a valid index was found.
        * ``"vector_count"`` (int)  – Number of vectors in the index.
        * ``"store_path"`` (str)    – Resolved absolute path that was checked.
        * ``"error"`` (str)         – Present only if loading raised an exception.

    Example::

        stats = get_vector_store_stats()
        # {"exists": True, "vector_count": 1024, "store_path": "/abs/path/..."}
    """
    abs_path = os.path.abspath(store_path)
    index_file = os.path.join(abs_path, "index.faiss")

    if not os.path.exists(index_file):
        logger.info("No FAISS index found at '%s'", abs_path)
        return {"exists": False, "store_path": abs_path}

    try:
        embeddings = _load_embeddings(settings.EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            abs_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        count = vector_store.index.ntotal
        logger.info("FAISS index at '%s' contains %d vector(s)", abs_path, count)
        return {"exists": True, "vector_count": count, "store_path": abs_path}
    except Exception as exc:
        logger.error("Failed to read FAISS stats from '%s': %s", abs_path, exc)
        return {"exists": True, "vector_count": -1, "store_path": abs_path, "error": str(exc)}
