"""
chatbot/retriever.py
--------------------
Loads the FAISS vector store and embedding model ONCE and caches them for the
lifetime of the process (or Streamlit session).

Caching strategy
~~~~~~~~~~~~~~~~
* If Streamlit is available (i.e. the app is running under ``streamlit run``),
  ``@st.cache_resource`` is used so the index survives hot-reloads and is
  shared across all browser sessions.
* Otherwise a plain module-level singleton dict is used — sufficient for
  CLI usage, scripts, and tests.

Public API
~~~~~~~~~~
    get_retriever(top_k, search_type)  – Return a cached LangChain retriever.
    reload_retriever()                 – Bust the cache and force a fresh load
                                        (call this after re-ingestion).
"""

import logging
import os
from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Detect whether we are running inside a Streamlit process ──────────────────
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

# ── Module-level singleton fallback (used when Streamlit is not available) ────
_cache: dict = {}          # keys: "embeddings", "vector_store"


# ── Internal loaders ──────────────────────────────────────────────────────────

def _build_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the HuggingFace embedding model."""
    logger.info("Loading embedding model '%s'", settings.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def _build_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Load the FAISS index from disk.

    Raises:
        FileNotFoundError: If the index directory or ``index.faiss`` file is absent.
        RuntimeError:      If loading fails for any other reason.
    """
    store_path = settings.VECTOR_STORE_PATH
    index_file = os.path.join(store_path, "index.faiss")

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Vector store not found at '{os.path.abspath(store_path)}'. "
            "Run `python -m ingest.run_ingest` first to build the index."
        )

    try:
        logger.info("Loading FAISS index from '%s'", store_path)
        vector_store = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded successfully")
        return vector_store
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FAISS index from '{store_path}': {exc}"
        ) from exc


# ── Streamlit-cached loader (only defined when st is importable) ──────────────

if _STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner="Loading knowledge base…")
    def _get_cached_vector_store() -> FAISS:
        """Load and cache embeddings + FAISS index for the Streamlit session."""
        embeddings = _build_embeddings()
        return _build_vector_store(embeddings)


# ── Singleton loader (non-Streamlit path) ─────────────────────────────────────

def _get_singleton_vector_store() -> FAISS:
    """Return the module-level cached vector store, building it on first call."""
    if "vector_store" not in _cache:
        embeddings = _build_embeddings()
        _cache["embeddings"] = embeddings
        _cache["vector_store"] = _build_vector_store(embeddings)
    return _cache["vector_store"]


# ── Public API ────────────────────────────────────────────────────────────────

def get_retriever(
    top_k: int = settings.TOP_K_RETRIEVAL,
    search_type: Literal["similarity", "mmr"] = "similarity",
):
    """
    Return a cached LangChain retriever backed by the FAISS vector store.

    The underlying embeddings and index are loaded **once** and reused on
    every subsequent call, eliminating repeated disk I/O.

    Args:
        top_k:       Number of documents to retrieve per query
                     (default: ``settings.TOP_K_RETRIEVAL``).
        search_type: Retrieval strategy to use:

                     * ``"similarity"`` *(default)* – standard cosine/dot-product
                       nearest-neighbour search.
                     * ``"mmr"`` – Maximum Marginal Relevance; trades a small
                       amount of pure relevance for greater diversity in the
                       returned documents. Useful when the top results tend to
                       be near-duplicates.

    Returns:
        A LangChain ``VectorStoreRetriever`` instance.

    Raises:
        FileNotFoundError: If the vector store has not been built yet.
        ValueError:        If an unsupported *search_type* is supplied.

    Example::

        retriever = get_retriever(top_k=6, search_type="mmr")
        docs = retriever.invoke("What is the refund policy?")
    """
    if search_type not in ("similarity", "mmr"):
        raise ValueError(
            f"Unsupported search_type '{search_type}'. "
            "Choose 'similarity' or 'mmr'."
        )

    # Obtain the cached vector store via the appropriate strategy
    if _STREAMLIT_AVAILABLE:
        vector_store = _get_cached_vector_store()
    else:
        vector_store = _get_singleton_vector_store()

    search_kwargs: dict = {"k": top_k}

    # MMR accepts an optional fetch_k (pool size before re-ranking); we default
    # to 4× top_k to give the algorithm enough candidates.
    if search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 4

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    logger.debug(
        "Retriever ready — search_type=%s, top_k=%d", search_type, top_k
    )
    return retriever


def reload_retriever() -> None:
    """
    Bust the vector store cache and force a fresh load from disk on the next
    call to :func:`get_retriever`.

    Call this after re-running the ingestion pipeline so the chatbot picks up
    newly indexed documents without restarting the process.

    Example::

        run_ingestion()       # rebuild the FAISS index
        reload_retriever()    # invalidate the old cached index
        retriever = get_retriever()   # loads the fresh index
    """
    if _STREAMLIT_AVAILABLE:
        _get_cached_vector_store.clear()
        logger.info("Streamlit cache cleared — retriever will reload on next call")
    else:
        _cache.clear()
        logger.info("Module-level cache cleared — retriever will reload on next call")
