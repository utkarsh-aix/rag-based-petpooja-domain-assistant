"""
tests/conftest.py
-----------------
Shared pytest fixtures for the RAG chatbot test suite.

The fixtures here allow tests to run without a real FAISS index on disk,
without hitting the Google Gemini API, and without loading the embedding
model — keeping the suite fast and fully offline.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Fake document factory ─────────────────────────────────────────────────────

def _make_doc(content: str, source: str = "faqs.md", category: str = "support") -> Document:
    """Return a minimal Document with metadata that matches the real ingest output."""
    return Document(
        page_content=content,
        metadata={
            "source_file": source,
            "source_path": f"data/{category}/{source}",
            "file_type": source.rsplit(".", 1)[-1],
            "category": category,
            "chunk_index": 0,
        },
    )


FAKE_DOCS = [
    _make_doc(
        "Petpooja is a leading restaurant POS and management platform.",
        source="about.md",
        category="company",
    ),
    _make_doc(
        "The POS software supports billing, inventory, and online order management.",
        source="product_overview.md",
        category="product",
    ),
]


# ── Retriever fixture ─────────────────────────────────────────────────────────

@pytest.fixture()
def mock_retriever():
    """
    Return a MagicMock retriever whose ``.invoke()`` returns FAKE_DOCS.

    Patches ``chatbot.retriever.get_retriever`` for the duration of the test.
    """
    retriever = MagicMock()
    retriever.invoke.return_value = FAKE_DOCS
    return retriever


@pytest.fixture()
def empty_retriever():
    """
    Return a MagicMock retriever whose ``.invoke()`` returns an empty list.

    Used to test the no-results / fallback code path.
    """
    retriever = MagicMock()
    retriever.invoke.return_value = []
    return retriever


# ── LLM fixture ───────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_llm_response():
    """
    Patch ``ChatGoogleGenerativeAI`` so no real API call is made.

    The mock LLM's ``.invoke()`` returns a fake response with
    ``.content = "Mocked LLM answer."`` and ``.stream()`` yields a single
    chunk with the same content.
    """
    fake_chunk = MagicMock()
    fake_chunk.content = "Mocked LLM answer."

    fake_response = MagicMock()
    fake_response.content = "Mocked LLM answer."

    mock_cls = MagicMock()
    mock_cls.return_value.invoke.return_value = fake_response
    mock_cls.return_value.stream.return_value = iter([fake_chunk])

    with patch("chatbot.rag_chain._llm") as mock_llm_instance:
        mock_llm_instance.invoke.return_value = fake_response
        mock_llm_instance.stream.return_value = iter([fake_chunk])
        yield mock_llm_instance
