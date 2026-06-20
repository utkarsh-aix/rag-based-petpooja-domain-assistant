"""
chatbot/rag_chain.py
--------------------
Core RAG pipeline functions for the Petpooja knowledge assistant.

The LLM is instantiated ONCE at module load time and reused across all calls.

Public API
~~~~~~~~~~
    ask_question(question)
        Backward-compatible wrapper. Returns the answer string only.

    ask_question_with_sources(question, chat_history=[])
        Full pipeline. Returns a result dict:
            {
                "answer":         str,
                "sources":        list[str],   # unique source filenames
                "retrieved_docs": int,          # number of chunks retrieved
            }

    stream_question(question, chat_history=[])
        Generator that yields LLM response tokens one at a time for
        true streaming in the UI. Also yields the result dict as the
        final item once streaming completes.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot.prompt import format_chat_history, get_conversational_prompt, get_prompt
from chatbot.retriever import get_retriever
from config.settings import settings

logger = logging.getLogger(__name__)

# ── LLM singleton — created once at import time ───────────────────────────────
_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
)

# Fallback message when the knowledge base has no relevant content
_NO_INFO_MSG = (
    "Sorry, this information is not available in the company knowledge base."
)
# Fallback message when the LLM call itself fails
_LLM_ERROR_MSG = (
    "I'm sorry, I encountered an error while generating a response. "
    "Please try again in a moment."
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_context_and_sources(documents: list) -> tuple[str, list[str]]:
    """
    Concatenate document page content into a single context string and
    collect unique source filenames.

    Args:
        documents: Retrieved :class:`langchain_core.documents.Document` list.

    Returns:
        ``(context_str, sources)`` where *sources* is a de-duplicated list
        of ``source_file`` metadata values.
    """
    context = "\n\n".join(doc.page_content for doc in documents)
    sources: list[str] = []
    for doc in documents:
        src = doc.metadata.get("source_file")
        if src and src not in sources:
            sources.append(src)
    return context, sources


def _build_prompt_str(
    context: str,
    question: str,
    chat_history: list[dict],
) -> str:
    """
    Select the appropriate prompt template based on whether chat history is
    present and return the fully formatted prompt string.
    """
    if chat_history:
        prompt = get_conversational_prompt()
        history_str = format_chat_history(chat_history)
        return prompt.format(
            context=context,
            question=question,
            chat_history=history_str,
        )
    else:
        prompt = get_prompt()
        return prompt.format(context=context, question=question)


# ── Public API ────────────────────────────────────────────────────────────────

def ask_question_with_sources(
    question: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Run the full RAG pipeline and return a structured result dict.

    Retrieves relevant document chunks, selects the appropriate prompt
    (conversational when *chat_history* is provided, single-turn otherwise),
    calls the LLM, and returns both the answer and provenance metadata.

    Args:
        question:     The user's query string.
        chat_history: Optional list of prior turns, each a dict with keys
                      ``"role"`` (``"user"``/``"bot"``) and ``"content"``.
                      Pass ``None`` or ``[]`` for a fresh (single-turn) query.

    Returns:
        A dict with the following keys:

        * ``"answer"``         (str)       – The LLM-generated answer.
        * ``"sources"``        (list[str]) – Unique source filenames used.
        * ``"retrieved_docs"`` (int)       – Number of chunks retrieved.

    Example::

        result = ask_question_with_sources(
            "What is the refund policy?",
            chat_history=st.session_state.chat_history,
        )
        print(result["answer"])
        print(result["sources"])      # ["policy.md"]
        print(result["retrieved_docs"])  # 4
    """
    chat_history = chat_history or []

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retriever = get_retriever()
    documents = retriever.invoke(question)
    retrieved_count = len(documents)

    logger.info(
        "Retrieved %d document chunk(s) for question: %r",
        retrieved_count,
        question[:80],
    )

    if not documents:
        logger.warning("No relevant documents found — returning no-info response")
        return {
            "answer": _NO_INFO_MSG,
            "sources": [],
            "retrieved_docs": 0,
        }

    context, sources = _build_context_and_sources(documents)
    logger.info("Sources used: %s", sources)

    # ── Prompt construction ────────────────────────────────────────────────────
    final_prompt = _build_prompt_str(context, question, chat_history)

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        response = _llm.invoke(final_prompt)
        answer = response.content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {
            "answer": _LLM_ERROR_MSG,
            "sources": [],
            "retrieved_docs": retrieved_count,
        }

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_docs": retrieved_count,
    }


def ask_question(question: str) -> str:
    """
    Backward-compatible single-turn wrapper around
    :func:`ask_question_with_sources`.

    Args:
        question: The user's query string.

    Returns:
        The answer string only (no sources or metadata).

    Example::

        answer = ask_question("What payment methods does Petpooja support?")
    """
    return ask_question_with_sources(question)["answer"]


def stream_question(
    question: str,
    chat_history: list[dict] | None = None,
):
    """
    Generator that streams LLM response tokens one at a time, enabling
    true character-by-character streaming in the UI.

    Retrieval and prompt construction are identical to
    :func:`ask_question_with_sources`.  The generator yields:

    * ``str`` tokens during streaming.
    * A final ``dict`` (same shape as :func:`ask_question_with_sources`)
      as the very last item — so the caller can capture sources and metadata
      after the stream ends.

    Args:
        question:     The user's query string.
        chat_history: Optional list of prior turns (same format as
                      :func:`ask_question_with_sources`).

    Yields:
        ``str`` – individual text tokens from the LLM.
        ``dict`` – result summary as the final yielded value.

    Example (Streamlit)::

        placeholder = st.empty()
        full_text   = ""
        result_meta = {}

        for chunk in stream_question(question, chat_history):
            if isinstance(chunk, str):
                full_text += chunk
                placeholder.markdown(full_text + "▌")
            else:
                result_meta = chunk   # final dict with sources etc.

        placeholder.markdown(full_text)
    """
    chat_history = chat_history or []

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retriever = get_retriever()
    documents = retriever.invoke(question)
    retrieved_count = len(documents)

    logger.info(
        "Stream — retrieved %d chunk(s) for question: %r",
        retrieved_count,
        question[:80],
    )

    if not documents:
        logger.warning("No relevant documents found — streaming no-info response")
        yield _NO_INFO_MSG
        yield {"answer": _NO_INFO_MSG, "sources": [], "retrieved_docs": 0}
        return

    context, sources = _build_context_and_sources(documents)
    logger.info("Stream — sources: %s", sources)

    final_prompt = _build_prompt_str(context, question, chat_history)

    # ── Streaming LLM call ────────────────────────────────────────────────────
    full_answer = ""
    try:
        for chunk in _llm.stream(final_prompt):
            token = chunk.content
            if token:
                full_answer += token
                yield token
    except Exception as exc:
        logger.error("LLM stream failed: %s", exc)
        yield _LLM_ERROR_MSG
        yield {
            "answer": _LLM_ERROR_MSG,
            "sources": [],
            "retrieved_docs": retrieved_count,
        }
        return

    # Final metadata dict — always the last yielded item
    yield {
        "answer": full_answer,
        "sources": sources,
        "retrieved_docs": retrieved_count,
    }
