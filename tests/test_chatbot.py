"""
tests/test_chatbot.py
---------------------
Pytest suite for the Petpooja RAG chatbot.

All tests are fully offline — no FAISS index or Google Gemini API required.
The real retriever and LLM are replaced by fixtures defined in conftest.py.

Run from the project root:
    pytest tests/ -v
"""

from unittest.mock import patch

import pytest

# ── Module under test ─────────────────────────────────────────────────────────
from chatbot.prompt import get_prompt
from chatbot.rag_chain import ask_question, ask_question_with_sources, stream_question
from utils.helpers import clean_query, format_sources


# ═════════════════════════════════════════════════════════════════════════════
# 1. ask_question()  — backward-compatible wrapper
# ═════════════════════════════════════════════════════════════════════════════

class TestAskQuestion:
    """Tests for the simple ask_question() wrapper."""

    def test_ask_question_returns_string(self, mock_retriever, mock_llm_response):
        """Return value must be a non-empty string."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question("What is Petpooja?")

        assert isinstance(result, str), "ask_question must return a str"
        assert len(result) > 0, "Returned answer must not be empty"

    def test_ask_question_content(self, mock_retriever, mock_llm_response):
        """The returned string should be the mocked LLM answer."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question("What is Petpooja?")

        assert result == "Mocked LLM answer."


# ═════════════════════════════════════════════════════════════════════════════
# 2. ask_question_with_sources()  — structured result dict
# ═════════════════════════════════════════════════════════════════════════════

class TestAskQuestionWithSources:
    """Tests for ask_question_with_sources()."""

    def test_returns_dict_with_required_keys(self, mock_retriever, mock_llm_response):
        """Result must be a dict with 'answer', 'sources', 'retrieved_docs'."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question_with_sources("What is Petpooja?")

        assert isinstance(result, dict), "Result must be a dict"
        assert "answer" in result, "Result must have 'answer' key"
        assert "sources" in result, "Result must have 'sources' key"
        assert "retrieved_docs" in result, "Result must have 'retrieved_docs' key"

    def test_answer_is_string(self, mock_retriever, mock_llm_response):
        """'answer' value must be a non-empty string."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question_with_sources("Tell me about the POS")

        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sources_is_list(self, mock_retriever, mock_llm_response):
        """'sources' value must be a list."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question_with_sources("Tell me about the POS")

        assert isinstance(result["sources"], list)

    def test_sources_populated_from_docs(self, mock_retriever, mock_llm_response):
        """Sources should contain the unique source_file values from retrieved docs."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question_with_sources("Tell me about the POS")

        # FAKE_DOCS have "about.md" and "product_overview.md"
        assert "about.md" in result["sources"]
        assert "product_overview.md" in result["sources"]

    def test_retrieved_docs_count(self, mock_retriever, mock_llm_response):
        """'retrieved_docs' should equal the number of docs returned by the retriever."""
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question_with_sources("Tell me about the POS")

        assert result["retrieved_docs"] == 2   # FAKE_DOCS has 2 entries

    def test_with_chat_history(self, mock_retriever, mock_llm_response):
        """Function should accept a non-empty chat_history without raising."""
        history = [
            {"role": "user",  "content": "What is Petpooja?"},
            {"role": "bot",   "content": "Petpooja is a POS platform."},
        ]
        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            result = ask_question_with_sources("Does it support KOT?", chat_history=history)

        assert isinstance(result["answer"], str)


# ═════════════════════════════════════════════════════════════════════════════
# 3. No-results fallback
# ═════════════════════════════════════════════════════════════════════════════

class TestFallbackBehaviour:
    """Tests for the case where the retriever finds no relevant documents."""

    def test_unknown_question_fallback_message(self, empty_retriever, mock_llm_response):
        """When no docs are retrieved, answer must contain the standard fallback."""
        with patch("chatbot.rag_chain.get_retriever", return_value=empty_retriever):
            result = ask_question_with_sources("xyzzy unknown question 12345")

        assert "not available" in result["answer"].lower(), (
            "Fallback message should mention information not being available"
        )

    def test_unknown_question_empty_sources(self, empty_retriever, mock_llm_response):
        """When no docs are retrieved, sources list must be empty."""
        with patch("chatbot.rag_chain.get_retriever", return_value=empty_retriever):
            result = ask_question_with_sources("xyzzy unknown question 12345")

        assert result["sources"] == []

    def test_unknown_question_zero_retrieved_docs(self, empty_retriever, mock_llm_response):
        """When no docs are retrieved, retrieved_docs must be 0."""
        with patch("chatbot.rag_chain.get_retriever", return_value=empty_retriever):
            result = ask_question_with_sources("xyzzy unknown question 12345")

        assert result["retrieved_docs"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# 4. stream_question()  — generator
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamQuestion:
    """Tests for the streaming generator."""

    def test_yields_string_tokens(self, mock_retriever, mock_llm_response):
        """Stream should yield at least one string token before the final dict."""
        tokens = []
        final_meta = {}

        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            for chunk in stream_question("What is Petpooja?"):
                if isinstance(chunk, str):
                    tokens.append(chunk)
                else:
                    final_meta = chunk

        assert len(tokens) >= 1, "Stream must yield at least one string token"

    def test_final_item_is_dict(self, mock_retriever, mock_llm_response):
        """Last item yielded by the generator must be the metadata dict."""
        last_item = None

        with patch("chatbot.rag_chain.get_retriever", return_value=mock_retriever):
            for chunk in stream_question("What is Petpooja?"):
                last_item = chunk

        assert isinstance(last_item, dict)
        assert "answer" in last_item
        assert "sources" in last_item

    def test_stream_fallback_on_empty_retriever(self, empty_retriever, mock_llm_response):
        """When retriever returns nothing, stream must still yield a fallback string."""
        items = []

        with patch("chatbot.rag_chain.get_retriever", return_value=empty_retriever):
            for chunk in stream_question("unknown"):
                items.append(chunk)

        str_items = [i for i in items if isinstance(i, str)]
        assert len(str_items) >= 1
        assert "not available" in str_items[0].lower()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Prompt template unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPromptTemplate:
    """Unit tests for chatbot.prompt — no LLM or retriever needed."""

    def test_prompt_includes_context(self):
        """Formatted prompt must contain the context string."""
        prompt = get_prompt()
        filled = prompt.format(context="Some context text", question="What is X?")
        assert "Some context text" in filled

    def test_prompt_includes_question(self):
        """Formatted prompt must contain the question string."""
        prompt = get_prompt()
        filled = prompt.format(context="ctx", question="My test question?")
        assert "My test question?" in filled

    def test_prompt_has_both_variables(self):
        """Both {context} and {question} must appear in the formatted output."""
        prompt = get_prompt()
        filled = prompt.format(context="CTX", question="QST")
        assert "CTX" in filled
        assert "QST" in filled

    def test_prompt_input_variables(self):
        """PromptTemplate must declare exactly context and question as input vars."""
        prompt = get_prompt()
        assert set(prompt.input_variables) == {"context", "question"}


# ═════════════════════════════════════════════════════════════════════════════
# 6. utils.helpers — clean_query
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanQuery:
    """Unit tests for utils.helpers.clean_query."""

    def test_strips_leading_trailing_whitespace(self):
        assert clean_query("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert clean_query("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert clean_query("hello\t\nworld") == "hello world"

    def test_removes_curly_braces(self):
        result = clean_query("What is {pricing}?")
        assert "{" not in result
        assert "}" not in result

    def test_removes_backticks(self):
        result = clean_query("What is `this`?")
        assert "`" not in result

    def test_empty_string_returns_empty(self):
        assert clean_query("") == ""

    def test_clean_query_preserves_alphanumeric(self):
        result = clean_query("What is Petpooja POS v2?")
        assert "Petpooja" in result
        assert "POS" in result


# ═════════════════════════════════════════════════════════════════════════════
# 7. utils.helpers — format_sources
# ═════════════════════════════════════════════════════════════════════════════

class TestFormatSources:
    """Unit tests for utils.helpers.format_sources."""

    def test_empty_list_returns_empty_string(self):
        assert format_sources([]) == ""

    def test_single_source(self):
        result = format_sources(["faqs.md"])
        assert result == "📄 faqs.md"

    def test_multiple_sources_joined_with_dot(self):
        result = format_sources(["faqs.md", "pricing.md"])
        assert result == "📄 faqs.md · 📄 pricing.md"

    def test_each_source_has_emoji(self):
        result = format_sources(["a.md", "b.md", "c.md"])
        assert result.count("📄") == 3

    def test_separator_between_sources(self):
        result = format_sources(["a.md", "b.md"])
        assert "·" in result
