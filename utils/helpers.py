"""
utils/helpers.py
----------------
Reusable utility functions shared across the RAG chatbot project.

Functions:
    format_sources       – Format a list of source filenames into markdown.
    truncate_text        – Truncate text to a character limit without mid-word cuts.
    clean_query          – Sanitise a user query before sending to the LLM.
    export_chat_to_text  – Export chat history as a human-readable text string.
    get_project_root     – Return the absolute path to the project root directory.
"""

import re
from datetime import datetime
from pathlib import Path


# ── Path helpers ─────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Resolves the root as two levels above this file
    (utils/helpers.py → utils/ → project_root/).
    Useful for building absolute paths to artefacts like
    ``vector_store/`` or ``data/`` regardless of the working directory.

    Returns:
        Path: Absolute path to the project root.

    Example::

        root = get_project_root()
        vector_store = root / "vector_store" / "faiss_index"
    """
    return Path(__file__).resolve().parent.parent


# ── Source formatting ─────────────────────────────────────────────────────────

def format_sources(sources: list[str]) -> str:
    """
    Format a list of source filenames into a readable markdown string.

    Each filename is prefixed with a document emoji and entries are joined
    with a middle-dot separator.

    Args:
        sources: List of source filenames, e.g. ``["faqs.md", "pricing.md"]``.

    Returns:
        Formatted string like ``"📄 faqs.md · 📄 pricing.md"``,
        or an empty string if *sources* is empty.

    Example::

        >>> format_sources(["faqs.md", "pricing.md"])
        '📄 faqs.md · 📄 pricing.md'
        >>> format_sources([])
        ''
    """
    if not sources:
        return ""
    return " · ".join(f"📄 {s}" for s in sources)


# ── Text utilities ────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 200) -> str:
    """
    Truncate *text* to at most *max_chars* characters without cutting mid-word.

    If the text fits within the limit it is returned unchanged.
    Otherwise the last complete word that still fits is kept and ``"..."``
    is appended.

    Args:
        text:      The input string to truncate.
        max_chars: Maximum allowed character count (default: 200).

    Returns:
        The (possibly truncated) string.

    Example::

        >>> truncate_text("Hello world, this is a test.", max_chars=15)
        'Hello world,...'
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Step back to the last whitespace to avoid a mid-word cut
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "..."


def clean_query(query: str) -> str:
    """
    Sanitise a user query before it is passed to the LLM prompt.

    Steps applied:
    1. Strip leading/trailing whitespace.
    2. Collapse runs of whitespace (spaces, tabs, newlines) to a single space.
    3. Remove characters that can break prompt formatting:
       curly braces ``{}`` (template placeholders), backticks, and null bytes.

    Args:
        query: Raw user input string.

    Returns:
        Cleaned query string.

    Example::

        >>> clean_query("  What  is  {pricing}? \\n\\n")
        'What is pricing?'
    """
    # Strip edges
    query = query.strip()
    # Collapse internal whitespace
    query = re.sub(r"\s+", " ", query)
    # Remove characters that interfere with prompt templates or injection
    query = re.sub(r"[{}`\x00]", "", query)
    return query


# ── Chat export ───────────────────────────────────────────────────────────────

def export_chat_to_text(chat_history: list[dict]) -> str:
    """
    Export a chat history list to a human-readable plain-text string.

    Each entry in *chat_history* is a dict with the following keys:

    * ``"role"``     – ``"user"`` or ``"bot"`` (required)
    * ``"content"``  – Message text (required)
    * ``"sources"``  – List of source filenames (optional)
    * ``"timestamp"``– ISO-format datetime string (optional)

    The output groups messages into ``[User] / [Assistant]`` blocks,
    appends source filenames when present, and separates turns with ``---``.

    Args:
        chat_history: List of message dicts as described above.

    Returns:
        Formatted multi-line string, or an empty string if the history is empty.

    Example output::

        [User]  (2024-01-01 10:00)
        What is the refund policy?

        [Assistant]  (2024-01-01 10:00)
        You can request a refund within 30 days.
        Sources: 📄 policy.md
        ---
    """
    if not chat_history:
        return ""

    lines: list[str] = []

    for entry in chat_history:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        sources = entry.get("sources", [])
        timestamp = entry.get("timestamp", "")

        # Build header
        if role == "user":
            header = "[User]"
        elif role == "bot":
            header = "[Assistant]"
        else:
            header = f"[{role.capitalize()}]"

        # Append timestamp if available
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                header += f"  ({dt.strftime('%Y-%m-%d %H:%M')})"
            except ValueError:
                header += f"  ({timestamp})"

        lines.append(header)
        lines.append(content)

        if sources:
            lines.append(f"Sources: {format_sources(sources)}")

        lines.append("---")
        lines.append("")  # blank line between turns

    # Remove trailing blank line
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)

