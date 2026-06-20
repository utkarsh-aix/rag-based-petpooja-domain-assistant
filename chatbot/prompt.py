"""
chatbot/prompt.py
-----------------
Prompt templates for the Petpooja RAG chatbot.

Functions:
    get_prompt()                  – Standard single-turn RAG prompt.
    get_conversational_prompt()   – Multi-turn prompt that uses chat history
                                    to resolve follow-up questions.
    format_chat_history(messages) – Format the last 6 messages for injection
                                    into the conversational prompt.
"""

from langchain_core.prompts import PromptTemplate


# ── Standard single-turn prompt ───────────────────────────────────────────────

_RAG_TEMPLATE = """\
You are Petpooja's friendly and knowledgeable support assistant. \
Petpooja is a leading restaurant management platform, and your role is to help \
users find accurate, helpful answers from the company's knowledge base.

RESPONSE GUIDELINES:
- Be warm, conversational, and professional — never robotic or terse.
- Format your answers using markdown:
  - Use **bold** for key terms, feature names, and important values.
  - Use bullet points (- item) when listing multiple items or steps.
  - Use short paragraphs for explanatory answers.
- Length: 2-4 sentences for simple questions; structured bullet/section \
response for complex ones.
- Grounding rules:
  - Answer ONLY using information present in the Context below.
  - If the context is fully relevant, answer directly and confidently.
  - If the context is only partially relevant, use what is available and \
begin your answer with: "Based on available information, ..."
  - If the context contains NO relevant information, respond with exactly: \
"Sorry, this information is not available in the company knowledge base."
  - Never invent, guess, or extrapolate facts not present in the context.

---

Context:
{context}

---

Question:
{question}

Answer:
"""

# ── Conversational multi-turn prompt ─────────────────────────────────────────

_CONVERSATIONAL_TEMPLATE = """\
You are Petpooja's friendly and knowledgeable support assistant. \
Petpooja is a leading restaurant management platform, and your role is to help \
users find accurate, helpful answers from the company's knowledge base.

CONVERSATION HISTORY (last few exchanges):
{chat_history}

RESPONSE GUIDELINES:
- Be warm, conversational, and professional — never robotic or terse.
- Format your answers using markdown:
  - Use **bold** for key terms, feature names, and important values.
  - Use bullet points (- item) when listing multiple items or steps.
  - Use short paragraphs for explanatory answers.
- Length: 2-4 sentences for simple questions; structured bullet/section \
response for complex ones.
- Follow-up awareness: The user may use pronouns like "it", "that", "those", \
or refer to something mentioned earlier in the conversation. Use the \
Conversation History above to resolve what they are referring to before answering.
- Grounding rules:
  - Answer ONLY using information present in the Context below.
  - If the context is fully relevant, answer directly and confidently.
  - If the context is only partially relevant, use what is available and \
begin your answer with: "Based on available information, ..."
  - If the context contains NO relevant information, respond with exactly: \
"Sorry, this information is not available in the company knowledge base."
  - Never invent, guess, or extrapolate facts not present in the context.

---

Context:
{context}

---

Question:
{question}

Answer:
"""


# ── Helper ────────────────────────────────────────────────────────────────────

def format_chat_history(messages: list[dict], max_messages: int = 6) -> str:
    """
    Format the tail of a chat history list into a plain-text string for
    injection into the conversational prompt.

    Only the last *max_messages* entries are included to keep the prompt
    within a reasonable token budget.  Each entry is rendered on its own
    line as ``User: ...`` or ``Assistant: ...``.

    Args:
        messages:     List of dicts with ``"role"`` (``"user"``/``"bot"``)
                      and ``"content"`` keys.
        max_messages: Maximum number of recent messages to include
                      (default: 6).

    Returns:
        A formatted string, or ``"(no prior conversation)"`` when the list
        is empty.

    Example::

        history = [
            {"role": "user",    "content": "What is Petpooja?"},
            {"role": "bot",     "content": "Petpooja is a restaurant POS..."},
            {"role": "user",    "content": "Does it support online orders?"},
        ]
        print(format_chat_history(history))
        # User: What is Petpooja?
        # Assistant: Petpooja is a restaurant POS...
        # User: Does it support online orders?
    """
    if not messages:
        return "(no prior conversation)"

    recent = messages[-max_messages:]
    lines: list[str] = []

    for entry in recent:
        role = entry.get("role", "unknown")
        content = entry.get("content", "").strip()
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


# ── Public factory functions ───────────────────────────────────────────────────

def get_prompt() -> PromptTemplate:
    """
    Return the standard single-turn RAG prompt template.

    Input variables: ``context``, ``question``.

    The prompt instructs the model to:
    - Adopt the Petpooja support assistant persona.
    - Answer strictly from the provided context.
    - Use markdown formatting (bold, bullets).
    - Handle partial context gracefully with a "Based on available
      information, ..." prefix.

    Returns:
        :class:`langchain_core.prompts.PromptTemplate`

    Example::

        prompt = get_prompt()
        filled = prompt.format(context="...", question="What is Petpooja?")
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=_RAG_TEMPLATE,
    )


def get_conversational_prompt() -> PromptTemplate:
    """
    Return a multi-turn RAG prompt template that incorporates chat history.

    Input variables: ``context``, ``question``, ``chat_history``.

    Use :func:`format_chat_history` to prepare the ``chat_history`` string
    before calling ``prompt.format(...)``.

    The prompt instructs the model to:
    - Resolve pronouns and references using the conversation history.
    - Answer strictly from the provided context.
    - Use markdown formatting (bold, bullets).
    - Handle partial context gracefully.

    Returns:
        :class:`langchain_core.prompts.PromptTemplate`

    Example::

        from chatbot.prompt import get_conversational_prompt, format_chat_history

        prompt   = get_conversational_prompt()
        history  = format_chat_history(st.session_state.chat_history)
        filled   = prompt.format(
            context=context,
            question=question,
            chat_history=history,
        )
    """
    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=_CONVERSATIONAL_TEMPLATE,
    )
