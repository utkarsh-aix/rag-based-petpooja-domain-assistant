"""
app/app.py
----------
Petpooja Knowledge Assistant — Flask Backend

Run from the project root:
    python3 app/app.py
"""

import os
import sys
import json
from flask import Flask, render_template, request, Response, stream_with_context

# ── Project root on sys.path so all absolute imports work ─────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chatbot.rag_chain import stream_question
from utils.helpers import export_chat_to_text

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    """Serve the main chatbot page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat request and stream responses via Server-Sent Events (SSE).
    Accepts JSON body:
      {
        "question": str,
        "chat_history": list[dict]
      }
    """
    data = request.json or {}
    question = data.get('question', '')
    chat_history = data.get('chat_history', [])

    # Format the last 6 messages of chat_history as expected by stream_question
    history_for_llm = []
    for entry in chat_history[-6:]:
        history_for_llm.append({
            "role": entry.get("role"),
            "content": entry.get("content")
        })

    def generate():
        try:
            # Stream the question using stream_question from rag_chain
            for chunk in stream_question(question, history_for_llm):
                if isinstance(chunk, str):
                    # Yield regular tokens
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    # Yield final metadata (answer, sources, retrieved_docs)
                    yield f"data: {json.dumps({'done': True, 'sources': chunk.get('sources', []), 'answer': chunk.get('answer', '')})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/export', methods=['POST'])
def export_chat():
    """
    Export chat history.
    Accepts JSON body:
      {
        "chat_history": list[dict],
        "format": str ("text" or "json")
      }
    """
    data = request.json or {}
    chat_history = data.get('chat_history', [])

    export_content = export_chat_to_text(chat_history)
    filename = "petpooja_chat.txt"
    mimetype = "text/plain"

    return Response(
        export_content,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)