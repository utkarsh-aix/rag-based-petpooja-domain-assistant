# 🍽️ Petpooja Knowledge Assistant (RAG Chatbot)

A context-grounded RAG chatbot designed for **Petpooja** (restaurant SaaS) documentation. It retrieves relevant knowledge from company docs (.md, .txt, .pdf, .docx) and uses Gemini 2.5 Flash to stream accurate answers.

---

## 📸 Screenshots

| 🖥️ UI Dashboard | 💬 Query Response | ☀️ Light Theme |
| :---: | :---: | :---: |
| ![UI Dashboard](screenshots/dashboard.png) | ![Query Response](screenshots/query_response.png) | ![Light Theme](screenshots/light_theme.png) |

---

## ✨ Features

- **Semantic Retrieval**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and a local FAISS database with similarity/MMR search.
- **Strict Grounding**: Only answers based on the provided company documents to avoid hallucinations.
- **Real-time Streaming**: Server-Sent Events (SSE) for token-by-token streaming in the UI.
- **Chat History**: Maintains conversation memory (last 6 turns) for contextual follow-up questions.
- **Export History**: Download full chat transcripts as plain text.

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Add Documents & Index
Place your `.md`, `.txt`, `.pdf`, or `.docx` files under the `data/` folder, then build the vector index:
```bash
python3 -m ingest.run_ingest
```

### 4. Run the Application
Launch the Flask development server:
```bash
python3 app/app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 📂 Project Structure

- `app/` — Flask backend, single-page HTML template, and CSS/JS static files.
- `chatbot/` — Prompt templates, retriever settings, and RAG execution chain.
- `config/` — Environment variable loading.
- `data/` — Knowledge base documents organized by subfolder categories.
- `ingest/` — Parsing, text-splitting, and FAISS vector index builder.
- `utils/` — Query cleaning and file export helpers.

---

## 🛠️ Tech Stack

- **RAG & LLM**: LangChain, Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **Embeddings & Vector Store**: HuggingFace SentenceTransformers, FAISS
- **Backend & Frontend**: Flask, Vanilla CSS/JS
- **Document Loaders**: PyPDF, python-docx
