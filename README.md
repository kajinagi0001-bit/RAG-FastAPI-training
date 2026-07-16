# RAG FastAPI Practice

FastAPI, SQLite, and SQLAlchemy practice project for learning the API and DB side of a small RAG application.

The current goals are:

1. Create documents through an API
2. Split document content into chunks
3. Upload Markdown, text, and PDF files as documents
4. Read documents and chunks from SQLite
5. Store an embedding for each chunk
6. Search chunks by vector similarity
7. Generate an answer from retrieved context with an LLM
8. Store conversation messages

## Setup

```powershell
cd work/rag-fastapi-practice
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## OpenAI Embeddings

Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_GENERATION_MODEL=gpt-5.6
```

The app loads `.env` automatically at startup.

By default, the app uses:

```text
text-embedding-3-small
```

You can override it in `.env`:

```text
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

If `OPENAI_API_KEY` is not set, the app falls back to a local hash-based embedding implementation so tests and local API exploration still work without network calls.

To force the local implementation:

```text
EMBEDDING_PROVIDER=local
```

To force OpenAI:

```text
EMBEDDING_PROVIDER=openai
```

LLM answer generation uses the Responses API. It defaults to:

```text
OPENAI_GENERATION_MODEL=gpt-5.6
```

You can force local fallback generation:

```text
GENERATION_PROVIDER=local
```

Or force OpenAI generation:

```text
GENERATION_PROVIDER=openai
```

If your default Python is too new for some packages, use the bundled Python shown by Codex and create `.venv311` instead:

```powershell
C:\Users\nagi0\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run

```powershell
uvicorn app.main:app --reload
```

OpenAPI UI:

```text
http://127.0.0.1:8000/docs
```

## Try It

Create a document:

```powershell
curl -X POST http://127.0.0.1:8000/documents `
  -H "Content-Type: application/json" `
  -d "{\"title\":\"RAG basics\",\"content\":\"RAG retrieves relevant documents from a database and passes them to an LLM as context.\"}"
```

List chunks for a document:

```powershell
curl http://127.0.0.1:8000/documents/1/chunks
```

Upload a Markdown, text, or PDF file:

```powershell
curl -X POST http://127.0.0.1:8000/documents/upload `
  -F "file=@notes.md"
```

```powershell
curl -X POST http://127.0.0.1:8000/documents/upload `
  -F "file=@paper.pdf"
```

Ask a question:

```powershell
curl -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How does RAG retrieve documents?\"}"
```

Create a conversation:

```powershell
curl -X POST http://127.0.0.1:8000/conversations `
  -H "Content-Type: application/json" `
  -d "{\"title\":\"Agent memory\"}"
```

Ask in a conversation:

```powershell
curl -X POST http://127.0.0.1:8000/conversations/1/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"What does conversation memory store?\"}"
```

List conversation messages:

```powershell
curl http://127.0.0.1:8000/conversations/1/messages
```

## API

```text
POST /documents
POST /documents/upload
GET  /documents
GET  /documents/{id}
GET  /documents/{id}/chunks
POST /chat
POST /conversations
GET  /conversations
GET  /conversations/{id}/messages
POST /conversations/{id}/chat
```

## Current Implementation

This version splits each document into chunks when it is saved.

Documents can be created from JSON or uploaded as `.md`, `.txt`, or `.pdf` files.

Each chunk also gets an embedding stored in SQLite as JSON.

`POST /chat` embeds the question, searches chunks by cosine similarity, and sends the retrieved context to an LLM to generate the final answer.

`POST /conversations/{id}/chat` does the same RAG flow and also stores the user and assistant messages.

It also passes the most recent conversation messages to the LLM prompt. The default history limit is:

```text
CONVERSATION_HISTORY_LIMIT=6
```

That keeps the DB/API flow visible:

```text
request -> FastAPI -> optional conversation memory -> question embedding -> vector retriever -> LLM -> response with sources
```

Next steps:

1. Move vector search to pgvector, FAISS, or another vector store
2. Add evaluation cases for answer quality
3. Add retrieval evaluation metrics

## Tests

```powershell
pytest
```
