# RAG FastAPI Practice

FastAPI, SQLite, and SQLAlchemy practice project for learning the API and DB side of a small RAG application.

The current goals are:

1. Create documents through an API
2. Split document content into chunks
3. Read documents and chunks from SQLite
4. Store an embedding for each chunk
5. Search chunks by vector similarity
6. Generate an answer from retrieved context with an LLM

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

Ask a question:

```powershell
curl -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How does RAG retrieve documents?\"}"
```

## API

```text
POST /documents
GET  /documents
GET  /documents/{id}
GET  /documents/{id}/chunks
POST /chat
```

## Current Implementation

This version splits each document into chunks when it is saved.

Each chunk also gets an embedding stored in SQLite as JSON.

`POST /chat` embeds the question, searches chunks by cosine similarity, and sends the retrieved context to an LLM to generate the final answer.

That keeps the DB/API flow visible:

```text
request -> FastAPI -> question embedding -> vector retriever -> LLM -> response with sources
```

Next steps:

1. Move vector search to pgvector, FAISS, or another vector store
2. Add conversation history
3. Add evaluation cases for answer quality

## Tests

```powershell
pytest
```
