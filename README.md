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

Ask through the agent loop:

```powershell
curl -X POST http://127.0.0.1:8000/agent `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How does RAG retrieve documents?\"}"
```

Ask through the OpenAI tool-calling agent:

```powershell
curl -X POST http://127.0.0.1:8000/agent/tool-calling `
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
POST /agent
POST /agent/tool-calling
POST /conversations
GET  /conversations
GET  /conversations/{id}/messages
POST /conversations/{id}/chat
GET  /rag-runs
GET  /rag-runs/{id}
GET  /rag-runs/{id}/tool-calls
POST /rag-runs/{id}/feedback
GET  /rag-runs/{id}/feedback
POST /rag-runs/{id}/judge
```

## Current Implementation

This version splits each document into chunks when it is saved.

Documents can be created from JSON or uploaded as `.md`, `.txt`, or `.pdf` files.

Each chunk also gets an embedding stored in SQLite as JSON.

`POST /chat` embeds the question, searches chunks by cosine similarity, and sends the retrieved context to an LLM to generate the final answer.

`POST /conversations/{id}/chat` does the same RAG flow and also stores the user and assistant messages.

The core RAG operations are separated into internal tool functions in `app/tools.py`:

```text
create_document_with_chunks
search_knowledge_base
get_document
get_document_chunks
answer_with_context
run_rag_chat
create_feedback
```

FastAPI endpoints call these functions, and future agent loops can call the same functions directly.

`POST /agent` runs a first agent loop:

```text
plan
  -> search_knowledge_base
  -> decide_answer or decide_retry_search
  -> optional retry search
  -> answer_with_context
  -> log_rag_run
```

It returns the final answer, sources, and a `steps` list showing which actions were taken.
If retrieval is weak, the agent increases `top_k` and searches once more before answering.

`POST /agent/tool-calling` passes function tool definitions to the OpenAI Responses API when `GENERATION_PROVIDER=openai`.
The currently exposed function tools are:

```text
search_knowledge_base
get_document
get_document_chunks
answer_with_context
```

When `GENERATION_PROVIDER=local`, this endpoint falls back to the local `/agent` loop so the project still works without external API calls.

OpenAI tool-calling runs are traced in `agent_tool_calls`:

```text
rag_run_id
step
tool_name
arguments_json
output_json
```

Review tool calls for a RAG run:

```powershell
curl http://127.0.0.1:8000/rag-runs/1/tool-calls
```

It also passes the most recent conversation messages to the LLM prompt. The default history limit is:

```text
CONVERSATION_HISTORY_LIMIT=6
```

That keeps the DB/API flow visible:

```text
request -> FastAPI -> optional conversation memory -> question embedding -> vector retriever -> LLM -> response with sources
```

Each RAG run is logged for answer quality review:

```text
question
answer
retrieved_sources_json
embedding_model
generation_model
```

Review runs:

```powershell
curl http://127.0.0.1:8000/rag-runs
curl http://127.0.0.1:8000/rag-runs/1
```

Add manual feedback:

```powershell
curl -X POST http://127.0.0.1:8000/rag-runs/1/feedback `
  -H "Content-Type: application/json" `
  -d "{\"groundedness\":5,\"answer_quality\":4,\"source_usefulness\":5,\"notes\":\"The answer is grounded in the retrieved source.\"}"
```

Review feedback:

```powershell
curl http://127.0.0.1:8000/rag-runs/1/feedback
```

Run LLM-as-a-judge for a saved RAG run:

```powershell
curl -X POST http://127.0.0.1:8000/rag-runs/1/judge
```

This asks the configured generation model to score the saved answer, then stores the result as feedback.

Manual feedback uses 1 to 5 scores:

```text
groundedness
answer_quality
source_usefulness
```

Next steps:

1. Move vector search to pgvector, FAISS, or another vector store
2. Add long-term memory tools
3. Add tool-call quality labels

## Tests

```powershell
pytest
```

## Retrieval Evaluation

Retrieval eval cases live in:

```text
evals/retrieval_cases.json
```

Run:

```powershell
python -m scripts.eval_retrieval
```

Example output:

```text
total: 2
hit@k: 0.50
mrr: 0.50
```

Each case can use:

```json
{
  "question": "What does RAG retrieve?",
  "expected_text": "relevant chunks",
  "expected_document_id": 1,
  "top_k": 3
}
```

## Answer Quality Evaluation

After adding manual feedback through `/rag-runs/{id}/feedback`, summarize answer quality:

```powershell
python -m scripts.eval_answers
```

Example output:

```text
total_feedback: 3
avg_groundedness: 4.33
avg_answer_quality: 4.00
avg_source_usefulness: 3.67
low_quality_count: 1

lowest_rated:
- rag_run_id=2 groundedness=2 answer_quality=3 source_usefulness=2 question=What is memory?
```

## Tool Call Evaluation

Summarize OpenAI tool-calling traces:

```powershell
python -m scripts.eval_tool_calls
```

Example output:

```text
total_tool_calls: 4
runs_with_tool_calls: 2
avg_tool_calls_per_run: 2.00
tool_counts:
- answer_with_context: 2
- search_knowledge_base: 2
```
