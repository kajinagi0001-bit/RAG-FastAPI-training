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
GET  /
POST /documents
POST /documents/upload
GET  /documents
GET  /documents/{id}
GET  /documents/{id}/chunks
POST /chat
POST /agent
POST /agent/tool-calling
POST /memories
GET  /memories
POST /memories/search
POST /memories/{id}/feedback
GET  /memories/{id}/feedback
POST /memories/{id}/judge
POST /conversations
GET  /conversations
GET  /conversations/{id}/messages
POST /conversations/{id}/chat
GET  /rag-runs
GET  /rag-runs/{id}
GET  /rag-runs/{id}/tool-calls
POST /tool-calls/{id}/feedback
GET  /tool-calls/{id}/feedback
POST /tool-calls/{id}/judge
POST /rag-runs/{id}/feedback
GET  /rag-runs/{id}/feedback
POST /rag-runs/{id}/judge
GET  /dashboard
```

## Current Implementation

This version splits each document into chunks when it is saved.

Documents can be created from JSON or uploaded as `.md`, `.txt`, or `.pdf` files.

Each chunk also gets an embedding stored in SQLite as JSON.

Open the chat UI:

```text
http://127.0.0.1:8000/
```

The chat UI can call normal RAG, the local agent loop, or the OpenAI tool-calling agent. It also links to the evaluation dashboard.

The same page can upload knowledge files:

```text
.md
.txt
.pdf
```

After upload, the file is saved as a document, split into chunks, embedded, and becomes searchable from chat.

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
  -> search_memories
  -> search_knowledge_base
  -> decide_answer or decide_retry_search
  -> optional retry search
  -> answer_with_context
  -> log_rag_run
```

It returns the final answer, sources, and a `steps` list showing which actions were taken.
If retrieval is weak, the agent increases `top_k` and searches once more before answering.
If relevant long-term memories exist, they are passed into answer generation as memory history.

`POST /agent/tool-calling` passes function tool definitions to the OpenAI Responses API when `GENERATION_PROVIDER=openai`.
The currently exposed function tools are:

```text
search_knowledge_base
get_document
get_document_chunks
create_memory
search_memories
answer_with_context
```

When `GENERATION_PROVIDER=local`, this endpoint falls back to the local `/agent` loop so the project still works without external API calls.

Long-term memory is stored separately from conversation messages:

```text
POST /memories
GET  /memories
POST /memories/search
```

Example:

```powershell
curl -X POST http://127.0.0.1:8000/memories `
  -H "Content-Type: application/json" `
  -d "{\"content\":\"The user prefers implementation-first explanations.\",\"source\":\"user\"}"
```

Search memories:

```powershell
curl -X POST http://127.0.0.1:8000/memories/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"explanation preference\",\"top_k\":3}"
```

Add feedback to a memory:

```powershell
curl -X POST http://127.0.0.1:8000/memories/1/feedback `
  -H "Content-Type: application/json" `
  -d "{\"importance\":5,\"accuracy\":5,\"future_usefulness\":4,\"notes\":\"Useful durable preference.\"}"
```

Memory feedback uses 1 to 5 scores:

```text
importance
accuracy
future_usefulness
```

Run automatic memory judging:

```powershell
curl -X POST http://127.0.0.1:8000/memories/1/judge
```

This scores the memory with the configured generation model, then stores the result as memory feedback. With `GENERATION_PROVIDER=local`, it uses a simple local heuristic instead.

The local `/agent` loop now searches memories before answering. The OpenAI tool-calling agent can also call `search_memories` and `create_memory`.

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

Add feedback to a tool call:

```powershell
curl -X POST http://127.0.0.1:8000/tool-calls/1/feedback `
  -H "Content-Type: application/json" `
  -d "{\"tool_choice_quality\":5,\"argument_quality\":4,\"output_usefulness\":5,\"notes\":\"The search tool was appropriate.\"}"
```

Tool call feedback uses 1 to 5 scores:

```text
tool_choice_quality
argument_quality
output_usefulness
```

Run automatic tool-call judging:

```powershell
curl -X POST http://127.0.0.1:8000/tool-calls/1/judge
```

This scores the tool call with the configured generation model, then stores the result as tool call feedback. With `GENERATION_PROVIDER=local`, it uses a simple local heuristic instead.

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
run_type
embedding_model
generation_model
```

`run_type` records which path created the run:

```text
chat
conversation_rag
agent
tool_calling_agent
unknown
```

`unknown` is used only for older rows that were created before run type tracking existed.

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
2. Add memory cleanup suggestions
3. Add agent run comparison reports

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
feedback_count: 1
avg_tool_choice_quality: 5.00
avg_argument_quality: 4.00
avg_output_usefulness: 5.00
low_quality_feedback_count: 0
tool_counts:
- answer_with_context: 2
- search_knowledge_base: 2
```

## Memory Evaluation

Summarize memory feedback:

```powershell
python -m scripts.eval_memories
```

Example output:

```text
feedback_count: 2
avg_importance: 3.50
avg_accuracy: 4.00
avg_future_usefulness: 3.00
low_quality_count: 1

lowest_rated_memories:
- memory_id=2 importance=2 accuracy=3 future_usefulness=2 content=A vague temporary preference.

cleanup_suggestions:
- memory_id=2 action=delete_candidate avg_importance=2.00 avg_accuracy=3.00 avg_future_usefulness=2.00 reason=At least one average quality dimension is 2 or lower. content=A vague temporary preference.
```

Cleanup suggestion actions:

```text
keep
review
delete_candidate
```

## Agent Run Comparison

Compare saved RAG and agent runs:

```powershell
python -m scripts.eval_agent_runs
```

Example output:

```text
total_runs: 132

run_type_summary:
- run_type=agent run_count=5 feedback_count=2 avg_groundedness=4.50 avg_answer_quality=4.00 avg_source_usefulness=4.50 avg_tool_calls=0.00
- run_type=chat run_count=20 feedback_count=8 avg_groundedness=4.25 avg_answer_quality=4.00 avg_source_usefulness=4.75 avg_tool_calls=0.00
- run_type=conversation_rag run_count=10 feedback_count=1 avg_groundedness=4.00 avg_answer_quality=4.00 avg_source_usefulness=5.00 avg_tool_calls=0.00
- run_type=tool_calling_agent run_count=3 feedback_count=1 avg_groundedness=5.00 avg_answer_quality=5.00 avg_source_usefulness=5.00 avg_tool_calls=2.00

lowest_rated_runs:
- rag_run_id=129 run_type=chat groundedness=4 answer_quality=4 source_usefulness=5 tool_call_count=0 question=How does RAG retrieve documents?
```

Current run type tracking:

```text
chat: /chat
conversation_rag: /conversations/{id}/chat
agent: /agent
tool_calling_agent: /agent/tool-calling with OpenAI tool calls
unknown: old rows before explicit run type tracking
```

The comparison script uses the saved `rag_runs.run_type` value first. It only falls back to feature-based classification for old `unknown` rows.

## Evaluation Dashboard

Open a simple inspection page:

```text
http://127.0.0.1:8000/dashboard
```

The page shows:

```text
run_type summary
recent runs
low-rated answers
recent tool calls
memory cleanup candidates
```

This is intentionally implemented inside FastAPI as a small HTML page so the evaluation data flow stays easy to inspect.
