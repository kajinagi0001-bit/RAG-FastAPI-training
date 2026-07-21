# AI Agent Learning Plan

This project is moving from a basic RAG API toward a structured AI agent learning project.

The goal is not production hardening first. The goal is to learn the core building blocks of AI agents by implementing them one by one.

## Current Baseline

The project can already:

1. Create documents through FastAPI
2. Store documents in SQLite
3. Split document content into chunks
4. Create OpenAI embeddings for chunks
5. Store chunk embeddings in SQLite as JSON
6. Embed user questions
7. Search chunks by cosine similarity
8. Generate answers with an LLM using retrieved context
9. Return both `answer` and `sources`

Current RAG flow:

```text
document -> chunk -> embed -> store
question -> embed -> retrieve chunks -> generate answer -> return sources
```

## Learning Roadmap

The roadmap is organized around the major parts of an AI agent system:

```text
1. Knowledge Ingestion
2. Memory
3. Retrieval
4. Generation
5. Evaluation
6. Agent Loop
7. Tool Use
```

## Phase 1: Knowledge Ingestion

Purpose:

Build the pipeline that lets the agent receive external knowledge.

Target API:

```text
POST /documents/upload
```

Start with:

```text
.md
.txt
```

Then add:

```text
.pdf
```

Implementation tasks:

1. Accept files with FastAPI `UploadFile`
2. Detect file type
3. Read Markdown and text files as UTF-8
4. Extract text from PDFs
5. Normalize extracted text
6. Save as `Document`
7. Split into chunks
8. Generate embeddings
9. Save chunks and embeddings

Pipeline shape:

```text
file -> extract -> normalize -> chunk -> embed -> store
```

Concepts learned:

```text
Knowledge ingestion
Document pipelines
Unstructured data processing
RAG preprocessing
```

Recommended first implementation:

```text
POST /documents/upload
Supported: .md, .txt
Title: filename
Content: file text
```

Add PDF support after the text pipeline works.

## Phase 2: Conversation Memory

Purpose:

Move from one-off RAG calls to a conversational agent with memory.

Tables:

```text
conversations
- id
- title
- created_at

messages
- id
- conversation_id
- role
- content
- created_at
```

Target APIs:

```text
POST /conversations
GET  /conversations
GET  /conversations/{id}/messages
POST /conversations/{id}/chat
```

Flow:

```text
user question
  -> save user message
  -> retrieve relevant chunks
  -> generate answer
  -> save assistant message
  -> return answer and sources
```

Concepts learned:

```text
Short-term memory
Conversation state
Chat API design
Prompting with history
```

Implementation note:

First, save conversation history only. After that works, include the most recent N messages in the LLM prompt.

## Phase 3: Retrieval Evaluation

Purpose:

Measure whether retrieval is improving instead of relying on intuition.

Start with a JSON or CSV eval file.

Example eval case:

```json
{
  "question": "What does RAG retrieve?",
  "expected_document_id": 1,
  "expected_text": "RAG retrieves relevant documents"
}
```

Evaluation script:

```text
python scripts/eval_retrieval.py
```

Metrics:

```text
Recall@k
Hit@k
MRR
```

Example output:

```text
total: 10
hit@1: 0.60
hit@3: 0.80
mrr: 0.72
```

Concepts learned:

```text
Retrieval evaluation
Measuring search quality
Comparing chunk sizes
Comparing top_k values
Comparing embedding models
```

## Phase 4: Answer Quality Evaluation

Purpose:

Track whether the final answer is grounded in retrieved sources.

Add a run log table:

```text
rag_runs
- id
- question
- answer
- retrieved_sources_json
- run_type
- embedding_model
- generation_model
- created_at
```

Review APIs:

```text
GET /rag-runs
GET /rag-runs/{id}
```

Manual feedback table:

```text
rag_run_feedback
- id
- rag_run_id
- groundedness
- answer_quality
- source_usefulness
- notes
- created_at
```

Manual feedback APIs:

```text
POST /rag-runs/{id}/feedback
GET /rag-runs/{id}/feedback
POST /rag-runs/{id}/judge
```

Evaluation dimensions:

```text
groundedness
faithfulness
answer_quality
source_usefulness
```

Start with manual review. Then add LLM-as-a-judge.

LLM-as-a-judge flow:

```text
rag_run
  -> question + answer + retrieved_sources_json
  -> judge model
  -> groundedness / answer_quality / source_usefulness / notes
  -> save as rag_run_feedback
```

Answer summary script:

```text
python -m scripts.eval_answers
```

Summary metrics:

```text
total_feedback
avg_groundedness
avg_answer_quality
avg_source_usefulness
low_quality_count
lowest_rated
```

Concepts learned:

```text
RAG evaluation
Hallucination detection
Observability
Agent improvement loops
```

## Phase 5: Agent Loop

Purpose:

Move from a RAG API to an agent that can choose actions.

Agent loop:

```text
user goal
  -> plan
  -> choose tool
  -> use tool
  -> observe result
  -> answer or continue
```

Current first implementation:

```text
POST /agent

question
  -> plan
  -> search_memories
  -> search_knowledge_base
  -> decide_answer or decide_retry_search
  -> optional retry search with larger top_k
  -> answer_with_context
  -> log_rag_run
  -> return answer + sources + steps
```

Current conditional decision:

```text
if no results or best score < 0.25:
  retry search once with a larger top_k
else:
  answer with current context
```

Memory-aware agent behavior:

```text
/agent
  -> search long-term memories
  -> append matching memories to answer history
  -> search knowledge base
  -> answer with retrieved document context plus memory context
```

OpenAI tool-calling entrypoint:

```text
POST /agent/tool-calling

available tools:
- search_knowledge_base
- get_document
- get_document_chunks
- create_memory
- search_memories
- answer_with_context
```

When OpenAI generation is enabled, the Responses API receives the tool definitions and chooses function calls.
When local generation is enabled, this endpoint falls back to the local agent loop.

Richer tool-calling behavior:

```text
search_knowledge_base
  -> can identify promising document ids
get_document
  -> can inspect document title and full content
get_document_chunks
  -> can load all chunks for a document into latest_results
answer_with_context
  -> can answer using the latest retrieved or loaded context
```

Tool-call tracing:

```text
agent_tool_calls
- id
- rag_run_id
- step
- tool_name
- arguments_json
- output_json
- created_at
```

Tool-call feedback:

```text
agent_tool_call_feedback
- id
- tool_call_id
- tool_choice_quality
- argument_quality
- output_usefulness
- notes
- created_at
```

Trace review API:

```text
GET /rag-runs/{id}/tool-calls
POST /tool-calls/{id}/feedback
GET /tool-calls/{id}/feedback
POST /tool-calls/{id}/judge
```

Tool-call evaluation script:

```text
python -m scripts.eval_tool_calls
```

Summary metrics:

```text
total_tool_calls
runs_with_tool_calls
avg_tool_calls_per_run
tool_counts
feedback_count
avg_tool_choice_quality
avg_argument_quality
avg_output_usefulness
low_quality_feedback_count
```

Automatic tool-call judging:

```text
agent_tool_call
  -> rag_run.question + tool_name + arguments_json + output_json
  -> judge model
  -> tool_choice_quality / argument_quality / output_usefulness / notes
  -> save as agent_tool_call_feedback
```

Long-term memory:

```text
memories
- id
- content
- source
- embedding_json
- embedding_model
- created_at
```

Memory APIs:

```text
POST /memories
GET /memories
POST /memories/search
POST /memories/{id}/feedback
GET /memories/{id}/feedback
POST /memories/{id}/judge
```

Memory tools:

```text
create_memory(content, source)
search_memories(query, top_k)
```

Memory quality feedback:

```text
memory_feedback
- id
- memory_id
- importance
- accuracy
- future_usefulness
- notes
- created_at
```

Memory evaluation script:

```text
python -m scripts.eval_memories
```

Summary metrics:

```text
feedback_count
avg_importance
avg_accuracy
avg_future_usefulness
low_quality_count
lowest_rated_memories
cleanup_suggestions
```

Memory cleanup suggestion actions:

```text
keep
review
delete_candidate
```

Agent run comparison reports:

```text
python -m scripts.eval_agent_runs
```

Compared run types:

```text
tool_calling_agent
conversation_rag
chat_or_local_agent
```

Summary metrics:

```text
run_count
feedback_count
avg_groundedness
avg_answer_quality
avg_source_usefulness
avg_tool_calls
lowest_rated_runs
```

Automatic memory judging:

```text
memory
  -> judge model
  -> importance / accuracy / future_usefulness / notes
  -> save as memory_feedback
```

Initial internal tools:

```text
create_document_with_chunks(title, content)
search_knowledge_base(question)
get_document(document_id)
get_document_chunks(document_id)
answer_with_context(question, sources)
run_rag_chat(question)
create_feedback(rag_run_id, scores)
```

These are implemented as Python functions in `app/tools.py`. Later, expose selected tools through OpenAI tool calling.

Concepts learned:

```text
Tool use
Planning
Observation
Agent orchestration
```

## Phase 6: Tool Use

Purpose:

Let the LLM call structured tools instead of following one fixed pipeline.

Possible tools:

```text
search_knowledge_base
get_document
get_document_chunks
summarize_document
create_eval_case
```

Learning goal:

Understand how an agent decides what action to take and how tool outputs become context for the next step.

## Recommended Implementation Order

1. Markdown and text upload - done
2. PDF upload - done
3. Conversation and message tables
4. `/conversations/{id}/chat`
5. `rag_runs` logging
6. Retrieval eval script
7. Answer eval script
8. Tool function separation
9. Agent loop
10. OpenAI tool calling

Progress:

```text
Markdown/text upload: done
PDF upload: done
Conversation and message tables: done
/conversations/{id}/chat: done
Conversation history in prompt: done
Retrieval evaluation: done
Answer quality run logging: done
Manual answer quality labels: done
Answer evaluation summary script: done
LLM-as-a-judge answer evaluation: done
Tool function separation: done
Agent loop: done
Conditional agent decisions: done
OpenAI tool calling: done
Richer tool-calling decisions: done
Tool-call tracing and evaluation: done
Long-term memory tools: done
Memory-aware agent behavior: done
Tool-call quality labels: done
Memory quality evaluation: done
Automatic memory judging: done
Automatic tool-call judging: done
Memory cleanup suggestions: done
Agent run comparison reports: done
Agent run type tracking: done
Evaluation dashboard: done
Chat UI: done
```

## Next Concrete Task

Implement:

```text
Dashboard API
```

Purpose:

```text
Expose dashboard and chat-supporting data as JSON so the UI can become richer without duplicating backend logic.
```

Candidate endpoint:

```text
GET /dashboard/data
```

Learning goal:

```text
Separate data APIs from HTML rendering.
```
