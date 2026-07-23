import json
from datetime import datetime

import app.tool_calling_agent as tool_calling_agent
from app.models import Chunk, Document, Memory
from app.retrieval import SearchableChunk, SearchResult
from app.schemas import ChatResponse, Source
from app.tool_calling_agent import ToolCallingState, execute_tool_call


def fake_embedding(_: str) -> list[float]:
    return [0.0, 1.0]


def test_execute_search_tool_call_stores_latest_results(monkeypatch) -> None:
    chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="RAG retrieves relevant chunks.",
        chunk_index=2,
        embedding=fake_embedding("RAG retrieves relevant chunks."),
    )
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "search_knowledge_base",
        lambda db, question, top_k: [SearchResult(chunk=chunk, score=0.9)],
    )
    state = ToolCallingState()

    output = execute_tool_call(
        db=object(),
        state=state,
        name="search_knowledge_base",
        raw_arguments=json.dumps({"question": "What does RAG retrieve?", "top_k": 3}),
        default_question="fallback",
        default_top_k=1,
    )

    data = json.loads(output)
    assert data["retrieved_count"] == 1
    assert data["sources"][0]["document_id"] == 7
    assert state.latest_results[0].score == 0.9


def test_execute_answer_tool_call_uses_latest_results(monkeypatch) -> None:
    chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="RAG retrieves relevant chunks.",
        chunk_index=2,
        embedding=fake_embedding("RAG retrieves relevant chunks."),
    )
    state = ToolCallingState(
        latest_results=[SearchResult(chunk=chunk, score=0.9)]
    )
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "answer_with_context",
        lambda question, results, history=None: ChatResponse(
            answer="Tool-called answer.",
            sources=[
                Source(
                    document_id=7,
                    chunk_id=1,
                    chunk_index=2,
                    title="RAG",
                    score=0.9,
                    content="RAG retrieves relevant chunks.",
                )
            ],
        ),
    )

    output = execute_tool_call(
        db=object(),
        state=state,
        name="answer_with_context",
        raw_arguments=json.dumps({"question": "What does RAG retrieve?"}),
        default_question="fallback",
        default_top_k=1,
    )

    data = json.loads(output)
    assert data["answer"] == "Tool-called answer."
    assert state.latest_response is not None
    assert state.latest_response.sources[0].document_id == 7


def test_execute_get_document_tool_call_returns_document(monkeypatch) -> None:
    document = Document(id=7, title="Agent plan", content="Agents can choose tools.")
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "get_document",
        lambda db, document_id: document,
    )

    output = execute_tool_call(
        db=object(),
        state=ToolCallingState(),
        name="get_document",
        raw_arguments=json.dumps({"document_id": 7}),
        default_question="fallback",
        default_top_k=1,
    )

    data = json.loads(output)
    assert data["id"] == 7
    assert data["title"] == "Agent plan"
    assert data["content"] == "Agents can choose tools."


def test_execute_get_document_chunks_tool_call_updates_latest_results(monkeypatch) -> None:
    document = Document(id=7, title="Agent plan", content="Agents can choose tools.")
    chunk = Chunk(
        id=3,
        document_id=7,
        document=document,
        content="Agents inspect document chunks before answering.",
        chunk_index=0,
    )
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "get_document_chunks",
        lambda db, document_id: [chunk],
    )
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "chunk_read_embedding",
        lambda chunk: fake_embedding(chunk.content),
    )
    state = ToolCallingState()

    output = execute_tool_call(
        db=object(),
        state=state,
        name="get_document_chunks",
        raw_arguments=json.dumps({"document_id": 7}),
        default_question="fallback",
        default_top_k=1,
    )

    data = json.loads(output)
    assert data["document_id"] == 7
    assert data["chunk_count"] == 1
    assert data["chunks"][0]["chunk_id"] == 3
    assert state.latest_results[0].chunk.title == "Agent plan"
    assert state.latest_results[0].score == 1.0


def test_execute_create_memory_tool_call_returns_memory(monkeypatch) -> None:
    memory = Memory(
        id=5,
        content="The user prefers concise explanations.",
        source="user",
        embedding_json="[]",
        embedding_model="text-embedding-3-small",
        created_at=datetime(2026, 7, 17, 0, 0, 0),
    )
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "create_memory",
        lambda db, content, source=None: memory,
    )

    output = execute_tool_call(
        db=object(),
        state=ToolCallingState(),
        name="create_memory",
        raw_arguments=json.dumps(
            {
                "content": "The user prefers concise explanations.",
                "source": "user",
            }
        ),
        default_question="fallback",
        default_top_k=1,
    )

    data = json.loads(output)
    assert data["id"] == 5
    assert data["source"] == "user"


def test_execute_search_memories_tool_call_returns_matches(monkeypatch) -> None:
    memory = Memory(
        id=5,
        content="The user prefers concise explanations.",
        source="user",
        embedding_json="[]",
        embedding_model="text-embedding-3-small",
        created_at=datetime(2026, 7, 17, 0, 0, 0),
    )
    monkeypatch.setattr(
        tool_calling_agent.tools,
        "search_memories",
        lambda db, query, top_k: [(memory, 0.8)],
    )

    output = execute_tool_call(
        db=object(),
        state=ToolCallingState(),
        name="search_memories",
        raw_arguments=json.dumps({"query": "explanation preference", "top_k": 3}),
        default_question="fallback",
        default_top_k=1,
    )

    data = json.loads(output)
    assert data["memory_count"] == 1
    assert data["memories"][0]["content"] == "The user prefers concise explanations."
    assert data["memories"][0]["score"] == 0.8
