import app.agent as agent
from app.agent import run_agent
from app.embedding import embed_text_local
from app.retrieval import SearchableChunk, SearchResult
from app.schemas import ChatResponse, Source


def test_run_agent_records_steps_and_returns_answer(monkeypatch) -> None:
    chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="RAG retrieves relevant chunks.",
        chunk_index=2,
        embedding=embed_text_local("RAG retrieves relevant chunks."),
    )

    monkeypatch.setattr(
        agent.tools,
        "search_memories",
        lambda db, query, top_k: [],
    )
    monkeypatch.setattr(
        agent.tools,
        "search_knowledge_base",
        lambda db, question, top_k: [SearchResult(chunk=chunk, score=0.9)],
    )
    monkeypatch.setattr(
        agent.tools,
        "answer_with_context",
        lambda question, results, history=None: ChatResponse(
            answer="Agent answer from context.",
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

    logged = []
    monkeypatch.setattr(
        agent.tools,
        "log_rag_run",
        lambda db, question, response, conversation_id, run_type="unknown": logged.append(
            (question, run_type)
        ),
    )

    response = run_agent(db=object(), question="What does RAG retrieve?", top_k=3)

    assert response.answer == "Agent answer from context."
    assert response.sources[0].document_id == 7
    assert [step.action for step in response.steps] == [
        "plan",
        "search_memories",
        "search_knowledge_base",
        "decide_answer",
        "answer_with_context",
        "log_rag_run",
    ]
    assert logged == [("What does RAG retrieve?", "agent")]


def test_run_agent_retries_search_when_retrieval_is_weak(monkeypatch) -> None:
    weak_chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="Weak result.",
        chunk_index=0,
        embedding=embed_text_local("Weak result."),
    )
    strong_chunk = SearchableChunk(
        chunk_id=2,
        document_id=8,
        title="Agent",
        content="Agent loops can retry searches.",
        chunk_index=0,
        embedding=embed_text_local("Agent loops can retry searches."),
    )
    calls = []

    def fake_search(db, question, top_k):
        calls.append(top_k)
        if len(calls) == 1:
            return [SearchResult(chunk=weak_chunk, score=0.1)]
        return [SearchResult(chunk=strong_chunk, score=0.8)]

    monkeypatch.setattr(agent.tools, "search_knowledge_base", fake_search)
    monkeypatch.setattr(agent.tools, "search_memories", lambda db, query, top_k: [])
    monkeypatch.setattr(
        agent.tools,
        "answer_with_context",
        lambda question, results, history=None: ChatResponse(
            answer="Retried answer from stronger context.",
            sources=[
                Source(
                    document_id=results[0].chunk.document_id,
                    chunk_id=results[0].chunk.chunk_id,
                    chunk_index=results[0].chunk.chunk_index,
                    title=results[0].chunk.title,
                    score=results[0].score,
                    content=results[0].chunk.content,
                )
            ],
        ),
    )
    monkeypatch.setattr(
        agent.tools,
        "log_rag_run",
        lambda db, question, response, conversation_id, run_type="unknown": None,
    )

    response = run_agent(db=object(), question="How can an agent recover?", top_k=3)

    assert calls == [3, 6]
    assert response.answer == "Retried answer from stronger context."
    assert [step.action for step in response.steps] == [
        "plan",
        "search_memories",
        "search_knowledge_base",
        "decide_retry_search",
        "search_knowledge_base",
        "answer_with_context",
        "log_rag_run",
    ]


def test_run_agent_passes_memories_to_answer_history(monkeypatch) -> None:
    from app.models import Memory

    chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="RAG retrieves relevant chunks.",
        chunk_index=2,
        embedding=embed_text_local("RAG retrieves relevant chunks."),
    )
    memory = Memory(
        id=1,
        content="The user prefers implementation-first explanations.",
        source="user",
        embedding_json="[]",
        embedding_model="local-hash-64",
    )
    captured_history = []

    monkeypatch.setattr(agent.tools, "search_memories", lambda db, query, top_k: [(memory, 0.8)])
    monkeypatch.setattr(
        agent.tools,
        "search_knowledge_base",
        lambda db, question, top_k: [SearchResult(chunk=chunk, score=0.9)],
    )

    def fake_answer(question, results, history=None):
        captured_history.append(history)
        return ChatResponse(answer="Memory-aware answer.", sources=[])

    monkeypatch.setattr(agent.tools, "answer_with_context", fake_answer)
    monkeypatch.setattr(
        agent.tools,
        "log_rag_run",
        lambda db, question, response, conversation_id, run_type="unknown": None,
    )

    response = run_agent(db=object(), question="How should you explain this?", top_k=3)

    assert response.answer == "Memory-aware answer."
    assert captured_history[-1][-1][0] == "memory"
    assert "implementation-first" in captured_history[-1][-1][1]
