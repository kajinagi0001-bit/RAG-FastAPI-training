import app.tools as tools
from app.embedding import embed_text_local
from app.retrieval import SearchableChunk, SearchResult
from app.tools import (
    answer_with_context,
    create_memory_feedback,
    create_tool_call_feedback,
    log_rag_run,
    log_tool_call,
)


class FakeDb:
    def __init__(self) -> None:
        self.added = []

    def add(self, item) -> None:
        self.added.append(item)

    def commit(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def refresh(self, item) -> None:
        pass


def test_answer_with_context_returns_no_evidence_message_for_empty_results() -> None:
    response = answer_with_context(question="What is RAG?", results=[])

    assert response.sources == []
    assert "No relevant evidence" in response.answer


def test_answer_with_context_converts_search_results_to_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "generate_answer",
        lambda question, results, history=None: "Answer based on context.",
    )
    chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="RAG retrieves relevant chunks.",
        chunk_index=2,
        embedding=embed_text_local("RAG retrieves relevant chunks."),
    )

    response = answer_with_context(
        question="What does RAG retrieve?",
        results=[SearchResult(chunk=chunk, score=0.9)],
    )

    assert response.answer == "Answer based on context."
    assert response.sources[0].document_id == 7
    assert response.sources[0].chunk_index == 2
    assert response.sources[0].score == 0.9


def test_log_tool_call_adds_trace_to_db() -> None:
    db = FakeDb()

    tool_call = log_tool_call(
        db=db,
        rag_run_id=10,
        step=1,
        tool_name="search_knowledge_base",
        arguments_json='{"question":"RAG"}',
        output_json='{"retrieved_count":1}',
    )

    assert db.added == [tool_call]
    assert tool_call.rag_run_id == 10
    assert tool_call.tool_name == "search_knowledge_base"


def test_log_rag_run_saves_run_type() -> None:
    db = FakeDb()
    response = answer_with_context(question="What is RAG?", results=[])

    run = log_rag_run(
        db=db,
        question="What is RAG?",
        response=response,
        conversation_id=None,
        run_type="chat",
    )

    assert db.added == [run]
    assert run.run_type == "chat"
    assert run.question == "What is RAG?"


def test_create_tool_call_feedback_adds_feedback_to_db() -> None:
    db = FakeDb()

    feedback = create_tool_call_feedback(
        db=db,
        tool_call_id=5,
        tool_choice_quality=4,
        argument_quality=5,
        output_usefulness=4,
        notes="Useful search call.",
    )

    assert db.added == [feedback]
    assert feedback.tool_call_id == 5
    assert feedback.argument_quality == 5


def test_create_memory_feedback_adds_feedback_to_db() -> None:
    db = FakeDb()

    feedback = create_memory_feedback(
        db=db,
        memory_id=3,
        importance=5,
        accuracy=4,
        future_usefulness=5,
        notes="Worth keeping.",
    )

    assert db.added == [feedback]
    assert feedback.memory_id == 3
    assert feedback.future_usefulness == 5
