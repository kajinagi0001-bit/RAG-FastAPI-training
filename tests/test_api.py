from fastapi.testclient import TestClient
from datetime import datetime

import app.main as main
import app.retrieval_service as retrieval_service
import app.tools as tools
from app.judge import JudgeResult, MemoryJudgeResult, ToolCallJudgeResult
from app.main import app
from app.models import AgentToolCall, Memory, RagRun
from app.schemas import AgentResponse, AgentStep


client = TestClient(app)


def fake_embedding(_: str) -> list[float]:
    return [0.0, 1.0]


def test_chat_ui_returns_html() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Chat" in response.text
    assert "Document Upload" in response.text
    assert "Timing" in response.text
    assert "/documents/upload" in response.text
    assert "Evaluation Dashboard" in response.text
    assert "/agent/tool-calling" in response.text


def test_create_list_and_chat(monkeypatch) -> None:
    monkeypatch.setattr(tools, "embed_text", fake_embedding)
    monkeypatch.setattr(retrieval_service, "embed_text", fake_embedding)
    monkeypatch.setattr(retrieval_service, "embedding_model_name", lambda: "text-embedding-3-small")
    monkeypatch.setattr(tools, "search_memories", lambda db, query, top_k: [])
    monkeypatch.setattr(
        tools,
        "generate_answer",
        lambda question, results, history=None: "Generated answer from retrieved context.",
    )

    created = client.post(
        "/documents",
        json={
            "title": "RAG basics",
            "content": "RAG retrieves relevant documents and passes them to an LLM.",
        },
    )
    assert created.status_code == 201

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert any(document["title"] == "RAG basics" for document in listed.json())

    chunks = client.get(f"/documents/{created.json()['id']}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()[0]["chunk_index"] == 0
    assert chunks.json()[0]["document_id"] == created.json()["id"]
    assert chunks.json()[0]["embedding"]

    chat = client.post("/chat", json={"question": "How does RAG retrieve documents?"})
    assert chat.status_code == 200
    sources = chat.json()["sources"]
    assert sources
    assert sources[0]["chunk_id"]
    assert sources[0]["chunk_index"] == 0
    assert sources[0]["score"] > 0
    assert chat.json()["answer"] == "Generated answer from retrieved context."
    assert "total" in chat.json()["timings"]
    assert "retrieval" in chat.json()["timings"]
    assert "generation" in chat.json()["timings"]

    runs = client.get("/rag-runs")
    assert runs.status_code == 200
    matching_runs = [
        run for run in runs.json() if run["question"] == "How does RAG retrieve documents?"
    ]
    assert matching_runs
    assert matching_runs[0]["run_type"] == "chat"

    feedback = client.post(
        f"/rag-runs/{matching_runs[0]['id']}/feedback",
        json={
            "groundedness": 5,
            "answer_quality": 4,
            "source_usefulness": 5,
            "notes": "The answer used the retrieved context.",
        },
    )
    assert feedback.status_code == 201
    assert feedback.json()["rag_run_id"] == matching_runs[0]["id"]
    assert feedback.json()["groundedness"] == 5

    tool_calls = client.get(f"/rag-runs/{matching_runs[0]['id']}/tool-calls")
    assert tool_calls.status_code == 200
    assert isinstance(tool_calls.json(), list)

    listed_feedback = client.get(f"/rag-runs/{matching_runs[0]['id']}/feedback")
    assert listed_feedback.status_code == 200
    assert any(item["notes"] == "The answer used the retrieved context." for item in listed_feedback.json())

    monkeypatch.setattr(
        main,
        "judge_answer",
        lambda question, answer, retrieved_sources_json: JudgeResult(
            groundedness=4,
            answer_quality=4,
            source_usefulness=5,
            notes="The answer is supported by the retrieved source.",
        ),
        raising=False,
    )
    judged = client.post(f"/rag-runs/{matching_runs[0]['id']}/judge")
    assert judged.status_code == 201
    assert judged.json()["groundedness"] == 4
    assert judged.json()["notes"].startswith("LLM judge:")

    agent_response = client.post("/agent", json={"question": "How does RAG retrieve documents?"})
    assert agent_response.status_code == 200
    assert agent_response.json()["answer"] == "Generated answer from retrieved context."
    assert [step["action"] for step in agent_response.json()["steps"]] == [
        "plan",
        "search_memories",
        "search_knowledge_base",
        "decide_answer",
        "answer_with_context",
        "log_rag_run",
    ]
    assert "memory_search" in agent_response.json()["timings"]
    assert "generation" in agent_response.json()["timings"]
    agent_runs = [
        run
        for run in client.get("/rag-runs").json()
        if run["question"] == "How does RAG retrieve documents?"
    ]
    assert agent_runs[0]["run_type"] == "agent"

    monkeypatch.setattr(
        main,
        "run_tool_calling_agent",
        lambda db, question, top_k: AgentResponse(
            answer="Tool-calling agent answer.",
            sources=[],
            steps=[
                AgentStep(
                    step=1,
                    action="openai_tool_calling_start",
                    observation="Test tool-calling path.",
                )
            ],
        ),
    )
    tool_calling_agent = client.post(
        "/agent/tool-calling",
        json={"question": "How does RAG retrieve documents?"},
    )
    assert tool_calling_agent.status_code == 200
    assert tool_calling_agent.json()["answer"] == "Tool-calling agent answer."
    assert tool_calling_agent.json()["steps"]


def test_dashboard_returns_evaluation_html() -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Evaluation Dashboard" in response.text
    assert "Run Type Summary" in response.text
    assert "Recent Runs" in response.text
    assert "Low Rated Answers" in response.text
    assert "Recent Tool Calls" in response.text
    assert "Memory Cleanup Candidates" in response.text


def test_memory_endpoints(monkeypatch) -> None:
    memory = Memory(
        id=1,
        content="The user prefers implementation-first explanations.",
        source="user",
        embedding_json="[]",
        embedding_model="text-embedding-3-small",
        created_at=datetime(2026, 7, 17, 0, 0, 0),
    )
    monkeypatch.setattr(main, "create_memory", lambda db, content, source=None: memory)
    monkeypatch.setattr(
        main,
        "create_memory_feedback",
        lambda db, memory_id, importance, accuracy, future_usefulness, notes: main.MemoryFeedback(
            id=1,
            memory_id=memory_id,
            importance=importance,
            accuracy=accuracy,
            future_usefulness=future_usefulness,
            notes=notes,
            created_at=datetime(2026, 7, 17, 0, 0, 0),
        ),
        raising=False,
    )
    monkeypatch.setattr(main, "tool_list_memories", lambda db: [memory])
    monkeypatch.setattr(main, "search_memories", lambda db, query, top_k: [(memory, 0.75)])

    created = client.post(
        "/memories",
        json={
            "content": "The user prefers implementation-first explanations.",
            "source": "user",
        },
    )
    assert created.status_code == 201
    assert created.json()["content"] == "The user prefers implementation-first explanations."

    listed = client.get("/memories")
    assert listed.status_code == 200
    assert listed.json()[0]["source"] == "user"

    searched = client.post(
        "/memories/search",
        json={"query": "explanation preference", "top_k": 3},
    )
    assert searched.status_code == 200
    assert searched.json()[0]["score"] == 0.75

    monkeypatch.setattr(
        main,
        "judge_memory",
        lambda content, source: MemoryJudgeResult(
            importance=5,
            accuracy=4,
            future_usefulness=5,
            notes="Useful durable memory.",
        ),
    )

    class FakeDb:
        def get(self, model, item_id):
            if model is Memory and item_id == 1:
                return memory
            return None

    def override_get_db():
        yield FakeDb()

    app.dependency_overrides[main.get_db] = override_get_db
    try:
        judged = client.post("/memories/1/judge")
    finally:
        app.dependency_overrides.clear()
    assert judged.status_code == 201
    assert judged.json()["importance"] == 5
    assert judged.json()["notes"].startswith("Memory judge:")


def test_upload_markdown_document(monkeypatch) -> None:
    monkeypatch.setattr(tools, "embed_text", fake_embedding)
    monkeypatch.setattr(retrieval_service, "embed_text", fake_embedding)
    monkeypatch.setattr(retrieval_service, "embedding_model_name", lambda: "text-embedding-3-small")

    uploaded = client.post(
        "/documents/upload",
        files={
            "file": (
                "agent-notes.md",
                b"# Agent Notes\n\nRAG agents retrieve knowledge before answering.",
                "text/markdown",
            )
        },
    )

    assert uploaded.status_code == 201
    assert uploaded.json()["title"] == "agent-notes"
    assert "RAG agents retrieve" in uploaded.json()["content"]

    chunks = client.get(f"/documents/{uploaded.json()['id']}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()[0]["embedding"]


def test_upload_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.csv", b"title,content", "text/csv")},
    )

    assert response.status_code == 400


def test_conversation_chat_saves_messages(monkeypatch) -> None:
    monkeypatch.setattr(tools, "embed_text", fake_embedding)
    monkeypatch.setattr(retrieval_service, "embed_text", fake_embedding)
    monkeypatch.setattr(retrieval_service, "embedding_model_name", lambda: "text-embedding-3-small")

    captured_history = []

    def fake_generate_answer(question, results, history=None):
        captured_history.append(history)
        return "Conversation answer from retrieved context."

    monkeypatch.setattr(
        tools,
        "generate_answer",
        fake_generate_answer,
    )

    client.post(
        "/documents",
        json={
            "title": "Agent memory",
            "content": "Conversation memory stores user and assistant messages.",
        },
    )

    conversation = client.post("/conversations", json={"title": "Memory test"})
    assert conversation.status_code == 201
    conversation_id = conversation.json()["id"]

    listed = client.get("/conversations")
    assert listed.status_code == 200
    assert any(item["id"] == conversation_id for item in listed.json())

    chat = client.post(
        f"/conversations/{conversation_id}/chat",
        json={"question": "What does conversation memory store?"},
    )
    assert chat.status_code == 200
    assert chat.json()["answer"] == "Conversation answer from retrieved context."
    assert chat.json()["sources"]

    messages = client.get(f"/conversations/{conversation_id}/messages")
    assert messages.status_code == 200
    assert [message["role"] for message in messages.json()[-2:]] == ["user", "assistant"]
    assert messages.json()[-2]["content"] == "What does conversation memory store?"
    assert messages.json()[-1]["content"] == "Conversation answer from retrieved context."
    assert captured_history[-1] == [
        ("user", "What does conversation memory store?"),
    ]

    runs = client.get("/rag-runs")
    conversation_runs = [
        run
        for run in runs.json()
        if run["conversation_id"] == conversation_id
        and run["question"] == "What does conversation memory store?"
    ]
    assert conversation_runs
    run_detail = client.get(f"/rag-runs/{conversation_runs[0]['id']}")
    assert run_detail.status_code == 200
    assert "Conversation answer from retrieved context." in run_detail.json()["answer"]

    second_chat = client.post(
        f"/conversations/{conversation_id}/chat",
        json={"question": "What did I just ask?"},
    )
    assert second_chat.status_code == 200
    assert captured_history[-1][-2:] == [
        ("assistant", "Conversation answer from retrieved context."),
        ("user", "What did I just ask?"),
    ]


def test_conversation_chat_returns_404_for_missing_conversation() -> None:
    response = client.post(
        "/conversations/999999/chat",
        json={"question": "hello"},
    )

    assert response.status_code == 404


def test_rag_run_feedback_returns_404_for_missing_run() -> None:
    response = client.post(
        "/rag-runs/999999/feedback",
        json={
            "groundedness": 3,
            "answer_quality": 3,
            "source_usefulness": 3,
        },
    )

    assert response.status_code == 404


def test_rag_run_feedback_rejects_scores_outside_range() -> None:
    response = client.post(
        "/rag-runs/1/feedback",
        json={
            "groundedness": 6,
            "answer_quality": 3,
            "source_usefulness": 3,
        },
    )

    assert response.status_code == 422


def test_tool_call_feedback_returns_404_for_missing_tool_call() -> None:
    response = client.post(
        "/tool-calls/999999/feedback",
        json={
            "tool_choice_quality": 3,
            "argument_quality": 3,
            "output_usefulness": 3,
        },
    )

    assert response.status_code == 404


def test_tool_call_feedback_rejects_scores_outside_range() -> None:
    response = client.post(
        "/tool-calls/1/feedback",
        json={
            "tool_choice_quality": 6,
            "argument_quality": 3,
            "output_usefulness": 3,
        },
    )

    assert response.status_code == 422


def test_tool_call_judge_saves_feedback(monkeypatch) -> None:
    tool_call = AgentToolCall(
        id=1,
        rag_run_id=10,
        step=1,
        tool_name="search_knowledge_base",
        arguments_json='{"question":"RAG","top_k":3}',
        output_json='{"retrieved_count":1}',
    )
    rag_run = RagRun(
        id=10,
        question="How does RAG retrieve documents?",
        answer="",
        retrieved_sources_json="[]",
        embedding_model="text-embedding-3-small",
        generation_model="gpt-4o-mini",
    )

    class FakeDb:
        def get(self, model, item_id):
            if model is AgentToolCall and item_id == 1:
                return tool_call
            if model is RagRun and item_id == 10:
                return rag_run
            return None

    def override_get_db():
        yield FakeDb()

    monkeypatch.setattr(
        main,
        "judge_tool_call",
        lambda user_question, tool_name, arguments_json, output_json: ToolCallJudgeResult(
            tool_choice_quality=5,
            argument_quality=4,
            output_usefulness=5,
            notes="Appropriate tool call.",
        ),
    )
    monkeypatch.setattr(
        main,
        "create_tool_call_feedback",
        lambda db, tool_call_id, tool_choice_quality, argument_quality, output_usefulness, notes: main.AgentToolCallFeedback(
            id=1,
            tool_call_id=tool_call_id,
            tool_choice_quality=tool_choice_quality,
            argument_quality=argument_quality,
            output_usefulness=output_usefulness,
            notes=notes,
            created_at=datetime(2026, 7, 17, 0, 0, 0),
        ),
    )

    app.dependency_overrides[main.get_db] = override_get_db
    try:
        response = client.post("/tool-calls/1/judge")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["tool_choice_quality"] == 5
    assert response.json()["notes"].startswith("Tool call judge:")


def test_tool_call_judge_returns_404_for_missing_tool_call() -> None:
    response = client.post("/tool-calls/999999/judge")

    assert response.status_code == 404


def test_memory_feedback_returns_404_for_missing_memory() -> None:
    response = client.post(
        "/memories/999999/feedback",
        json={
            "importance": 3,
            "accuracy": 3,
            "future_usefulness": 3,
        },
    )

    assert response.status_code == 404


def test_memory_feedback_rejects_scores_outside_range() -> None:
    response = client.post(
        "/memories/1/feedback",
        json={
            "importance": 6,
            "accuracy": 3,
            "future_usefulness": 3,
        },
    )

    assert response.status_code == 422


def test_memory_judge_returns_404_for_missing_memory() -> None:
    response = client.post("/memories/999999/judge")

    assert response.status_code == 404


def test_list_document_chunks_returns_404_for_missing_document() -> None:
    response = client.get("/documents/999999/chunks")

    assert response.status_code == 404
