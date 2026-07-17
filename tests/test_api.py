from fastapi.testclient import TestClient

import app.main as main
import app.retrieval_service as retrieval_service
import app.tools as tools
from app.embedding import embed_text_local
from app.judge import JudgeResult
from app.main import app
from app.schemas import AgentResponse, AgentStep


client = TestClient(app)


def test_create_list_and_chat(monkeypatch) -> None:
    monkeypatch.setattr(tools, "embed_text", embed_text_local)
    monkeypatch.setattr(retrieval_service, "embed_text", embed_text_local)
    monkeypatch.setattr(retrieval_service, "embedding_model_name", lambda: "local-hash-64")
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

    runs = client.get("/rag-runs")
    assert runs.status_code == 200
    matching_runs = [
        run for run in runs.json() if run["question"] == "How does RAG retrieve documents?"
    ]
    assert matching_runs

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
        "search_knowledge_base",
        "decide_answer",
        "answer_with_context",
        "log_rag_run",
    ]

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


def test_upload_markdown_document(monkeypatch) -> None:
    monkeypatch.setattr(tools, "embed_text", embed_text_local)
    monkeypatch.setattr(retrieval_service, "embed_text", embed_text_local)
    monkeypatch.setattr(retrieval_service, "embedding_model_name", lambda: "local-hash-64")

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
    monkeypatch.setattr(tools, "embed_text", embed_text_local)
    monkeypatch.setattr(retrieval_service, "embed_text", embed_text_local)
    monkeypatch.setattr(retrieval_service, "embedding_model_name", lambda: "local-hash-64")

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


def test_list_document_chunks_returns_404_for_missing_document() -> None:
    response = client.get("/documents/999999/chunks")

    assert response.status_code == 404
