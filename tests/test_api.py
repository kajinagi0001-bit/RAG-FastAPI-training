from fastapi.testclient import TestClient

import app.main as main
from app.embedding import embed_text_local
from app.main import app


client = TestClient(app)


def test_create_list_and_chat(monkeypatch) -> None:
    monkeypatch.setattr(main, "embed_text", embed_text_local)
    monkeypatch.setattr(main, "_embedding_model_name", lambda: "local-hash-64")
    monkeypatch.setattr(
        main,
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


def test_upload_markdown_document(monkeypatch) -> None:
    monkeypatch.setattr(main, "embed_text", embed_text_local)
    monkeypatch.setattr(main, "_embedding_model_name", lambda: "local-hash-64")

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
    monkeypatch.setattr(main, "embed_text", embed_text_local)
    monkeypatch.setattr(main, "_embedding_model_name", lambda: "local-hash-64")

    captured_history = []

    def fake_generate_answer(question, results, history=None):
        captured_history.append(history)
        return "Conversation answer from retrieved context."

    monkeypatch.setattr(
        main,
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


def test_list_document_chunks_returns_404_for_missing_document() -> None:
    response = client.get("/documents/999999/chunks")

    assert response.status_code == 404
