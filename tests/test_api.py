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
        lambda question, results: "Generated answer from retrieved context.",
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

def test_list_document_chunks_returns_404_for_missing_document() -> None:
    response = client.get("/documents/999999/chunks")

    assert response.status_code == 404
