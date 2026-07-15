import pytest

from app.embedding import (
    cosine_similarity,
    deserialize_embedding,
    embed_text_local,
    serialize_embedding,
)


def test_embed_text_returns_normalized_vector() -> None:
    embedding = embed_text_local("RAG retrieves documents")

    assert len(embedding) == 64
    assert cosine_similarity(embedding, embedding) == pytest.approx(1.0)


def test_embedding_can_round_trip_through_json() -> None:
    embedding = embed_text_local("SQLite stores chunk embeddings")

    assert deserialize_embedding(serialize_embedding(embedding)) == embedding
