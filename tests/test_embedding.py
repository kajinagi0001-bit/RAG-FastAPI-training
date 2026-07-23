import pytest

from app.embedding import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
)


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    embedding = [0.0, 1.0, 0.0]

    assert cosine_similarity(embedding, embedding) == pytest.approx(1.0)


def test_embedding_can_round_trip_through_json() -> None:
    embedding = [0.1, 0.2, 0.3]

    assert deserialize_embedding(serialize_embedding(embedding)) == embedding
