import hashlib
import json
import math

from app.retrieval import tokenize
from app.settings import settings


def embed_text(text: str) -> list[float]:
    if settings.embedding_provider == "local":
        return embed_text_local(text, settings.local_embedding_dimensions)
    if settings.embedding_provider == "openai":
        return embed_text_openai(text, settings.openai_embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def embed_text_openai(text: str, model: str) -> list[float]:
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model=model,
    )
    return response.data[0].embedding


def embed_text_local(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


def serialize_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def deserialize_embedding(value: str) -> list[float]:
    loaded = json.loads(value)
    return [float(item) for item in loaded]

