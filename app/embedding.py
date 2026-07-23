import json
import math

from app.settings import settings


def embed_text(text: str) -> list[float]:
    return embed_text_openai(text, settings.openai_embedding_model)


def embed_text_openai(text: str, model: str) -> list[float]:
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model=model,
    )
    return response.data[0].embedding


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
