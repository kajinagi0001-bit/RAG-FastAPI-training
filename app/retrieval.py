import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]+")


@dataclass(frozen=True)
class SearchableChunk:
    chunk_id: int
    document_id: int
    title: str
    content: str
    chunk_index: int
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    chunk: SearchableChunk
    score: float


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


# questionの埋め込みとChunkのコサイン類似度を計算し、スコアでソートしたtop_k件のSearchResultを返す
def search_documents(
    query_embedding: list[float],
    chunks: list[SearchableChunk],
    top_k: int,
) -> list[SearchResult]:
    from app.embedding import cosine_similarity

    scored: list[SearchResult] = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        if score > 0:
            scored.append(SearchResult(chunk=chunk, score=score))

    return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

