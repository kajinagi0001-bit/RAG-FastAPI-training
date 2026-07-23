from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embedding import deserialize_embedding, embed_text
from app.models import Chunk
from app.retrieval import SearchableChunk, SearchResult, search_documents
from app.settings import settings


# questionの埋め込みを作成し、DBから関連性の高いChunkを検索し、tok_k件のSearchResultを返す
def retrieve_chunks(db: Session, question: str, top_k: int) -> list[SearchResult]:
    query_embedding = embed_text(question)
    chunks = [
        SearchableChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=chunk.document.title,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            embedding=chunk_embedding(chunk, expected_dimensions=len(query_embedding)),
        )
        for chunk in db.scalars(select(Chunk).join(Chunk.document))
    ]
    return search_documents(query_embedding, chunks, top_k)

# Chunkのembeddingを取得する。DBに保存されているembeddingが存在し、かつ期待される次元数と一致する場合はそれを返す。そうでない場合は新たに埋め込みを作成して返す。
def chunk_embedding(chunk: Chunk, expected_dimensions: int | None = None) -> list[float]:
    if chunk.embedding_json and chunk.embedding_model == embedding_model_name():
        embedding = deserialize_embedding(chunk.embedding_json)
        if expected_dimensions is None or len(embedding) == expected_dimensions:
            return embedding
    return embed_text(f"{chunk.document.title} {chunk.content}")


def embedding_model_name() -> str:
    return settings.openai_embedding_model
