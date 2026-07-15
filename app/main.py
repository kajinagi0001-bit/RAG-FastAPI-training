from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking import chunk_text
from app.database import Base, engine, get_db
from app.db_schema import ensure_schema
from app.embedding import deserialize_embedding, embed_text, serialize_embedding
from app.llm import generate_answer
from app.models import Chunk, Document
from app.retrieval import SearchableChunk, search_documents
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkRead,
    DocumentCreate,
    DocumentRead,
    Source,
)


Base.metadata.create_all(bind=engine)
ensure_schema()

app = FastAPI(title="RAG FastAPI Practice")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentRead, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> Document:
    document = Document(title=payload.title, content=payload.content)
    db.add(document)
    db.flush()

    for chunk_index, content in enumerate(chunk_text(payload.content)):
        embedding = embed_text(f"{payload.title} {content}")
        db.add(
            Chunk(
                document_id=document.id,
                content=content,
                chunk_index=chunk_index,
                embedding_json=serialize_embedding(embedding),
                embedding_model=_embedding_model_name(),
            )
        )

    db.commit()
    db.refresh(document)
    return document


@app.get("/documents", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@app.get("/documents/{document_id}", response_model=DocumentRead)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/documents/{document_id}/chunks", response_model=list[ChunkRead])
def list_document_chunks(document_id: int, db: Session = Depends(get_db)) -> list[ChunkRead]:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
        )
    )
    return [
        ChunkRead(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            embedding=_chunk_embedding(chunk),
        )
        for chunk in chunks
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    chunks = [
        SearchableChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=chunk.document.title,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            embedding=_chunk_embedding(chunk),
        )
        for chunk in db.scalars(select(Chunk).join(Chunk.document))
    ]
    query_embedding = embed_text(payload.question)
    results = search_documents(query_embedding, chunks, payload.top_k)

    if not results:
        return ChatResponse(
            answer="No relevant evidence was found in the database chunks.",
            sources=[],
        )

    answer = generate_answer(payload.question, results)

    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                document_id=result.chunk.document_id,
                chunk_id=result.chunk.chunk_id,
                chunk_index=result.chunk.chunk_index,
                title=result.chunk.title,
                score=result.score,
                content=result.chunk.content,
            )
            for result in results
        ],
    )


def _chunk_embedding(chunk: Chunk) -> list[float]:
    if chunk.embedding_json and chunk.embedding_model == _embedding_model_name():
        return deserialize_embedding(chunk.embedding_json)
    return embed_text(f"{chunk.document.title} {chunk.content}")


def _embedding_model_name() -> str:
    from app.settings import settings

    if settings.embedding_provider == "local":
        return f"local-hash-{settings.local_embedding_dimensions}"
    return settings.openai_embedding_model
