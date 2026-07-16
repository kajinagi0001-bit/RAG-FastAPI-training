from fastapi import Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking import chunk_text
from app.database import Base, engine, get_db
from app.db_schema import ensure_schema
from app.embedding import deserialize_embedding, embed_text, serialize_embedding
from app.ingestion import extract_upload_text
from app.llm import generate_answer
from app.models import Chunk, Conversation, Document, Message
from app.retrieval import SearchableChunk, search_documents
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkRead,
    ConversationCreate,
    ConversationRead,
    DocumentCreate,
    DocumentRead,
    MessageRead,
    Source,
)
from app.settings import settings


Base.metadata.create_all(bind=engine)
ensure_schema()

app = FastAPI(title="RAG FastAPI Practice")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentRead, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> Document:
    return _create_document_with_chunks(
        db=db,
        title=payload.title,
        content=payload.content,
    )


@app.post("/documents/upload", response_model=DocumentRead, status_code=201)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> Document:
    filename = file.filename or "uploaded-document.txt"
    raw_content = await file.read()
    title, content = extract_upload_text(filename, raw_content)

    return _create_document_with_chunks(
        db=db,
        title=title,
        content=content,
    )


def _create_document_with_chunks(db: Session, title: str, content: str) -> Document:
    document = Document(title=title, content=content)
    db.add(document)
    db.flush()

    for chunk_index, chunk_content in enumerate(chunk_text(content)):
        embedding = embed_text(f"{title} {chunk_content}")
        db.add(
            Chunk(
                document_id=document.id,
                content=chunk_content,
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
    return _run_rag_chat(db=db, question=payload.question, top_k=payload.top_k)


@app.post("/conversations", response_model=ConversationRead, status_code=201)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
) -> Conversation:
    conversation = Conversation(title=payload.title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@app.get("/conversations", response_model=list[ConversationRead])
def list_conversations(db: Session = Depends(get_db)) -> list[Conversation]:
    return list(db.scalars(select(Conversation).order_by(Conversation.created_at.desc())))


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageRead])
def list_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
) -> list[Message]:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    )


@app.post("/conversations/{conversation_id}/chat", response_model=ChatResponse)
def chat_in_conversation(
    conversation_id: int,
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.add(
        Message(
            conversation_id=conversation_id,
            role="user",
            content=payload.question,
        )
    )
    db.flush()

    history = _recent_conversation_history(
        db=db,
        conversation_id=conversation_id,
        limit=settings.conversation_history_limit,
    )
    response = _run_rag_chat(
        db=db,
        question=payload.question,
        top_k=payload.top_k,
        history=history,
    )

    db.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.answer,
        )
    )
    db.commit()

    return response


def _run_rag_chat(
    db: Session,
    question: str,
    top_k: int,
    history: list[tuple[str, str]] | None = None,
) -> ChatResponse:
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
    query_embedding = embed_text(question)
    results = search_documents(query_embedding, chunks, top_k)

    if not results:
        return ChatResponse(
            answer="No relevant evidence was found in the database chunks.",
            sources=[],
        )

    answer = generate_answer(question, results, history=history)

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


def _recent_conversation_history(
    db: Session,
    conversation_id: int,
    limit: int,
) -> list[tuple[str, str]]:
    messages = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
    )
    messages.reverse()
    return [(message.role, message.content) for message in messages]


def _chunk_embedding(chunk: Chunk) -> list[float]:
    if chunk.embedding_json and chunk.embedding_model == _embedding_model_name():
        return deserialize_embedding(chunk.embedding_json)
    return embed_text(f"{chunk.document.title} {chunk.content}")


def _embedding_model_name() -> str:
    from app.settings import settings

    if settings.embedding_provider == "local":
        return f"local-hash-{settings.local_embedding_dimensions}"
    return settings.openai_embedding_model
