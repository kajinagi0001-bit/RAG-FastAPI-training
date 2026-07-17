from fastapi import Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import run_agent
from app.database import Base, engine, get_db
from app.db_schema import ensure_schema
from app.ingestion import extract_upload_text
from app.judge import judge_answer
from app.models import AgentToolCall, Conversation, Document, Message, RagRun, RagRunFeedback
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AgentToolCallRead,
    ChatRequest,
    ChatResponse,
    ChunkRead,
    ConversationCreate,
    ConversationRead,
    DocumentCreate,
    DocumentRead,
    MessageRead,
    RagRunFeedbackCreate,
    RagRunFeedbackRead,
    RagRunRead,
)
from app.settings import settings
from app.tool_calling_agent import run_tool_calling_agent
from app.tools import (
    chunk_read_embedding,
    create_document_with_chunks,
    create_feedback,
    get_document as tool_get_document,
    get_document_chunks as tool_get_document_chunks,
    run_rag_chat,
)


Base.metadata.create_all(bind=engine)
ensure_schema()

app = FastAPI(title="RAG FastAPI Practice")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentRead, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> Document:
    return create_document_with_chunks(
        db=db,
        title=payload.title,
        content=payload.content,
    )


@app.post("/documents/upload", response_model=DocumentRead, status_code=201)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> Document:
    filename = file.filename or "uploaded-document.txt"
    raw_content = await file.read()
    title, content = extract_upload_text(filename, raw_content)

    return create_document_with_chunks(
        db=db,
        title=title,
        content=content,
    )


@app.get("/documents", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@app.get("/documents/{document_id}", response_model=DocumentRead)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    document = tool_get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/documents/{document_id}/chunks", response_model=list[ChunkRead])
def list_document_chunks(document_id: int, db: Session = Depends(get_db)) -> list[ChunkRead]:
    document = tool_get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = tool_get_document_chunks(db, document_id)
    return [
        ChunkRead(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            embedding=chunk_read_embedding(chunk),
        )
        for chunk in chunks
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    response = run_rag_chat(db=db, question=payload.question, top_k=payload.top_k)
    db.commit()
    return response


@app.post("/agent", response_model=AgentResponse)
def agent(payload: AgentRequest, db: Session = Depends(get_db)) -> AgentResponse:
    response = run_agent(db=db, question=payload.question, top_k=payload.top_k)
    db.commit()
    return response


@app.post("/agent/tool-calling", response_model=AgentResponse)
def tool_calling_agent(payload: AgentRequest, db: Session = Depends(get_db)) -> AgentResponse:
    response = run_tool_calling_agent(db=db, question=payload.question, top_k=payload.top_k)
    db.commit()
    return response


@app.get("/rag-runs", response_model=list[RagRunRead])
def list_rag_runs(db: Session = Depends(get_db)) -> list[RagRun]:
    return list(db.scalars(select(RagRun).order_by(RagRun.created_at.desc(), RagRun.id.desc())))


@app.get("/rag-runs/{run_id}", response_model=RagRunRead)
def get_rag_run(run_id: int, db: Session = Depends(get_db)) -> RagRun:
    run = db.get(RagRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run not found")
    return run


@app.get("/rag-runs/{run_id}/tool-calls", response_model=list[AgentToolCallRead])
def list_rag_run_tool_calls(
    run_id: int,
    db: Session = Depends(get_db),
) -> list[AgentToolCall]:
    run = db.get(RagRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run not found")

    return list(
        db.scalars(
            select(AgentToolCall)
            .where(AgentToolCall.rag_run_id == run_id)
            .order_by(AgentToolCall.step.asc(), AgentToolCall.id.asc())
        )
    )


@app.post("/rag-runs/{run_id}/feedback", response_model=RagRunFeedbackRead, status_code=201)
def create_rag_run_feedback(
    run_id: int,
    payload: RagRunFeedbackCreate,
    db: Session = Depends(get_db),
) -> RagRunFeedback:
    run = db.get(RagRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run not found")

    return create_feedback(
        db=db,
        rag_run_id=run_id,
        groundedness=payload.groundedness,
        answer_quality=payload.answer_quality,
        source_usefulness=payload.source_usefulness,
        notes=payload.notes,
    )


@app.get("/rag-runs/{run_id}/feedback", response_model=list[RagRunFeedbackRead])
def list_rag_run_feedback(
    run_id: int,
    db: Session = Depends(get_db),
) -> list[RagRunFeedback]:
    run = db.get(RagRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run not found")

    return list(
        db.scalars(
            select(RagRunFeedback)
            .where(RagRunFeedback.rag_run_id == run_id)
            .order_by(RagRunFeedback.created_at.desc(), RagRunFeedback.id.desc())
        )
    )


@app.post("/rag-runs/{run_id}/judge", response_model=RagRunFeedbackRead, status_code=201)
def judge_rag_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> RagRunFeedback:
    run = db.get(RagRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run not found")

    judge_result = judge_answer(
        question=run.question,
        answer=run.answer,
        retrieved_sources_json=run.retrieved_sources_json,
    )
    return create_feedback(
        db=db,
        rag_run_id=run_id,
        groundedness=judge_result.groundedness,
        answer_quality=judge_result.answer_quality,
        source_usefulness=judge_result.source_usefulness,
        notes=f"LLM judge: {judge_result.notes}",
    )


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
    response = run_rag_chat(
        db=db,
        question=payload.question,
        top_k=payload.top_k,
        history=history,
        conversation_id=conversation_id,
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
