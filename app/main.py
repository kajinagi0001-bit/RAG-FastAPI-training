from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import run_agent
from app.database import Base, engine, get_db
from app.db_schema import ensure_schema
from app.dashboard import load_dashboard_data, render_dashboard_html
from app.chat_ui import render_chat_ui_html
from app.ingestion import extract_upload_text
from app.judge import judge_answer, judge_memory, judge_tool_call
from app.models import AgentToolCall, AgentToolCallFeedback, Conversation, Document, Memory, MemoryFeedback, Message, RagRun, RagRunFeedback
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AgentToolCallFeedbackCreate,
    AgentToolCallFeedbackRead,
    AgentToolCallRead,
    ChatRequest,
    ChatResponse,
    ChunkRead,
    ConversationCreate,
    ConversationRead,
    DocumentCreate,
    DocumentRead,
    MessageRead,
    MemoryCreate,
    MemoryFeedbackCreate,
    MemoryFeedbackRead,
    MemoryRead,
    MemorySearchRequest,
    MemorySearchResult,
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
    create_memory_feedback,
    create_memory,
    create_tool_call_feedback,
    get_document as tool_get_document,
    get_document_chunks as tool_get_document_chunks,
    list_memories as tool_list_memories,
    run_rag_chat,
    search_memories,
)


Base.metadata.create_all(bind=engine)
ensure_schema()

app = FastAPI(title="RAG FastAPI Practice")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def chat_ui() -> HTMLResponse:
    return HTMLResponse(render_chat_ui_html())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: Session = Depends(get_db)) -> HTMLResponse:
    data = load_dashboard_data(db)
    return HTMLResponse(render_dashboard_html(data))


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


@app.post("/memories", response_model=MemoryRead, status_code=201)
def create_long_term_memory(
    payload: MemoryCreate,
    db: Session = Depends(get_db),
) -> Memory:
    return create_memory(db=db, content=payload.content, source=payload.source)


@app.get("/memories", response_model=list[MemoryRead])
def list_long_term_memories(db: Session = Depends(get_db)) -> list[Memory]:
    return tool_list_memories(db)


@app.post("/memories/search", response_model=list[MemorySearchResult])
def search_long_term_memories(
    payload: MemorySearchRequest,
    db: Session = Depends(get_db),
) -> list[MemorySearchResult]:
    return [
        MemorySearchResult(
            id=memory.id,
            content=memory.content,
            source=memory.source,
            score=score,
            created_at=memory.created_at,
        )
        for memory, score in search_memories(db=db, query=payload.query, top_k=payload.top_k)
    ]


@app.post("/memories/{memory_id}/feedback", response_model=MemoryFeedbackRead, status_code=201)
def create_long_term_memory_feedback(
    memory_id: int,
    payload: MemoryFeedbackCreate,
    db: Session = Depends(get_db),
) -> MemoryFeedback:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    return create_memory_feedback(
        db=db,
        memory_id=memory_id,
        importance=payload.importance,
        accuracy=payload.accuracy,
        future_usefulness=payload.future_usefulness,
        notes=payload.notes,
    )


@app.get("/memories/{memory_id}/feedback", response_model=list[MemoryFeedbackRead])
def list_long_term_memory_feedback(
    memory_id: int,
    db: Session = Depends(get_db),
) -> list[MemoryFeedback]:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    return list(
        db.scalars(
            select(MemoryFeedback)
            .where(MemoryFeedback.memory_id == memory_id)
            .order_by(MemoryFeedback.created_at.desc(), MemoryFeedback.id.desc())
        )
    )


@app.post("/memories/{memory_id}/judge", response_model=MemoryFeedbackRead, status_code=201)
def judge_long_term_memory(
    memory_id: int,
    db: Session = Depends(get_db),
) -> MemoryFeedback:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    judge_result = judge_memory(content=memory.content, source=memory.source)
    return create_memory_feedback(
        db=db,
        memory_id=memory_id,
        importance=judge_result.importance,
        accuracy=judge_result.accuracy,
        future_usefulness=judge_result.future_usefulness,
        notes=f"Memory judge: {judge_result.notes}",
    )


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


@app.post("/tool-calls/{tool_call_id}/feedback", response_model=AgentToolCallFeedbackRead, status_code=201)
def create_agent_tool_call_feedback(
    tool_call_id: int,
    payload: AgentToolCallFeedbackCreate,
    db: Session = Depends(get_db),
) -> AgentToolCallFeedback:
    tool_call = db.get(AgentToolCall, tool_call_id)
    if tool_call is None:
        raise HTTPException(status_code=404, detail="Tool call not found")

    return create_tool_call_feedback(
        db=db,
        tool_call_id=tool_call_id,
        tool_choice_quality=payload.tool_choice_quality,
        argument_quality=payload.argument_quality,
        output_usefulness=payload.output_usefulness,
        notes=payload.notes,
    )


@app.get("/tool-calls/{tool_call_id}/feedback", response_model=list[AgentToolCallFeedbackRead])
def list_agent_tool_call_feedback(
    tool_call_id: int,
    db: Session = Depends(get_db),
) -> list[AgentToolCallFeedback]:
    tool_call = db.get(AgentToolCall, tool_call_id)
    if tool_call is None:
        raise HTTPException(status_code=404, detail="Tool call not found")

    return list(
        db.scalars(
            select(AgentToolCallFeedback)
            .where(AgentToolCallFeedback.tool_call_id == tool_call_id)
            .order_by(AgentToolCallFeedback.created_at.desc(), AgentToolCallFeedback.id.desc())
        )
    )


@app.post("/tool-calls/{tool_call_id}/judge", response_model=AgentToolCallFeedbackRead, status_code=201)
def judge_agent_tool_call(
    tool_call_id: int,
    db: Session = Depends(get_db),
) -> AgentToolCallFeedback:
    tool_call = db.get(AgentToolCall, tool_call_id)
    if tool_call is None:
        raise HTTPException(status_code=404, detail="Tool call not found")

    rag_run = db.get(RagRun, tool_call.rag_run_id)
    question = rag_run.question if rag_run is not None else ""
    judge_result = judge_tool_call(
        user_question=question,
        tool_name=tool_call.tool_name,
        arguments_json=tool_call.arguments_json,
        output_json=tool_call.output_json,
    )
    return create_tool_call_feedback(
        db=db,
        tool_call_id=tool_call_id,
        tool_choice_quality=judge_result.tool_choice_quality,
        argument_quality=judge_result.argument_quality,
        output_usefulness=judge_result.output_usefulness,
        notes=f"Tool call judge: {judge_result.notes}",
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
