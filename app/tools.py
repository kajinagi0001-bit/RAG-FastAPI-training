import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking import chunk_text
from app.embedding import cosine_similarity, deserialize_embedding, embed_text, serialize_embedding
from app.llm import generate_answer
from app.models import AgentToolCall, AgentToolCallFeedback, Chunk, Document, Memory, MemoryFeedback, RagRun, RagRunFeedback
from app.retrieval import SearchResult
from app.retrieval_service import chunk_embedding, embedding_model_name, retrieve_chunks
from app.schemas import ChatResponse, Source
from app.settings import settings
from app.timing import TimingRecorder

# titleとcontentから、DocumentとChunkを作成し、DBに保存
def create_document_with_chunks(db: Session, title: str, content: str) -> Document:
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
                embedding_model=embedding_model_name(),
            )
        )

    db.commit()
    db.refresh(document)
    return document


# DBからDocumentを取得し、titleとcontentを返す
def get_document(db: Session, document_id: int) -> Document | None:
    return db.get(Document, document_id)


# documebt_idに紐づくChunkをDBから取得する
def get_document_chunks(db: Session, document_id: int) -> list[Chunk]:
    return list(
        db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
        )
    )


# DBからquestionに関連するChunkを検索し、SearchResultのリストを返す
def search_knowledge_base(db: Session, question: str, top_k: int) -> list[SearchResult]:
    return retrieve_chunks(db=db, question=question, top_k=top_k)



def answer_with_context(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> ChatResponse:
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


def run_rag_chat(
    db: Session,
    question: str,
    top_k: int,
    history: list[tuple[str, str]] | None = None,
    conversation_id: int | None = None,
) -> ChatResponse:
    timings = TimingRecorder()
    with timings.measure("retrieval"):
        results = search_knowledge_base(db=db, question=question, top_k=top_k)
    with timings.measure("generation"):
        response = answer_with_context(question=question, results=results, history=history)
    with timings.measure("log_rag_run"):
        log_rag_run(
            db=db,
            question=question,
            response=response,
            conversation_id=conversation_id,
            run_type="conversation_rag" if conversation_id is not None else "chat",
        )
    response.timings = timings.finish()
    return response


def log_rag_run(
    db: Session,
    question: str,
    response: ChatResponse,
    conversation_id: int | None,
    run_type: str = "unknown",
) -> RagRun:
    run = RagRun(
        conversation_id=conversation_id,
        question=question,
        answer=response.answer,
        retrieved_sources_json=json.dumps(
            [source.model_dump() for source in response.sources],
            ensure_ascii=False,
        ),
        run_type=run_type,
        embedding_model=embedding_model_name(),
        generation_model=settings.openai_generation_model,
    )
    db.add(run)
    db.flush()
    return run


def log_tool_call(
    db: Session,
    rag_run_id: int,
    step: int,
    tool_name: str,
    arguments_json: str,
    output_json: str,
) -> AgentToolCall:
    tool_call = AgentToolCall(
        rag_run_id=rag_run_id,
        step=step,
        tool_name=tool_name,
        arguments_json=arguments_json,
        output_json=output_json,
    )
    db.add(tool_call)
    return tool_call


def create_tool_call_feedback(
    db: Session,
    tool_call_id: int,
    tool_choice_quality: int,
    argument_quality: int,
    output_usefulness: int,
    notes: str | None,
) -> AgentToolCallFeedback:
    feedback = AgentToolCallFeedback(
        tool_call_id=tool_call_id,
        tool_choice_quality=tool_choice_quality,
        argument_quality=argument_quality,
        output_usefulness=output_usefulness,
        notes=notes,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def create_feedback(
    db: Session,
    rag_run_id: int,
    groundedness: int,
    answer_quality: int,
    source_usefulness: int,
    notes: str | None,
) -> RagRunFeedback:
    feedback = RagRunFeedback(
        rag_run_id=rag_run_id,
        groundedness=groundedness,
        answer_quality=answer_quality,
        source_usefulness=source_usefulness,
        notes=notes,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def chunk_read_embedding(chunk: Chunk) -> list[float]:
    return chunk_embedding(chunk)


def create_memory(db: Session, content: str, source: str | None = None) -> Memory:
    embedding = embed_text(content)
    memory = Memory(
        content=content,
        source=source,
        embedding_json=serialize_embedding(embedding),
        embedding_model=embedding_model_name(),
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


def list_memories(db: Session) -> list[Memory]:
    return list(db.scalars(select(Memory).order_by(Memory.created_at.desc(), Memory.id.desc())))


def search_memories(db: Session, query: str, top_k: int) -> list[tuple[Memory, float]]:
    query_embedding = embed_text(query)
    scored: list[tuple[Memory, float]] = []
    for memory in db.scalars(select(Memory)):
        if memory.embedding_model != embedding_model_name():
            continue
        score = cosine_similarity(query_embedding, deserialize_embedding(memory.embedding_json))
        if score > 0:
            scored.append((memory, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def create_memory_feedback(
    db: Session,
    memory_id: int,
    importance: int,
    accuracy: int,
    future_usefulness: int,
    notes: str | None,
) -> MemoryFeedback:
    feedback = MemoryFeedback(
        memory_id=memory_id,
        importance=importance,
        accuracy=accuracy,
        future_usefulness=future_usefulness,
        notes=notes,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback
