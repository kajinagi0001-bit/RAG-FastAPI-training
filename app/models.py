from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(index=True)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20), index=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class RagRun(Base):
    __tablename__ = "rag_runs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    retrieved_sources_json: Mapped[str] = mapped_column(Text)
    run_type: Mapped[str] = mapped_column(String(50), default="unknown", nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100))
    generation_model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    feedback: Mapped[list["RagRunFeedback"]] = relationship(
        back_populates="rag_run",
        cascade="all, delete-orphan",
    )
    tool_calls: Mapped[list["AgentToolCall"]] = relationship(
        back_populates="rag_run",
        cascade="all, delete-orphan",
    )


class RagRunFeedback(Base):
    __tablename__ = "rag_run_feedback"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    rag_run_id: Mapped[int] = mapped_column(ForeignKey("rag_runs.id"), index=True)
    groundedness: Mapped[int] = mapped_column(index=True)
    answer_quality: Mapped[int] = mapped_column(index=True)
    source_usefulness: Mapped[int] = mapped_column(index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    rag_run: Mapped[RagRun] = relationship(back_populates="feedback")


class AgentToolCall(Base):
    __tablename__ = "agent_tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    rag_run_id: Mapped[int] = mapped_column(ForeignKey("rag_runs.id"), index=True)
    step: Mapped[int] = mapped_column(index=True)
    tool_name: Mapped[str] = mapped_column(String(100), index=True)
    arguments_json: Mapped[str] = mapped_column(Text)
    output_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    rag_run: Mapped[RagRun] = relationship(back_populates="tool_calls")
    feedback: Mapped[list["AgentToolCallFeedback"]] = relationship(
        back_populates="tool_call",
        cascade="all, delete-orphan",
    )


class AgentToolCallFeedback(Base):
    __tablename__ = "agent_tool_call_feedback"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tool_call_id: Mapped[int] = mapped_column(ForeignKey("agent_tool_calls.id"), index=True)
    tool_choice_quality: Mapped[int] = mapped_column(index=True)
    argument_quality: Mapped[int] = mapped_column(index=True)
    output_usefulness: Mapped[int] = mapped_column(index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    tool_call: Mapped[AgentToolCall] = relationship(back_populates="feedback")


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    embedding_json: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    feedback: Mapped[list["MemoryFeedback"]] = relationship(
        back_populates="memory",
        cascade="all, delete-orphan",
    )


class MemoryFeedback(Base):
    __tablename__ = "memory_feedback"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    memory_id: Mapped[int] = mapped_column(ForeignKey("memories.id"), index=True)
    importance: Mapped[int] = mapped_column(index=True)
    accuracy: Mapped[int] = mapped_column(index=True)
    future_usefulness: Mapped[int] = mapped_column(index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    memory: Mapped[Memory] = relationship(back_populates="feedback")
