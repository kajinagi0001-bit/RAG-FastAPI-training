from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    created_at: datetime


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    content: str
    chunk_index: int
    embedding: list[float]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    document_id: int
    chunk_id: int
    chunk_index: int
    title: str
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class AgentRequest(ChatRequest):
    pass


class AgentStep(BaseModel):
    step: int
    action: str
    observation: str


class AgentResponse(ChatResponse):
    steps: list[AgentStep]


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime


class RagRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int | None
    question: str
    answer: str
    retrieved_sources_json: str
    embedding_model: str
    generation_model: str
    created_at: datetime


class RagRunFeedbackCreate(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    answer_quality: int = Field(ge=1, le=5)
    source_usefulness: int = Field(ge=1, le=5)
    notes: str | None = None


class RagRunFeedbackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    rag_run_id: int
    groundedness: int
    answer_quality: int
    source_usefulness: int
    notes: str | None
    created_at: datetime


class AgentToolCallRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    rag_run_id: int
    step: int
    tool_name: str
    arguments_json: str
    output_json: str
    created_at: datetime
