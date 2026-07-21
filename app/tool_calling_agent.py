import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

import app.tools as tools
from app.agent import run_agent
from app.models import Chunk
from app.retrieval import SearchableChunk, SearchResult
from app.schemas import AgentResponse, AgentStep, ChatResponse
from app.settings import settings


TOOL_CALLING_INSTRUCTIONS = """
You are an agent that answers questions using the provided tools.
Use search_memories when user preferences or prior durable notes may affect the answer.
Use search_knowledge_base before answering factual questions about the local knowledge base.
Use get_document when you need document-level metadata or full document content.
Use get_document_chunks when you need to inspect all chunks for a specific document.
Use create_memory only when the user explicitly asks you to remember something.
Use answer_with_context after searching when you need to produce the final grounded answer.
""".strip()


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "Search local document chunks for information relevant to a question.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user question to search for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of chunks to retrieve.",
                },
            },
            "required": ["question", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_document",
        "description": "Fetch a document by id, including its title and full content.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "The document id to fetch.",
                }
            },
            "required": ["document_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_document_chunks",
        "description": "Fetch all chunks for a document by id.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "The document id whose chunks should be fetched.",
                }
            },
            "required": ["document_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "answer_with_context",
        "description": "Generate a final answer using the most recent search results.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer using retrieved context.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_memory",
        "description": "Store a durable memory for future conversations. Use only when explicitly asked to remember something.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store.",
                },
                "source": {
                    "type": ["string", "null"],
                    "description": "Optional source label, such as user or agent.",
                },
            },
            "required": ["content", "source"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_memories",
        "description": "Search durable memories for user preferences or long-term notes.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The memory search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve.",
                },
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolCallingState:
    latest_results: list[SearchResult] = field(default_factory=list)
    latest_response: ChatResponse | None = None


@dataclass(frozen=True)
class ToolCallTrace:
    step: int
    tool_name: str
    arguments_json: str
    output_json: str


def run_tool_calling_agent(
    db: Session,
    question: str,
    top_k: int,
    history: list[tuple[str, str]] | None = None,
    conversation_id: int | None = None,
    max_rounds: int = 4,
) -> AgentResponse:
    if settings.generation_provider != "openai":
        return run_agent(
            db=db,
            question=question,
            top_k=top_k,
            history=history,
            conversation_id=conversation_id,
        )

    from openai import OpenAI

    client = OpenAI()
    state = ToolCallingState()
    tool_call_traces: list[ToolCallTrace] = []
    steps = [
        AgentStep(
            step=1,
            action="openai_tool_calling_start",
            observation="Asked the model to choose from the available function tools.",
        )
    ]
    input_items: list[dict[str, Any]] = [
        {"role": "user", "content": f"Question: {question}\nDefault top_k: {top_k}"}
    ]

    response = None
    for _ in range(max_rounds):
        response = client.responses.create(
            model=settings.openai_generation_model,
            instructions=TOOL_CALLING_INSTRUCTIONS,
            tools=TOOL_DEFINITIONS,
            input=input_items,
            parallel_tool_calls=False,
        )
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            break

        input_items.extend(_response_output_as_input(response.output))
        for function_call in function_calls:
            output = execute_tool_call(
                db=db,
                state=state,
                name=function_call.name,
                raw_arguments=function_call.arguments,
                default_question=question,
                default_top_k=top_k,
                history=history,
            )
            steps.append(
                AgentStep(
                    step=len(steps) + 1,
                    action=function_call.name,
                    observation=output,
                )
            )
            tool_call_traces.append(
                ToolCallTrace(
                    step=len(tool_call_traces) + 1,
                    tool_name=function_call.name,
                    arguments_json=function_call.arguments or "{}",
                    output_json=output,
                )
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": output,
                }
            )

    final_answer = response.output_text if response is not None else ""
    if state.latest_response is None:
        state.latest_response = tools.answer_with_context(
            question=question,
            results=state.latest_results,
            history=history,
        )
    if not final_answer:
        final_answer = state.latest_response.answer

    final_response = ChatResponse(
        answer=final_answer,
        sources=state.latest_response.sources,
    )
    rag_run = tools.log_rag_run(
        db=db,
        question=question,
        response=final_response,
        conversation_id=conversation_id,
        run_type="tool_calling_agent",
    )
    for trace in tool_call_traces:
        tools.log_tool_call(
            db=db,
            rag_run_id=rag_run.id,
            step=trace.step,
            tool_name=trace.tool_name,
            arguments_json=trace.arguments_json,
            output_json=trace.output_json,
        )
    steps.append(
        AgentStep(
            step=len(steps) + 1,
            action="log_rag_run",
            observation="Saved the tool-calling agent run for later evaluation.",
        )
    )
    return AgentResponse(
        answer=final_response.answer,
        sources=final_response.sources,
        steps=steps,
    )


def execute_tool_call(
    db: Session,
    state: ToolCallingState,
    name: str,
    raw_arguments: str,
    default_question: str,
    default_top_k: int,
    history: list[tuple[str, str]] | None = None,
) -> str:
    arguments = json.loads(raw_arguments or "{}")
    if name == "search_knowledge_base":
        question = str(arguments.get("question") or default_question)
        top_k = int(arguments.get("top_k") or default_top_k)
        state.latest_results = tools.search_knowledge_base(
            db=db,
            question=question,
            top_k=top_k,
        )
        return json.dumps(
            {
                "retrieved_count": len(state.latest_results),
                "sources": [_result_to_dict(result) for result in state.latest_results],
            },
            ensure_ascii=False,
        )

    if name == "answer_with_context":
        question = str(arguments.get("question") or default_question)
        state.latest_response = tools.answer_with_context(
            question=question,
            results=state.latest_results,
            history=history,
        )
        return state.latest_response.model_dump_json()

    if name == "get_document":
        document_id = int(arguments["document_id"])
        document = tools.get_document(db=db, document_id=document_id)
        if document is None:
            return json.dumps(
                {"error": f"Document not found: {document_id}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "created_at": document.created_at.isoformat()
                if document.created_at is not None
                else None,
            },
            ensure_ascii=False,
        )

    if name == "get_document_chunks":
        document_id = int(arguments["document_id"])
        chunks = tools.get_document_chunks(db=db, document_id=document_id)
        state.latest_results = [_chunk_to_search_result(chunk) for chunk in chunks]
        return json.dumps(
            {
                "document_id": document_id,
                "chunk_count": len(chunks),
                "chunks": [_chunk_to_dict(chunk) for chunk in chunks],
            },
            ensure_ascii=False,
        )

    if name == "create_memory":
        memory = tools.create_memory(
            db=db,
            content=str(arguments["content"]),
            source=arguments.get("source"),
        )
        return json.dumps(
            {
                "id": memory.id,
                "content": memory.content,
                "source": memory.source,
                "created_at": memory.created_at.isoformat()
                if memory.created_at is not None
                else None,
            },
            ensure_ascii=False,
        )

    if name == "search_memories":
        query = str(arguments.get("query") or default_question)
        top_k = int(arguments.get("top_k") or default_top_k)
        memories = tools.search_memories(db=db, query=query, top_k=top_k)
        return json.dumps(
            {
                "memory_count": len(memories),
                "memories": [
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "source": memory.source,
                        "score": score,
                        "created_at": memory.created_at.isoformat()
                        if memory.created_at is not None
                        else None,
                    }
                    for memory, score in memories
                ],
            },
            ensure_ascii=False,
        )

    return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "document_id": result.chunk.document_id,
        "chunk_id": result.chunk.chunk_id,
        "chunk_index": result.chunk.chunk_index,
        "title": result.chunk.title,
        "score": result.score,
        "content": result.chunk.content,
    }


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.document.title,
        "content": chunk.content,
    }


def _chunk_to_search_result(chunk: Chunk) -> SearchResult:
    return SearchResult(
        chunk=SearchableChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=chunk.document.title,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            embedding=tools.chunk_read_embedding(chunk),
        ),
        score=1.0,
    )


def _response_output_as_input(output) -> list[dict[str, Any]]:
    return [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in output]
