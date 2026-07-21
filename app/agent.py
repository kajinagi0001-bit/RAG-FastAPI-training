from sqlalchemy.orm import Session

import app.tools as tools
from app.schemas import AgentResponse, AgentStep


MIN_USEFUL_SCORE = 0.25
MAX_AGENT_TOP_K = 10
MEMORY_TOP_K = 3


def run_agent(
    db: Session,
    question: str,
    top_k: int,
    history: list[tuple[str, str]] | None = None,
    conversation_id: int | None = None,
) -> AgentResponse:
    steps: list[AgentStep] = [
        AgentStep(
            step=1,
            action="plan",
            observation="Search long-term memory and the knowledge base, then answer from available context.",
        )
    ]

    memory_matches = tools.search_memories(db=db, query=question, top_k=MEMORY_TOP_K)
    steps.append(
        AgentStep(
            step=2,
            action="search_memories",
            observation=_memory_observation(memory_matches),
        )
    )
    memory_history = _memory_history(memory_matches)
    combined_history = [*(history or []), *memory_history]

    results = tools.search_knowledge_base(db=db, question=question, top_k=top_k)
    steps.append(
        AgentStep(
            step=3,
            action="search_knowledge_base",
            observation=_search_observation(results, top_k),
        )
    )

    if _retrieval_is_weak(results) and top_k < MAX_AGENT_TOP_K:
        retry_top_k = min(MAX_AGENT_TOP_K, max(top_k + 2, top_k * 2))
        steps.append(
            AgentStep(
                step=4,
                action="decide_retry_search",
                observation=(
                    "Retrieval looked weak, so retry with a larger top_k "
                    f"({top_k} -> {retry_top_k})."
                ),
            )
        )
        results = tools.search_knowledge_base(db=db, question=question, top_k=retry_top_k)
        steps.append(
            AgentStep(
                step=5,
                action="search_knowledge_base",
                observation=_search_observation(results, retry_top_k),
            )
        )
    else:
        steps.append(
            AgentStep(
                step=4,
                action="decide_answer",
                observation="Retrieval looked usable, so answer with the current context.",
            )
        )

    chat_response = tools.answer_with_context(
        question=question,
        results=results,
        history=combined_history,
    )
    steps.append(
        AgentStep(
            step=len(steps) + 1,
            action="answer_with_context",
            observation="Generated the final answer from retrieved context.",
        )
    )

    tools.log_rag_run(
        db=db,
        question=question,
        response=chat_response,
        conversation_id=conversation_id,
        run_type="agent",
    )
    steps.append(
        AgentStep(
            step=len(steps) + 1,
            action="log_rag_run",
            observation="Saved the agent run for later evaluation.",
        )
    )

    return AgentResponse(
        answer=chat_response.answer,
        sources=chat_response.sources,
        steps=steps,
    )


def _retrieval_is_weak(results) -> bool:
    if not results:
        return True
    return max(result.score for result in results) < MIN_USEFUL_SCORE


def _search_observation(results, top_k: int) -> str:
    if not results:
        return f"Retrieved 0 chunks with top_k={top_k}."
    best_score = max(result.score for result in results)
    return f"Retrieved {len(results)} chunks with top_k={top_k}. Best score: {best_score:.3f}."


def _memory_observation(memory_matches) -> str:
    if not memory_matches:
        return "Retrieved 0 memories."
    best_score = max(score for _, score in memory_matches)
    return f"Retrieved {len(memory_matches)} memories. Best score: {best_score:.3f}."


def _memory_history(memory_matches) -> list[tuple[str, str]]:
    return [
        (
            "memory",
            f"{memory.content} (source={memory.source or 'unknown'}, score={score:.3f})",
        )
        for memory, score in memory_matches
    ]
