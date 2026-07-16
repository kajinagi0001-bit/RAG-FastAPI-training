from app.retrieval import SearchResult
from app.settings import settings


SYSTEM_INSTRUCTIONS = """
You are a RAG assistant.
Answer only from the provided context.
If the context does not contain enough information, say that the database context is insufficient.
Keep the answer concise and include source references like [document_id:chunk_index].
""".strip()


def generate_answer(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> str:
    if settings.generation_provider == "local":
        return generate_answer_local(question, results, history=history)
    if settings.generation_provider == "openai":
        return generate_answer_openai(question, results, history=history)
    raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")


def generate_answer_openai(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=settings.openai_generation_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=_build_user_prompt(question, results, history=history),
    )
    return response.output_text


def generate_answer_local(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> str:
    source_text = "\n".join(
        f"- [{result.chunk.document_id}:{result.chunk.chunk_index}] {result.chunk.content}"
        for result in results
    )
    history_text = _format_history(history)
    history_prefix = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    return (
        f"{history_prefix}"
        "Local generation fallback. The following retrieved chunks are relevant to "
        f"the question: {question}\n\n{source_text}"
    )


def _build_user_prompt(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> str:
    context = "\n\n".join(
        (
            f"Source [document_id={result.chunk.document_id}, "
            f"chunk_index={result.chunk.chunk_index}, "
            f"title={result.chunk.title}, score={result.score:.4f}]\n"
            f"{result.chunk.content}"
        )
        for result in results
    )
    history_text = _format_history(history)
    history_section = f"Conversation History:\n{history_text}\n\n" if history_text else ""
    return f"{history_section}Question:\n{question}\n\nContext:\n{context}"


def _format_history(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    return "\n".join(f"{role}: {content}" for role, content in history)
