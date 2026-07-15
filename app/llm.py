from app.retrieval import SearchResult
from app.settings import settings


SYSTEM_INSTRUCTIONS = """
You are a RAG assistant.
Answer only from the provided context.
If the context does not contain enough information, say that the database context is insufficient.
Keep the answer concise and include source references like [document_id:chunk_index].
""".strip()


def generate_answer(question: str, results: list[SearchResult]) -> str:
    if settings.generation_provider == "local":
        return generate_answer_local(question, results)
    if settings.generation_provider == "openai":
        return generate_answer_openai(question, results)
    raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")


def generate_answer_openai(question: str, results: list[SearchResult]) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=settings.openai_generation_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=_build_user_prompt(question, results),
    )
    return response.output_text


def generate_answer_local(question: str, results: list[SearchResult]) -> str:
    source_text = "\n".join(
        f"- [{result.chunk.document_id}:{result.chunk.chunk_index}] {result.chunk.content}"
        for result in results
    )
    return (
        "Local generation fallback. The following retrieved chunks are relevant to "
        f"the question: {question}\n\n{source_text}"
    )


def _build_user_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        (
            f"Source [document_id={result.chunk.document_id}, "
            f"chunk_index={result.chunk.chunk_index}, "
            f"title={result.chunk.title}, score={result.score:.4f}]\n"
            f"{result.chunk.content}"
        )
        for result in results
    )
    return f"Question:\n{question}\n\nContext:\n{context}"

