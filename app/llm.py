from app.retrieval import SearchResult
from app.settings import settings


SYSTEM_INSTRUCTIONS = """
あなたはRAGのアシスタントです。
提供されたコンテキストからのみ回答してください。
コンテキストに十分な情報がない場合は、データベースのコンテキストが不十分であることを伝えてください。
回答は簡潔にし、[document_id:chunk_index]とソース参照を含めてください。
""".strip()


def generate_answer(
    question: str,
    results: list[SearchResult],
    history: list[tuple[str, str]] | None = None,
) -> str:
    return generate_answer_openai(question, results, history=history)


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
