import json
from dataclasses import dataclass

from app.settings import settings


JUDGE_INSTRUCTIONS = """
You are a strict evaluator for RAG answer quality.
Score the answer from 1 to 5 for each dimension:
- groundedness: whether the answer is supported by the retrieved sources
- answer_quality: whether the answer is useful and directly answers the question
- source_usefulness: whether the retrieved sources are useful for answering

Return only JSON with these keys:
groundedness, answer_quality, source_usefulness, notes
""".strip()


MEMORY_JUDGE_INSTRUCTIONS = """
You are a strict evaluator for long-term memory quality.
Score the memory from 1 to 5 for each dimension:
- importance: whether this is worth storing as durable memory
- accuracy: whether the memory is specific, clear, and likely correct
- future_usefulness: whether it is likely to help future answers

Return only JSON with these keys:
importance, accuracy, future_usefulness, notes
""".strip()


TOOL_CALL_JUDGE_INSTRUCTIONS = """
You are a strict evaluator for AI agent tool calls.
Score the tool call from 1 to 5 for each dimension:
- tool_choice_quality: whether the selected tool was appropriate for the user question
- argument_quality: whether the tool arguments were specific and correct
- output_usefulness: whether the tool output helped the agent answer or make progress

Return only JSON with these keys:
tool_choice_quality, argument_quality, output_usefulness, notes
""".strip()


@dataclass(frozen=True)
class JudgeResult:
    groundedness: int
    answer_quality: int
    source_usefulness: int
    notes: str


@dataclass(frozen=True)
class MemoryJudgeResult:
    importance: int
    accuracy: int
    future_usefulness: int
    notes: str


@dataclass(frozen=True)
class ToolCallJudgeResult:
    tool_choice_quality: int
    argument_quality: int
    output_usefulness: int
    notes: str


def judge_answer(
    question: str,
    answer: str,
    retrieved_sources_json: str,
) -> JudgeResult:
    if settings.generation_provider == "local":
        return judge_answer_local(question, answer, retrieved_sources_json)
    if settings.generation_provider == "openai":
        return judge_answer_openai(question, answer, retrieved_sources_json)
    raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")


def judge_memory(content: str, source: str | None) -> MemoryJudgeResult:
    if settings.generation_provider == "local":
        return judge_memory_local(content, source)
    if settings.generation_provider == "openai":
        return judge_memory_openai(content, source)
    raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")


def judge_tool_call(
    user_question: str,
    tool_name: str,
    arguments_json: str,
    output_json: str,
) -> ToolCallJudgeResult:
    if settings.generation_provider == "local":
        return judge_tool_call_local(user_question, tool_name, arguments_json, output_json)
    if settings.generation_provider == "openai":
        return judge_tool_call_openai(user_question, tool_name, arguments_json, output_json)
    raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")


def judge_answer_openai(
    question: str,
    answer: str,
    retrieved_sources_json: str,
) -> JudgeResult:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=settings.openai_generation_model,
        instructions=JUDGE_INSTRUCTIONS,
        input=_build_judge_prompt(question, answer, retrieved_sources_json),
    )
    return parse_judge_json(response.output_text)


def judge_memory_openai(content: str, source: str | None) -> MemoryJudgeResult:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=settings.openai_generation_model,
        instructions=MEMORY_JUDGE_INSTRUCTIONS,
        input=_build_memory_judge_prompt(content, source),
    )
    return parse_memory_judge_json(response.output_text)


def judge_tool_call_openai(
    user_question: str,
    tool_name: str,
    arguments_json: str,
    output_json: str,
) -> ToolCallJudgeResult:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=settings.openai_generation_model,
        instructions=TOOL_CALL_JUDGE_INSTRUCTIONS,
        input=_build_tool_call_judge_prompt(
            user_question=user_question,
            tool_name=tool_name,
            arguments_json=arguments_json,
            output_json=output_json,
        ),
    )
    return parse_tool_call_judge_json(response.output_text)


def judge_answer_local(
    question: str,
    answer: str,
    retrieved_sources_json: str,
) -> JudgeResult:
    sources = _load_sources(retrieved_sources_json)
    if not sources:
        return JudgeResult(
            groundedness=1,
            answer_quality=2 if answer else 1,
            source_usefulness=1,
            notes="Local judge: no retrieved sources were available.",
        )

    source_text = " ".join(str(source.get("content", "")) for source in sources)
    groundedness = 4 if _has_word_overlap(answer, source_text) else 2
    answer_quality = 4 if len(answer.strip()) >= 40 else 3
    source_usefulness = 4 if _has_word_overlap(question, source_text) else 2
    return JudgeResult(
        groundedness=groundedness,
        answer_quality=answer_quality,
        source_usefulness=source_usefulness,
        notes="Local judge: heuristic score based on source presence and word overlap.",
    )


def judge_memory_local(content: str, source: str | None) -> MemoryJudgeResult:
    stripped = content.strip()
    word_count = len(_words(stripped))
    importance = 4 if _looks_like_durable_memory(stripped) else 2
    accuracy = 4 if word_count >= 3 else 2
    future_usefulness = 4 if source or importance >= 4 else 3
    return MemoryJudgeResult(
        importance=importance,
        accuracy=accuracy,
        future_usefulness=future_usefulness,
        notes="Local judge: heuristic score based on specificity and durable-memory wording.",
    )


def judge_tool_call_local(
    user_question: str,
    tool_name: str,
    arguments_json: str,
    output_json: str,
) -> ToolCallJudgeResult:
    arguments = _load_json_object(arguments_json)
    output = _load_json_object(output_json)
    tool_choice_quality = 4 if tool_name in _known_tool_names() else 2
    argument_quality = 4 if arguments else 2
    output_usefulness = 2 if "error" in output else 4 if output else 2

    if tool_name in {"search_knowledge_base", "search_memories"}:
        query = str(arguments.get("question") or arguments.get("query") or "")
        if query and _has_word_overlap(user_question, query):
            argument_quality = 5

    return ToolCallJudgeResult(
        tool_choice_quality=tool_choice_quality,
        argument_quality=argument_quality,
        output_usefulness=output_usefulness,
        notes="Local judge: heuristic score based on known tool name, arguments, and output error status.",
    )


def parse_judge_json(raw_text: str) -> JudgeResult:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response was not valid JSON: {raw_text}") from exc

    return JudgeResult(
        groundedness=_score(data.get("groundedness")),
        answer_quality=_score(data.get("answer_quality")),
        source_usefulness=_score(data.get("source_usefulness")),
        notes=str(data.get("notes") or ""),
    )


def parse_memory_judge_json(raw_text: str) -> MemoryJudgeResult:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Memory judge response was not valid JSON: {raw_text}") from exc

    return MemoryJudgeResult(
        importance=_score(data.get("importance")),
        accuracy=_score(data.get("accuracy")),
        future_usefulness=_score(data.get("future_usefulness")),
        notes=str(data.get("notes") or ""),
    )


def parse_tool_call_judge_json(raw_text: str) -> ToolCallJudgeResult:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool call judge response was not valid JSON: {raw_text}") from exc

    return ToolCallJudgeResult(
        tool_choice_quality=_score(data.get("tool_choice_quality")),
        argument_quality=_score(data.get("argument_quality")),
        output_usefulness=_score(data.get("output_usefulness")),
        notes=str(data.get("notes") or ""),
    )


def _build_judge_prompt(
    question: str,
    answer: str,
    retrieved_sources_json: str,
) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Retrieved sources JSON:\n{retrieved_sources_json}"
    )


def _build_memory_judge_prompt(content: str, source: str | None) -> str:
    return (
        f"Memory content:\n{content}\n\n"
        f"Source:\n{source or 'unknown'}"
    )


def _build_tool_call_judge_prompt(
    user_question: str,
    tool_name: str,
    arguments_json: str,
    output_json: str,
) -> str:
    return (
        f"User question:\n{user_question}\n\n"
        f"Tool name:\n{tool_name}\n\n"
        f"Arguments JSON:\n{arguments_json}\n\n"
        f"Output JSON:\n{output_json}"
    )


def _score(value: object) -> int:
    score = int(value)
    if score < 1 or score > 5:
        raise ValueError(f"Judge score must be between 1 and 5: {score}")
    return score


def _load_sources(retrieved_sources_json: str) -> list[dict]:
    try:
        sources = json.loads(retrieved_sources_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(sources, list):
        return []
    return [source for source in sources if isinstance(source, dict)]


def _load_json_object(raw_text: str) -> dict:
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _has_word_overlap(left: str, right: str) -> bool:
    left_words = _words(left)
    right_words = _words(right)
    return bool(left_words & right_words)


def _words(text: str) -> set[str]:
    return {word for word in text.lower().split() if len(word) >= 4}


def _looks_like_durable_memory(content: str) -> bool:
    durable_terms = {
        "prefers",
        "preference",
        "likes",
        "uses",
        "wants",
        "needs",
        "works",
        "learning",
        "always",
        "usually",
    }
    return bool(_words(content) & durable_terms)


def _known_tool_names() -> set[str]:
    return {
        "search_knowledge_base",
        "get_document",
        "get_document_chunks",
        "answer_with_context",
        "create_memory",
        "search_memories",
    }
