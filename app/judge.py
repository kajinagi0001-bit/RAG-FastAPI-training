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
    return judge_answer_openai(question, answer, retrieved_sources_json)


def judge_memory(content: str, source: str | None) -> MemoryJudgeResult:
    return judge_memory_openai(content, source)


def judge_tool_call(
    user_question: str,
    tool_name: str,
    arguments_json: str,
    output_json: str,
) -> ToolCallJudgeResult:
    return judge_tool_call_openai(user_question, tool_name, arguments_json, output_json)


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


