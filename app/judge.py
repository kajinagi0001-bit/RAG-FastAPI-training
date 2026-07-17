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


@dataclass(frozen=True)
class JudgeResult:
    groundedness: int
    answer_quality: int
    source_usefulness: int
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


def _has_word_overlap(left: str, right: str) -> bool:
    left_words = _words(left)
    right_words = _words(right)
    return bool(left_words & right_words)


def _words(text: str) -> set[str]:
    return {word for word in text.lower().split() if len(word) >= 4}
