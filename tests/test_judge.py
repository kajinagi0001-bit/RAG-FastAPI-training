import json

from app.judge import judge_answer_local, parse_judge_json


def test_parse_judge_json_returns_scores() -> None:
    result = parse_judge_json(
        json.dumps(
            {
                "groundedness": 5,
                "answer_quality": 4,
                "source_usefulness": 3,
                "notes": "Mostly grounded.",
            }
        )
    )

    assert result.groundedness == 5
    assert result.answer_quality == 4
    assert result.source_usefulness == 3
    assert result.notes == "Mostly grounded."


def test_parse_judge_json_rejects_out_of_range_score() -> None:
    try:
        parse_judge_json(
            json.dumps(
                {
                    "groundedness": 6,
                    "answer_quality": 4,
                    "source_usefulness": 3,
                    "notes": "Invalid score.",
                }
            )
        )
    except ValueError as exc:
        assert "between 1 and 5" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_judge_answer_local_penalizes_missing_sources() -> None:
    result = judge_answer_local(
        question="What is RAG?",
        answer="RAG retrieves context before answering.",
        retrieved_sources_json="[]",
    )

    assert result.groundedness == 1
    assert result.source_usefulness == 1
