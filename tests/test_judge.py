import json

from app.judge import (
    parse_judge_json,
    parse_memory_judge_json,
    parse_tool_call_judge_json,
)


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


def test_parse_memory_judge_json_returns_scores() -> None:
    result = parse_memory_judge_json(
        json.dumps(
            {
                "importance": 5,
                "accuracy": 4,
                "future_usefulness": 5,
                "notes": "Useful durable preference.",
            }
        )
    )

    assert result.importance == 5
    assert result.accuracy == 4
    assert result.future_usefulness == 5
    assert result.notes == "Useful durable preference."


def test_parse_tool_call_judge_json_returns_scores() -> None:
    result = parse_tool_call_judge_json(
        json.dumps(
            {
                "tool_choice_quality": 5,
                "argument_quality": 4,
                "output_usefulness": 5,
                "notes": "Appropriate search call.",
            }
        )
    )

    assert result.tool_choice_quality == 5
    assert result.argument_quality == 4
    assert result.output_usefulness == 5
    assert result.notes == "Appropriate search call."


