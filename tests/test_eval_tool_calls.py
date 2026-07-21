from scripts.eval_tool_calls import ToolCallRecord, summarize_tool_calls


def test_summarize_tool_calls_returns_zero_metrics_for_empty_records() -> None:
    metrics = summarize_tool_calls([])

    assert metrics["total_tool_calls"] == 0
    assert metrics["runs_with_tool_calls"] == 0
    assert metrics["avg_tool_calls_per_run"] == 0.0
    assert metrics["tool_counts"] == {}
    assert metrics["feedback_count"] == 0
    assert metrics["avg_tool_choice_quality"] == 0.0
    assert metrics["avg_argument_quality"] == 0.0
    assert metrics["avg_output_usefulness"] == 0.0
    assert metrics["low_quality_feedback_count"] == 0


def test_summarize_tool_calls_counts_tools_and_runs() -> None:
    records = [
        ToolCallRecord(rag_run_id=1, step=1, tool_name="search_knowledge_base"),
        ToolCallRecord(
            rag_run_id=1,
            step=2,
            tool_name="answer_with_context",
            tool_choice_quality=5,
            argument_quality=4,
            output_usefulness=5,
        ),
        ToolCallRecord(rag_run_id=2, step=1, tool_name="search_knowledge_base"),
        ToolCallRecord(
            rag_run_id=2,
            step=2,
            tool_name="get_document",
            tool_choice_quality=2,
            argument_quality=3,
            output_usefulness=2,
        ),
    ]

    metrics = summarize_tool_calls(records)

    assert metrics["total_tool_calls"] == 4
    assert metrics["runs_with_tool_calls"] == 2
    assert metrics["avg_tool_calls_per_run"] == 2.0
    assert metrics["tool_counts"] == {
        "search_knowledge_base": 2,
        "answer_with_context": 1,
        "get_document": 1,
    }
    assert metrics["feedback_count"] == 2
    assert metrics["avg_tool_choice_quality"] == 3.5
    assert metrics["avg_argument_quality"] == 3.5
    assert metrics["avg_output_usefulness"] == 3.5
    assert metrics["low_quality_feedback_count"] == 1
