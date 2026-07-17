from scripts.eval_tool_calls import ToolCallRecord, summarize_tool_calls


def test_summarize_tool_calls_returns_zero_metrics_for_empty_records() -> None:
    metrics = summarize_tool_calls([])

    assert metrics["total_tool_calls"] == 0
    assert metrics["runs_with_tool_calls"] == 0
    assert metrics["avg_tool_calls_per_run"] == 0.0
    assert metrics["tool_counts"] == {}


def test_summarize_tool_calls_counts_tools_and_runs() -> None:
    records = [
        ToolCallRecord(rag_run_id=1, step=1, tool_name="search_knowledge_base"),
        ToolCallRecord(rag_run_id=1, step=2, tool_name="answer_with_context"),
        ToolCallRecord(rag_run_id=2, step=1, tool_name="search_knowledge_base"),
    ]

    metrics = summarize_tool_calls(records)

    assert metrics["total_tool_calls"] == 3
    assert metrics["runs_with_tool_calls"] == 2
    assert metrics["avg_tool_calls_per_run"] == 1.5
    assert metrics["tool_counts"] == {
        "search_knowledge_base": 2,
        "answer_with_context": 1,
    }
