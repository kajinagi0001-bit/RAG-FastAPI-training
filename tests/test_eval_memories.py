from scripts.eval_memories import (
    MemoryFeedbackRecord,
    lowest_rated_memories,
    suggest_memory_cleanup,
    summarize_memory_feedback,
)


def test_summarize_memory_feedback_returns_zero_metrics_for_empty_records() -> None:
    metrics = summarize_memory_feedback([])

    assert metrics["feedback_count"] == 0.0
    assert metrics["avg_importance"] == 0.0
    assert metrics["avg_accuracy"] == 0.0
    assert metrics["avg_future_usefulness"] == 0.0
    assert metrics["low_quality_count"] == 0.0


def test_summarize_memory_feedback_calculates_average_scores() -> None:
    records = [
        MemoryFeedbackRecord(
            memory_id=1,
            content="The user prefers concise answers.",
            importance=5,
            accuracy=5,
            future_usefulness=4,
        ),
        MemoryFeedbackRecord(
            memory_id=2,
            content="A vague temporary preference.",
            importance=2,
            accuracy=3,
            future_usefulness=2,
        ),
    ]

    metrics = summarize_memory_feedback(records)

    assert metrics["feedback_count"] == 2.0
    assert metrics["avg_importance"] == 3.5
    assert metrics["avg_accuracy"] == 4.0
    assert metrics["avg_future_usefulness"] == 3.0
    assert metrics["low_quality_count"] == 1.0


def test_lowest_rated_memories_orders_by_total_score() -> None:
    records = [
        MemoryFeedbackRecord(
            memory_id=1,
            content="High quality memory.",
            importance=5,
            accuracy=5,
            future_usefulness=5,
        ),
        MemoryFeedbackRecord(
            memory_id=2,
            content="Low quality memory.",
            importance=2,
            accuracy=2,
            future_usefulness=3,
        ),
    ]

    assert lowest_rated_memories(records, limit=1)[0].memory_id == 2


def test_suggest_memory_cleanup_groups_records_and_recommends_actions() -> None:
    records = [
        MemoryFeedbackRecord(
            memory_id=1,
            content="Strong durable preference.",
            importance=5,
            accuracy=5,
            future_usefulness=5,
        ),
        MemoryFeedbackRecord(
            memory_id=2,
            content="Needs review.",
            importance=3,
            accuracy=4,
            future_usefulness=4,
        ),
        MemoryFeedbackRecord(
            memory_id=3,
            content="Delete candidate.",
            importance=2,
            accuracy=4,
            future_usefulness=3,
        ),
    ]

    suggestions = suggest_memory_cleanup(records)

    by_memory_id = {suggestion.memory_id: suggestion for suggestion in suggestions}
    assert by_memory_id[1].action == "keep"
    assert by_memory_id[2].action == "review"
    assert by_memory_id[3].action == "delete_candidate"
    assert suggestions[0].action == "delete_candidate"
