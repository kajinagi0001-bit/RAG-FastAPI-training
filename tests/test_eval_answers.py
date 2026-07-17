from scripts.eval_answers import AnswerFeedback, lowest_rated, summarize_feedback


def test_summarize_feedback_returns_zero_metrics_for_empty_feedback() -> None:
    metrics = summarize_feedback([])

    assert metrics["total"] == 0.0
    assert metrics["avg_groundedness"] == 0.0
    assert metrics["avg_answer_quality"] == 0.0
    assert metrics["avg_source_usefulness"] == 0.0
    assert metrics["low_quality_count"] == 0.0


def test_summarize_feedback_calculates_average_scores() -> None:
    feedback = [
        AnswerFeedback(
            rag_run_id=1,
            question="What is RAG?",
            groundedness=5,
            answer_quality=4,
            source_usefulness=5,
            notes=None,
        ),
        AnswerFeedback(
            rag_run_id=2,
            question="What is memory?",
            groundedness=2,
            answer_quality=3,
            source_usefulness=2,
            notes="Needs better sources.",
        ),
    ]

    metrics = summarize_feedback(feedback)

    assert metrics["total"] == 2.0
    assert metrics["avg_groundedness"] == 3.5
    assert metrics["avg_answer_quality"] == 3.5
    assert metrics["avg_source_usefulness"] == 3.5
    assert metrics["low_quality_count"] == 1.0


def test_lowest_rated_orders_by_total_score() -> None:
    feedback = [
        AnswerFeedback(
            rag_run_id=1,
            question="High score",
            groundedness=5,
            answer_quality=5,
            source_usefulness=5,
            notes=None,
        ),
        AnswerFeedback(
            rag_run_id=2,
            question="Low score",
            groundedness=2,
            answer_quality=2,
            source_usefulness=3,
            notes=None,
        ),
    ]

    assert lowest_rated(feedback, limit=1)[0].rag_run_id == 2
