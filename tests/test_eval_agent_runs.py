from app.models import RagRun
from scripts.eval_agent_runs import (
    AgentRunRecord,
    classify_run,
    lowest_rated_runs,
    summarize_agent_runs,
)


def test_classify_run_identifies_tool_calling_agent() -> None:
    run = RagRun(
        id=1,
        question="Q",
        answer="A",
        retrieved_sources_json="[]",
        embedding_model="local",
        generation_model="local",
    )

    assert classify_run(run, tool_call_count=2) == "tool_calling_agent"


def test_classify_run_identifies_conversation_rag() -> None:
    run = RagRun(
        id=1,
        conversation_id=7,
        question="Q",
        answer="A",
        retrieved_sources_json="[]",
        embedding_model="local",
        generation_model="local",
    )

    assert classify_run(run, tool_call_count=0) == "conversation_rag"


def test_classify_run_prefers_explicit_run_type() -> None:
    run = RagRun(
        id=1,
        run_type="agent",
        question="Q",
        answer="A",
        retrieved_sources_json="[]",
        embedding_model="local",
        generation_model="local",
    )

    assert classify_run(run, tool_call_count=0) == "agent"


def test_summarize_agent_runs_groups_by_run_type() -> None:
    records = [
        AgentRunRecord(
            rag_run_id=1,
            run_type="chat_or_local_agent",
            question="Q1",
            tool_call_count=0,
            groundedness=5,
            answer_quality=4,
            source_usefulness=5,
        ),
        AgentRunRecord(
            rag_run_id=2,
            run_type="tool_calling_agent",
            question="Q2",
            tool_call_count=2,
            groundedness=3,
            answer_quality=3,
            source_usefulness=4,
        ),
    ]

    summary = summarize_agent_runs(records)

    assert summary["chat_or_local_agent"]["run_count"] == 1.0
    assert summary["chat_or_local_agent"]["avg_answer_quality"] == 4.0
    assert summary["tool_calling_agent"]["avg_tool_calls"] == 2.0


def test_lowest_rated_runs_orders_by_quality_scores() -> None:
    records = [
        AgentRunRecord(
            rag_run_id=1,
            run_type="chat_or_local_agent",
            question="High",
            tool_call_count=0,
            groundedness=5,
            answer_quality=5,
            source_usefulness=5,
        ),
        AgentRunRecord(
            rag_run_id=2,
            run_type="tool_calling_agent",
            question="Low",
            tool_call_count=2,
            groundedness=2,
            answer_quality=3,
            source_usefulness=2,
        ),
    ]

    assert lowest_rated_runs(records, limit=1)[0].rag_run_id == 2
