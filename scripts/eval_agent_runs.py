from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.db_schema import ensure_schema
from app.models import AgentToolCall, RagRun, RagRunFeedback


@dataclass(frozen=True)
class AgentRunRecord:
    rag_run_id: int
    run_type: str
    question: str
    tool_call_count: int
    groundedness: int | None = None
    answer_quality: int | None = None
    source_usefulness: int | None = None


def load_agent_runs(db: Session) -> list[AgentRunRecord]:
    runs = list(db.scalars(select(RagRun).order_by(RagRun.created_at.desc(), RagRun.id.desc())))
    feedback_by_run = _latest_feedback_by_run(db)
    tool_counts = _tool_counts_by_run(db)

    return [
        AgentRunRecord(
            rag_run_id=run.id,
            run_type=classify_run(run, tool_counts.get(run.id, 0)),
            question=run.question,
            tool_call_count=tool_counts.get(run.id, 0),
            groundedness=feedback_by_run.get(run.id).groundedness
            if run.id in feedback_by_run
            else None,
            answer_quality=feedback_by_run.get(run.id).answer_quality
            if run.id in feedback_by_run
            else None,
            source_usefulness=feedback_by_run.get(run.id).source_usefulness
            if run.id in feedback_by_run
            else None,
        )
        for run in runs
    ]


def classify_run(run: RagRun, tool_call_count: int) -> str:
    if run.run_type and run.run_type != "unknown":
        return run.run_type
    if tool_call_count > 0:
        return "tool_calling_agent"
    if run.conversation_id is not None:
        return "conversation_rag"
    return "chat_or_local_agent"


def summarize_agent_runs(records: list[AgentRunRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[AgentRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.run_type, []).append(record)

    return {
        run_type: {
            "run_count": float(len(items)),
            "feedback_count": float(sum(1 for item in items if item.answer_quality is not None)),
            "avg_groundedness": average(item.groundedness for item in items),
            "avg_answer_quality": average(item.answer_quality for item in items),
            "avg_source_usefulness": average(item.source_usefulness for item in items),
            "avg_tool_calls": average(item.tool_call_count for item in items),
        }
        for run_type, items in sorted(grouped.items())
    }


def lowest_rated_runs(records: list[AgentRunRecord], limit: int = 5) -> list[AgentRunRecord]:
    rated = [record for record in records if record.answer_quality is not None]
    return sorted(
        rated,
        key=lambda record: (
            record.groundedness + record.answer_quality + record.source_usefulness,
            record.answer_quality,
            record.groundedness,
        ),
    )[:limit]


def tool_usage_by_run_type(records: list[AgentRunRecord]) -> dict[str, int]:
    return dict(Counter(record.run_type for record in records if record.tool_call_count > 0))


def average(values) -> float:
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _latest_feedback_by_run(db: Session) -> dict[int, RagRunFeedback]:
    feedback_rows = db.scalars(
        select(RagRunFeedback).order_by(
            RagRunFeedback.rag_run_id.asc(),
            RagRunFeedback.created_at.desc(),
            RagRunFeedback.id.desc(),
        )
    )
    latest: dict[int, RagRunFeedback] = {}
    for feedback in feedback_rows:
        latest.setdefault(feedback.rag_run_id, feedback)
    return latest


def _tool_counts_by_run(db: Session) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for tool_call in db.scalars(select(AgentToolCall)):
        counts[tool_call.rag_run_id] += 1
    return dict(counts)


def main() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    with SessionLocal() as db:
        records = load_agent_runs(db)

    summary = summarize_agent_runs(records)
    print(f"total_runs: {len(records)}")
    print()
    print("run_type_summary:")
    for run_type, metrics in summary.items():
        print(
            f"- run_type={run_type} "
            f"run_count={int(metrics['run_count'])} "
            f"feedback_count={int(metrics['feedback_count'])} "
            f"avg_groundedness={metrics['avg_groundedness']:.2f} "
            f"avg_answer_quality={metrics['avg_answer_quality']:.2f} "
            f"avg_source_usefulness={metrics['avg_source_usefulness']:.2f} "
            f"avg_tool_calls={metrics['avg_tool_calls']:.2f}"
        )

    low_items = lowest_rated_runs(records)
    if low_items:
        print()
        print("lowest_rated_runs:")
        for item in low_items:
            print(
                f"- rag_run_id={item.rag_run_id} "
                f"run_type={item.run_type} "
                f"groundedness={item.groundedness} "
                f"answer_quality={item.answer_quality} "
                f"source_usefulness={item.source_usefulness} "
                f"tool_call_count={item.tool_call_count} "
                f"question={item.question}"
            )


if __name__ == "__main__":
    main()
