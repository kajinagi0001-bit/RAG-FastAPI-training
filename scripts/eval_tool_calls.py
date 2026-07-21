from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import AgentToolCall, AgentToolCallFeedback


@dataclass(frozen=True)
class ToolCallRecord:
    rag_run_id: int
    step: int
    tool_name: str
    tool_choice_quality: int | None = None
    argument_quality: int | None = None
    output_usefulness: int | None = None


def load_tool_calls(db: Session) -> list[ToolCallRecord]:
    rows = db.execute(
        select(AgentToolCall, AgentToolCallFeedback)
        .outerjoin(
            AgentToolCallFeedback,
            AgentToolCallFeedback.tool_call_id == AgentToolCall.id,
        )
        .order_by(
            AgentToolCall.rag_run_id.asc(),
            AgentToolCall.step.asc(),
            AgentToolCall.id.asc(),
        )
    ).all()
    return [
        ToolCallRecord(
            rag_run_id=tool_call.rag_run_id,
            step=tool_call.step,
            tool_name=tool_call.tool_name,
            tool_choice_quality=feedback.tool_choice_quality if feedback else None,
            argument_quality=feedback.argument_quality if feedback else None,
            output_usefulness=feedback.output_usefulness if feedback else None,
        )
        for tool_call, feedback in rows
    ]


def summarize_tool_calls(records: list[ToolCallRecord]) -> dict[str, object]:
    if not records:
        return {
            "total_tool_calls": 0,
            "runs_with_tool_calls": 0,
            "avg_tool_calls_per_run": 0.0,
            "tool_counts": {},
            "feedback_count": 0,
            "avg_tool_choice_quality": 0.0,
            "avg_argument_quality": 0.0,
            "avg_output_usefulness": 0.0,
            "low_quality_feedback_count": 0,
        }

    run_ids = {record.rag_run_id for record in records}
    tool_counts = Counter(record.tool_name for record in records)
    feedback_records = [
        record for record in records if record.tool_choice_quality is not None
    ]
    return {
        "total_tool_calls": len(records),
        "runs_with_tool_calls": len(run_ids),
        "avg_tool_calls_per_run": len(records) / len(run_ids),
        "tool_counts": dict(tool_counts),
        "feedback_count": len(feedback_records),
        "avg_tool_choice_quality": average(
            record.tool_choice_quality for record in feedback_records
        ),
        "avg_argument_quality": average(
            record.argument_quality for record in feedback_records
        ),
        "avg_output_usefulness": average(
            record.output_usefulness for record in feedback_records
        ),
        "low_quality_feedback_count": sum(
            1
            for record in feedback_records
            if min(
                record.tool_choice_quality,
                record.argument_quality,
                record.output_usefulness,
            )
            <= 2
        ),
    }


def average(values) -> float:
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        records = load_tool_calls(db)

    metrics = summarize_tool_calls(records)
    print(f"total_tool_calls: {metrics['total_tool_calls']}")
    print(f"runs_with_tool_calls: {metrics['runs_with_tool_calls']}")
    print(f"avg_tool_calls_per_run: {metrics['avg_tool_calls_per_run']:.2f}")
    print(f"feedback_count: {metrics['feedback_count']}")
    print(f"avg_tool_choice_quality: {metrics['avg_tool_choice_quality']:.2f}")
    print(f"avg_argument_quality: {metrics['avg_argument_quality']:.2f}")
    print(f"avg_output_usefulness: {metrics['avg_output_usefulness']:.2f}")
    print(f"low_quality_feedback_count: {metrics['low_quality_feedback_count']}")
    print("tool_counts:")
    for tool_name, count in sorted(metrics["tool_counts"].items()):
        print(f"- {tool_name}: {count}")


if __name__ == "__main__":
    main()
