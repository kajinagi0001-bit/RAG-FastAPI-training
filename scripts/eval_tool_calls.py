from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import AgentToolCall


@dataclass(frozen=True)
class ToolCallRecord:
    rag_run_id: int
    step: int
    tool_name: str


def load_tool_calls(db: Session) -> list[ToolCallRecord]:
    rows = db.scalars(
        select(AgentToolCall).order_by(
            AgentToolCall.rag_run_id.asc(),
            AgentToolCall.step.asc(),
            AgentToolCall.id.asc(),
        )
    )
    return [
        ToolCallRecord(
            rag_run_id=row.rag_run_id,
            step=row.step,
            tool_name=row.tool_name,
        )
        for row in rows
    ]


def summarize_tool_calls(records: list[ToolCallRecord]) -> dict[str, object]:
    if not records:
        return {
            "total_tool_calls": 0,
            "runs_with_tool_calls": 0,
            "avg_tool_calls_per_run": 0.0,
            "tool_counts": {},
        }

    run_ids = {record.rag_run_id for record in records}
    tool_counts = Counter(record.tool_name for record in records)
    return {
        "total_tool_calls": len(records),
        "runs_with_tool_calls": len(run_ids),
        "avg_tool_calls_per_run": len(records) / len(run_ids),
        "tool_counts": dict(tool_counts),
    }


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        records = load_tool_calls(db)

    metrics = summarize_tool_calls(records)
    print(f"total_tool_calls: {metrics['total_tool_calls']}")
    print(f"runs_with_tool_calls: {metrics['runs_with_tool_calls']}")
    print(f"avg_tool_calls_per_run: {metrics['avg_tool_calls_per_run']:.2f}")
    print("tool_counts:")
    for tool_name, count in sorted(metrics["tool_counts"].items()):
        print(f"- {tool_name}: {count}")


if __name__ == "__main__":
    main()
