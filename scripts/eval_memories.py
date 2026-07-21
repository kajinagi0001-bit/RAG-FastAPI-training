from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import Memory, MemoryFeedback


@dataclass(frozen=True)
class MemoryFeedbackRecord:
    memory_id: int
    content: str
    importance: int
    accuracy: int
    future_usefulness: int


@dataclass(frozen=True)
class MemoryCleanupSuggestion:
    memory_id: int
    content: str
    action: str
    reason: str
    avg_importance: float
    avg_accuracy: float
    avg_future_usefulness: float


def load_memory_feedback(db: Session) -> list[MemoryFeedbackRecord]:
    rows = db.execute(
        select(MemoryFeedback, Memory)
        .join(Memory, Memory.id == MemoryFeedback.memory_id)
        .order_by(MemoryFeedback.created_at.desc(), MemoryFeedback.id.desc())
    ).all()
    return [
        MemoryFeedbackRecord(
            memory_id=feedback.memory_id,
            content=memory.content,
            importance=feedback.importance,
            accuracy=feedback.accuracy,
            future_usefulness=feedback.future_usefulness,
        )
        for feedback, memory in rows
    ]


def summarize_memory_feedback(records: list[MemoryFeedbackRecord]) -> dict[str, float]:
    if not records:
        return {
            "feedback_count": 0.0,
            "avg_importance": 0.0,
            "avg_accuracy": 0.0,
            "avg_future_usefulness": 0.0,
            "low_quality_count": 0.0,
        }

    return {
        "feedback_count": float(len(records)),
        "avg_importance": average(record.importance for record in records),
        "avg_accuracy": average(record.accuracy for record in records),
        "avg_future_usefulness": average(record.future_usefulness for record in records),
        "low_quality_count": float(
            sum(
                1
                for record in records
                if min(record.importance, record.accuracy, record.future_usefulness) <= 2
            )
        ),
    }


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def lowest_rated_memories(
    records: list[MemoryFeedbackRecord],
    limit: int = 5,
) -> list[MemoryFeedbackRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.importance + record.accuracy + record.future_usefulness,
            record.future_usefulness,
            record.accuracy,
        ),
    )[:limit]


def suggest_memory_cleanup(records: list[MemoryFeedbackRecord]) -> list[MemoryCleanupSuggestion]:
    grouped: dict[int, list[MemoryFeedbackRecord]] = {}
    for record in records:
        grouped.setdefault(record.memory_id, []).append(record)

    suggestions = []
    for memory_id, items in grouped.items():
        avg_importance = average(item.importance for item in items)
        avg_accuracy = average(item.accuracy for item in items)
        avg_future_usefulness = average(item.future_usefulness for item in items)
        lowest_score = min(avg_importance, avg_accuracy, avg_future_usefulness)

        if lowest_score <= 2:
            action = "delete_candidate"
            reason = "At least one average quality dimension is 2 or lower."
        elif lowest_score < 4:
            action = "review"
            reason = "The memory is usable but has a weak quality dimension."
        else:
            action = "keep"
            reason = "All average quality dimensions are strong."

        suggestions.append(
            MemoryCleanupSuggestion(
                memory_id=memory_id,
                content=items[0].content,
                action=action,
                reason=reason,
                avg_importance=avg_importance,
                avg_accuracy=avg_accuracy,
                avg_future_usefulness=avg_future_usefulness,
            )
        )

    return sorted(
        suggestions,
        key=lambda item: (
            item.action != "delete_candidate",
            item.action != "review",
            item.memory_id,
        ),
    )


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        records = load_memory_feedback(db)

    metrics = summarize_memory_feedback(records)
    print(f"feedback_count: {int(metrics['feedback_count'])}")
    print(f"avg_importance: {metrics['avg_importance']:.2f}")
    print(f"avg_accuracy: {metrics['avg_accuracy']:.2f}")
    print(f"avg_future_usefulness: {metrics['avg_future_usefulness']:.2f}")
    print(f"low_quality_count: {int(metrics['low_quality_count'])}")

    low_items = lowest_rated_memories(records)
    if low_items:
        print()
        print("lowest_rated_memories:")
        for item in low_items:
            print(
                f"- memory_id={item.memory_id} "
                f"importance={item.importance} "
                f"accuracy={item.accuracy} "
                f"future_usefulness={item.future_usefulness} "
                f"content={item.content}"
            )

    suggestions = suggest_memory_cleanup(records)
    if suggestions:
        print()
        print("cleanup_suggestions:")
        for item in suggestions:
            print(
                f"- memory_id={item.memory_id} "
                f"action={item.action} "
                f"avg_importance={item.avg_importance:.2f} "
                f"avg_accuracy={item.avg_accuracy:.2f} "
                f"avg_future_usefulness={item.avg_future_usefulness:.2f} "
                f"reason={item.reason} "
                f"content={item.content}"
            )


if __name__ == "__main__":
    main()
