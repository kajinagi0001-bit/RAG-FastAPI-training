import argparse
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import RagRun, RagRunFeedback


@dataclass(frozen=True)
class AnswerFeedback:
    rag_run_id: int
    question: str
    groundedness: int
    answer_quality: int
    source_usefulness: int
    notes: str | None


def load_feedback(db: Session) -> list[AnswerFeedback]:
    rows = db.execute(
        select(RagRunFeedback, RagRun)
        .join(RagRun, RagRun.id == RagRunFeedback.rag_run_id)
        .order_by(RagRunFeedback.created_at.desc(), RagRunFeedback.id.desc())
    ).all()

    return [
        AnswerFeedback(
            rag_run_id=feedback.rag_run_id,
            question=run.question,
            groundedness=feedback.groundedness,
            answer_quality=feedback.answer_quality,
            source_usefulness=feedback.source_usefulness,
            notes=feedback.notes,
        )
        for feedback, run in rows
    ]


def summarize_feedback(feedback: list[AnswerFeedback]) -> dict[str, float]:
    if not feedback:
        return {
            "total": 0.0,
            "avg_groundedness": 0.0,
            "avg_answer_quality": 0.0,
            "avg_source_usefulness": 0.0,
            "low_quality_count": 0.0,
        }

    total = len(feedback)
    return {
        "total": float(total),
        "avg_groundedness": average(item.groundedness for item in feedback),
        "avg_answer_quality": average(item.answer_quality for item in feedback),
        "avg_source_usefulness": average(item.source_usefulness for item in feedback),
        "low_quality_count": float(
            sum(
                1
                for item in feedback
                if min(item.groundedness, item.answer_quality, item.source_usefulness) <= 2
            )
        ),
    }


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def lowest_rated(feedback: list[AnswerFeedback], limit: int) -> list[AnswerFeedback]:
    return sorted(
        feedback,
        key=lambda item: (
            item.groundedness + item.answer_quality + item.source_usefulness,
            item.answer_quality,
            item.groundedness,
        ),
    )[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manually reviewed RAG answer quality.")
    parser.add_argument(
        "--low-limit",
        type=int,
        default=5,
        help="Number of low-rated answers to print.",
    )
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        feedback = load_feedback(db)

    metrics = summarize_feedback(feedback)
    print(f"total_feedback: {int(metrics['total'])}")
    print(f"avg_groundedness: {metrics['avg_groundedness']:.2f}")
    print(f"avg_answer_quality: {metrics['avg_answer_quality']:.2f}")
    print(f"avg_source_usefulness: {metrics['avg_source_usefulness']:.2f}")
    print(f"low_quality_count: {int(metrics['low_quality_count'])}")

    low_items = lowest_rated(feedback, args.low_limit)
    if low_items:
        print()
        print("lowest_rated:")
        for item in low_items:
            print(
                f"- rag_run_id={item.rag_run_id} "
                f"groundedness={item.groundedness} "
                f"answer_quality={item.answer_quality} "
                f"source_usefulness={item.source_usefulness} "
                f"question={item.question}"
            )


if __name__ == "__main__":
    main()
