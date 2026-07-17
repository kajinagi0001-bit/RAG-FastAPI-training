import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from app.database import SessionLocal
from app.retrieval_service import retrieve_chunks


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_text: str | None
    expected_document_id: int | None
    top_k: int


def load_cases(path: Path) -> list[EvalCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=item["question"],
            expected_text=item.get("expected_text"),
            expected_document_id=item.get("expected_document_id"),
            top_k=int(item.get("top_k", 3)),
        )
        for item in raw_cases
    ]


def evaluate_case(case: EvalCase) -> tuple[bool, int | None]:
    with SessionLocal() as db:
        results = retrieve_chunks(db=db, question=case.question, top_k=case.top_k)

    for index, result in enumerate(results, start=1):
        if is_hit(case, result.chunk.document_id, result.chunk.content):
            return True, index
    return False, None


def is_hit(case: EvalCase, document_id: int, content: str) -> bool:
    document_matches = (
        case.expected_document_id is None
        or document_id == case.expected_document_id
    )
    text_matches = (
        case.expected_text is None
        or case.expected_text.lower() in content.lower()
    )
    return document_matches and text_matches


def summarize(cases: list[EvalCase]) -> dict[str, float]:
    if not cases:
        return {"total": 0, "hit@k": 0.0, "mrr": 0.0}

    hits = 0
    reciprocal_rank_sum = 0.0
    for case in cases:
        hit, rank = evaluate_case(case)
        if hit:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank

    return {
        "total": float(len(cases)),
        "hit@k": hits / len(cases),
        "mrr": reciprocal_rank_sum / len(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--cases",
        default="evals/retrieval_cases.json",
        help="Path to retrieval eval cases JSON.",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    metrics = summarize(cases)
    print(f"total: {int(metrics['total'])}")
    print(f"hit@k: {metrics['hit@k']:.2f}")
    print(f"mrr: {metrics['mrr']:.2f}")


if __name__ == "__main__":
    main()

