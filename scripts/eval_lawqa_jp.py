import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import run_agent
from app.database import Base, SessionLocal, engine
from app.db_schema import ensure_schema
from app.models import Document
from app.schemas import AgentResponse, ChatResponse
from app.tool_calling_agent import run_tool_calling_agent
from app.tools import create_document_with_chunks, run_rag_chat


DEFAULT_DATASET_PATH = Path("evals/lawqa_jp_20.json")
SUPPORTED_MODES = {"normal_rag", "agent", "tool_calling_agent"}


@dataclass(frozen=True)
class LawqaSample:
    filename: str
    context: str
    question: str
    instruction: str
    choices: str
    answer_label: str


@dataclass(frozen=True)
class LawqaEvalResult:
    filename: str
    expected_label: str
    predicted_label: str | None
    retrieval_hit: bool
    source_count: int
    answer: str


def load_lawqa_samples(path: Path, limit: int | None = None) -> list[LawqaSample]:
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    raw_samples = raw_data["samples"]
    if limit is not None:
        raw_samples = raw_samples[:limit]
    return [
        LawqaSample(
            filename=item["ファイル名"],
            context=item["コンテキスト"],
            question=item["問題文"],
            instruction=item["指示"],
            choices=item["選択肢"],
            answer_label=item["output"].strip().lower(),
        )
        for item in raw_samples
    ]


def ensure_lawqa_documents(db: Session, samples: list[LawqaSample]) -> None:
    existing_titles = set(
        db.scalars(
            select(Document.title).where(Document.title.like("lawqa_jp::%"))
        )
    )
    for sample in samples:
        title = document_title(sample)
        if title in existing_titles:
            continue
        create_document_with_chunks(db=db, title=title, content=sample.context)
        existing_titles.add(title)


def evaluate_samples(
    db: Session,
    samples: list[LawqaSample],
    mode: str,
    top_k: int,
) -> list[LawqaEvalResult]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    results = []
    for sample in samples:
        response = run_lawqa_case(db=db, sample=sample, mode=mode, top_k=top_k)
        db.commit()
        results.append(
            LawqaEvalResult(
                filename=sample.filename,
                expected_label=sample.answer_label,
                predicted_label=extract_answer_label(response.answer),
                retrieval_hit=any(
                    source.title == document_title(sample)
                    for source in response.sources
                ),
                source_count=len(response.sources),
                answer=response.answer,
            )
        )
    return results


def run_lawqa_case(
    db: Session,
    sample: LawqaSample,
    mode: str,
    top_k: int,
) -> ChatResponse | AgentResponse:
    question = build_lawqa_question(sample)
    if mode == "normal_rag":
        return run_rag_chat(db=db, question=question, top_k=top_k)
    if mode == "agent":
        return run_agent(db=db, question=question, top_k=top_k)
    if mode == "tool_calling_agent":
        return run_tool_calling_agent(db=db, question=question, top_k=top_k)
    raise ValueError(f"Unsupported mode: {mode}")


def build_lawqa_question(sample: LawqaSample) -> str:
    return (
        f"{sample.instruction}\n\n"
        f"問題文:\n{sample.question}\n\n"
        f"選択肢:\n{sample.choices}\n\n"
        "回答は a, b, c, d のいずれか1文字を最初に書いてください。"
    )


def extract_answer_label(answer: str) -> str | None:
    normalized = answer.strip().lower()
    patterns = [
        r"^(?:answer|回答|正解)?\s*[:：は]?\s*([abcd])(?:\b|[^a-z])",
        r"(?:answer|回答|正解|選択肢)\s*[:：は]?\s*([abcd])(?:\b|[^a-z])",
        r"\b([abcd])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return None


def summarize_results(results: list[LawqaEvalResult]) -> dict[str, float]:
    if not results:
        return {
            "total": 0.0,
            "answer_accuracy": 0.0,
            "retrieval_hit_rate": 0.0,
            "parsed_answer_rate": 0.0,
        }
    return {
        "total": float(len(results)),
        "answer_accuracy": sum(
            1 for result in results if result.predicted_label == result.expected_label
        )
        / len(results),
        "retrieval_hit_rate": sum(1 for result in results if result.retrieval_hit)
        / len(results),
        "parsed_answer_rate": sum(
            1 for result in results if result.predicted_label is not None
        )
        / len(results),
    }


def document_title(sample: LawqaSample) -> str:
    return f"lawqa_jp::{sample.filename}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG modes on lawqa_jp.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to local lawqa_jp subset JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default="normal_rag",
        help="Evaluation target mode.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print each prediction and retrieval hit.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found: {dataset_path}. "
            "Run `python -m scripts.download_lawqa_jp` first."
        )

    Base.metadata.create_all(bind=engine)
    ensure_schema()
    samples = load_lawqa_samples(dataset_path, limit=args.limit)
    with SessionLocal() as db:
        ensure_lawqa_documents(db, samples)
        results = evaluate_samples(
            db=db,
            samples=samples,
            mode=args.mode,
            top_k=args.top_k,
        )

    metrics = summarize_results(results)
    print(f"mode: {args.mode}")
    print(f"total: {int(metrics['total'])}")
    print(f"answer_accuracy: {metrics['answer_accuracy']:.2f}")
    print(f"retrieval_hit_rate: {metrics['retrieval_hit_rate']:.2f}")
    print(f"parsed_answer_rate: {metrics['parsed_answer_rate']:.2f}")

    if args.show_details:
        print()
        print("details:")
        for result in results:
            print(
                f"- expected={result.expected_label} "
                f"predicted={result.predicted_label or '-'} "
                f"retrieval_hit={result.retrieval_hit} "
                f"sources={result.source_count} "
                f"file={result.filename}"
            )


if __name__ == "__main__":
    main()
