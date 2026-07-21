from collections import Counter
from dataclasses import dataclass
from html import escape

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentToolCall, Memory, MemoryFeedback, RagRun, RagRunFeedback


@dataclass(frozen=True)
class DashboardRun:
    id: int
    run_type: str
    question: str
    answer_quality: int | None
    groundedness: int | None
    tool_call_count: int
    created_at: str


@dataclass(frozen=True)
class DashboardLowRatedRun:
    id: int
    run_type: str
    question: str
    groundedness: int
    answer_quality: int
    source_usefulness: int
    notes: str | None


@dataclass(frozen=True)
class DashboardToolCall:
    id: int
    rag_run_id: int
    step: int
    tool_name: str
    arguments_json: str
    output_json: str


@dataclass(frozen=True)
class DashboardMemoryCleanupCandidate:
    memory_id: int
    action: str
    reason: str
    avg_importance: float
    avg_accuracy: float
    avg_future_usefulness: float
    content: str


@dataclass(frozen=True)
class DashboardData:
    run_type_counts: dict[str, int]
    recent_runs: list[DashboardRun]
    low_rated_runs: list[DashboardLowRatedRun]
    recent_tool_calls: list[DashboardToolCall]
    memory_cleanup_candidates: list[DashboardMemoryCleanupCandidate]


def load_dashboard_data(db: Session, limit: int = 10) -> DashboardData:
    runs = list(db.scalars(select(RagRun).order_by(RagRun.created_at.desc(), RagRun.id.desc())))
    latest_feedback = _latest_feedback_by_run(db)
    tool_counts = _tool_counts_by_run(db)

    return DashboardData(
        run_type_counts=dict(Counter(run.run_type for run in runs)),
        recent_runs=[
            DashboardRun(
                id=run.id,
                run_type=run.run_type,
                question=run.question,
                answer_quality=latest_feedback[run.id].answer_quality
                if run.id in latest_feedback
                else None,
                groundedness=latest_feedback[run.id].groundedness
                if run.id in latest_feedback
                else None,
                tool_call_count=tool_counts.get(run.id, 0),
                created_at=run.created_at.isoformat() if run.created_at else "",
            )
            for run in runs[:limit]
        ],
        low_rated_runs=_load_low_rated_runs(db, limit),
        recent_tool_calls=_load_recent_tool_calls(db, limit),
        memory_cleanup_candidates=_load_memory_cleanup_candidates(db, limit),
    )


def render_dashboard_html(data: DashboardData) -> str:
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f6f8fb;
    }}
    header {{
      padding: 24px 32px;
      background: #ffffff;
      border-bottom: 1px solid #d9e2ec;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
    }}
    h2 {{
      margin-top: 28px;
      margin-bottom: 12px;
      font-size: 18px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .metric {{
      padding: 12px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
    }}
    .metric strong {{
      display: block;
      margin-top: 6px;
      font-size: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9e2ec;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid #e6edf5;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #eef3f8;
      font-weight: 700;
    }}
    a {{
      color: #0b7285;
      text-decoration: none;
    }}
    code {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    .empty {{
      padding: 14px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Evaluation Dashboard</h1>
  </header>
  <main>
    <h2>Run Type Summary</h2>
    {_render_run_type_summary(data.run_type_counts)}
    <h2>Recent Runs</h2>
    {_render_recent_runs(data.recent_runs)}
    <h2>Low Rated Answers</h2>
    {_render_low_rated_runs(data.low_rated_runs)}
    <h2>Recent Tool Calls</h2>
    {_render_recent_tool_calls(data.recent_tool_calls)}
    <h2>Memory Cleanup Candidates</h2>
    {_render_memory_cleanup_candidates(data.memory_cleanup_candidates)}
  </main>
</body>
</html>"""


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


def _load_low_rated_runs(db: Session, limit: int) -> list[DashboardLowRatedRun]:
    rows = db.execute(
        select(RagRunFeedback, RagRun)
        .join(RagRun, RagRun.id == RagRunFeedback.rag_run_id)
        .order_by(
            (
                RagRunFeedback.groundedness
                + RagRunFeedback.answer_quality
                + RagRunFeedback.source_usefulness
            ).asc(),
            RagRunFeedback.answer_quality.asc(),
            RagRunFeedback.id.desc(),
        )
        .limit(limit)
    ).all()

    return [
        DashboardLowRatedRun(
            id=run.id,
            run_type=run.run_type,
            question=run.question,
            groundedness=feedback.groundedness,
            answer_quality=feedback.answer_quality,
            source_usefulness=feedback.source_usefulness,
            notes=feedback.notes,
        )
        for feedback, run in rows
    ]


def _load_recent_tool_calls(db: Session, limit: int) -> list[DashboardToolCall]:
    tool_calls = db.scalars(
        select(AgentToolCall)
        .order_by(AgentToolCall.created_at.desc(), AgentToolCall.id.desc())
        .limit(limit)
    )
    return [
        DashboardToolCall(
            id=tool_call.id,
            rag_run_id=tool_call.rag_run_id,
            step=tool_call.step,
            tool_name=tool_call.tool_name,
            arguments_json=tool_call.arguments_json,
            output_json=tool_call.output_json,
        )
        for tool_call in tool_calls
    ]


def _load_memory_cleanup_candidates(
    db: Session,
    limit: int,
) -> list[DashboardMemoryCleanupCandidate]:
    rows = db.execute(select(MemoryFeedback, Memory).join(Memory, Memory.id == MemoryFeedback.memory_id)).all()
    grouped: dict[int, list[tuple[MemoryFeedback, Memory]]] = {}
    for feedback, memory in rows:
        grouped.setdefault(memory.id, []).append((feedback, memory))

    candidates: list[DashboardMemoryCleanupCandidate] = []
    for memory_id, items in grouped.items():
        feedback_items = [feedback for feedback, _ in items]
        memory = items[0][1]
        avg_importance = _average(feedback.importance for feedback in feedback_items)
        avg_accuracy = _average(feedback.accuracy for feedback in feedback_items)
        avg_future_usefulness = _average(feedback.future_usefulness for feedback in feedback_items)
        lowest_score = min(avg_importance, avg_accuracy, avg_future_usefulness)
        if lowest_score >= 4:
            continue
        action = "delete_candidate" if lowest_score <= 2 else "review"
        reason = (
            "At least one average quality dimension is 2 or lower."
            if action == "delete_candidate"
            else "The memory is usable but has a weak quality dimension."
        )
        candidates.append(
            DashboardMemoryCleanupCandidate(
                memory_id=memory_id,
                action=action,
                reason=reason,
                avg_importance=avg_importance,
                avg_accuracy=avg_accuracy,
                avg_future_usefulness=avg_future_usefulness,
                content=memory.content,
            )
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.action != "delete_candidate",
            item.avg_importance + item.avg_accuracy + item.avg_future_usefulness,
            item.memory_id,
        ),
    )[:limit]


def _average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _render_run_type_summary(counts: dict[str, int]) -> str:
    if not counts:
        return '<div class="empty">No runs yet.</div>'
    items = "".join(
        f'<div class="metric">{escape(run_type)}<strong>{count}</strong></div>'
        for run_type, count in sorted(counts.items())
    )
    return f'<div class="summary">{items}</div>'


def _render_recent_runs(runs: list[DashboardRun]) -> str:
    if not runs:
        return '<div class="empty">No recent runs.</div>'
    rows = "".join(
        "<tr>"
        f'<td><a href="/rag-runs/{run.id}">{run.id}</a></td>'
        f"<td>{escape(run.run_type)}</td>"
        f"<td>{escape(run.question)}</td>"
        f"<td>{_score(run.answer_quality)}</td>"
        f"<td>{_score(run.groundedness)}</td>"
        f"<td>{run.tool_call_count}</td>"
        f"<td>{escape(run.created_at)}</td>"
        "</tr>"
        for run in runs
    )
    return (
        "<table><thead><tr><th>ID</th><th>Type</th><th>Question</th>"
        "<th>Answer</th><th>Grounded</th><th>Tools</th><th>Created</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _render_low_rated_runs(runs: list[DashboardLowRatedRun]) -> str:
    if not runs:
        return '<div class="empty">No rated answers yet.</div>'
    rows = "".join(
        "<tr>"
        f'<td><a href="/rag-runs/{run.id}">{run.id}</a></td>'
        f"<td>{escape(run.run_type)}</td>"
        f"<td>{escape(run.question)}</td>"
        f"<td>{run.groundedness}/{run.answer_quality}/{run.source_usefulness}</td>"
        f"<td>{escape(run.notes or '')}</td>"
        "</tr>"
        for run in runs
    )
    return (
        "<table><thead><tr><th>ID</th><th>Type</th><th>Question</th>"
        "<th>G/A/S</th><th>Notes</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _render_recent_tool_calls(tool_calls: list[DashboardToolCall]) -> str:
    if not tool_calls:
        return '<div class="empty">No tool calls yet.</div>'
    rows = "".join(
        "<tr>"
        f'<td><a href="/rag-runs/{tool_call.rag_run_id}/tool-calls">{tool_call.rag_run_id}</a></td>'
        f"<td>{tool_call.step}</td>"
        f"<td>{escape(tool_call.tool_name)}</td>"
        f"<td><code>{escape(_shorten(tool_call.arguments_json))}</code></td>"
        f"<td><code>{escape(_shorten(tool_call.output_json))}</code></td>"
        "</tr>"
        for tool_call in tool_calls
    )
    return (
        "<table><thead><tr><th>Run ID</th><th>Step</th><th>Tool</th>"
        "<th>Arguments</th><th>Output</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _render_memory_cleanup_candidates(candidates: list[DashboardMemoryCleanupCandidate]) -> str:
    if not candidates:
        return '<div class="empty">No memory cleanup candidates.</div>'
    rows = "".join(
        "<tr>"
        f'<td><a href="/memories/{candidate.memory_id}/feedback">{candidate.memory_id}</a></td>'
        f"<td>{escape(candidate.action)}</td>"
        f"<td>{candidate.avg_importance:.2f}/{candidate.avg_accuracy:.2f}/{candidate.avg_future_usefulness:.2f}</td>"
        f"<td>{escape(candidate.reason)}</td>"
        f"<td>{escape(candidate.content)}</td>"
        "</tr>"
        for candidate in candidates
    )
    return (
        "<table><thead><tr><th>Memory ID</th><th>Action</th><th>I/A/F</th>"
        "<th>Reason</th><th>Content</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _score(value: int | None) -> str:
    return "-" if value is None else str(value)


def _shorten(value: str, max_length: int = 180) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."
