from scripts.eval_retrieval import EvalCase, is_hit


def test_is_hit_matches_expected_text() -> None:
    case = EvalCase(
        question="What does RAG retrieve?",
        expected_text="relevant chunks",
        expected_document_id=None,
        top_k=3,
    )

    assert is_hit(case, document_id=1, content="RAG retrieves relevant chunks.")


def test_is_hit_requires_expected_document_id() -> None:
    case = EvalCase(
        question="What does memory store?",
        expected_text=None,
        expected_document_id=7,
        top_k=3,
    )

    assert is_hit(case, document_id=7, content="anything")
    assert not is_hit(case, document_id=8, content="anything")

