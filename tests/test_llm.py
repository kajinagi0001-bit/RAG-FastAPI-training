from app.embedding import embed_text_local
from app.llm import generate_answer_local
from app.retrieval import SearchableChunk, SearchResult


def test_generate_answer_local_includes_sources() -> None:
    chunk = SearchableChunk(
        chunk_id=1,
        document_id=7,
        title="RAG",
        content="RAG retrieves relevant chunks before generation.",
        chunk_index=2,
        embedding=embed_text_local("RAG retrieves relevant chunks before generation."),
    )

    answer = generate_answer_local("What does RAG do?", [SearchResult(chunk=chunk, score=0.8)])

    assert "[7:2]" in answer
    assert "RAG retrieves relevant chunks" in answer
