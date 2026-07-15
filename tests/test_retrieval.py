from app.embedding import embed_text_local
from app.retrieval import SearchableChunk, search_documents, tokenize


def test_tokenize_handles_english_and_japanese() -> None:
    assert tokenize("RAG retrieves 文書") == {"rag", "retrieves", "文書"}


def test_search_documents_returns_best_match() -> None:
    chunks = [
        SearchableChunk(
            chunk_id=1,
            document_id=1,
            title="SQL",
            content="SQL stores relational data.",
            chunk_index=0,
            embedding=embed_text_local("SQL SQL stores relational data."),
        ),
        SearchableChunk(
            chunk_id=2,
            document_id=2,
            title="RAG",
            content="RAG retrieves relevant documents.",
            chunk_index=0,
            embedding=embed_text_local("RAG RAG retrieves relevant documents."),
        ),
    ]

    results = search_documents(
        embed_text_local("How does RAG retrieve documents?"),
        chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == 2
    assert results[0].score > 0
