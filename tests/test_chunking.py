from app.chunking import chunk_text


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    assert chunk_text("hello world", chunk_size=20, overlap=5) == ["hello world"]


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]

