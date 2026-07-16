from io import BytesIO

from pypdf import PdfWriter

from app.ingestion import extract_upload_text


def test_extract_upload_text_reads_txt() -> None:
    title, content = extract_upload_text("notes.txt", b"hello RAG")

    assert title == "notes"
    assert content == "hello RAG"


def test_extract_upload_text_rejects_unknown_extension() -> None:
    try:
        extract_upload_text("notes.csv", b"title,content")
    except Exception as exc:
        assert getattr(exc, "status_code") == 400
    else:
        raise AssertionError("Expected unsupported extension to fail")


def test_extract_upload_text_handles_empty_pdf() -> None:
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)

    try:
        extract_upload_text("blank.pdf", buffer.getvalue())
    except Exception as exc:
        assert getattr(exc, "status_code") == 400
    else:
        raise AssertionError("Expected blank PDF to fail")

