from io import BytesIO
from pathlib import Path

from fastapi import HTTPException


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def extract_upload_text(filename: str, raw_content: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported extensions: .md, .txt, .pdf",
        )

    if suffix == ".pdf":
        content = extract_pdf_text(raw_content)
    else:
        content = extract_text_file(raw_content)

    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return Path(filename).stem, content


def extract_text_file(raw_content: bytes) -> str:
    try:
        return raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be UTF-8 encoded.",
        ) from exc


def extract_pdf_text(raw_content: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw_content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from uploaded PDF.",
        ) from exc

    return "\n\n".join(page.strip() for page in pages if page.strip())

