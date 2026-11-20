import io
from pathlib import PurePath

from docx import Document
from pypdf import PdfReader

ALLOWED_EXTENSIONS = (".pdf", ".txt", ".docx")


class ExtractError(Exception):
    pass


def safe_filename(name: str) -> str:
    base = PurePath((name or "").replace("\\", "/")).name
    cleaned = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    return cleaned.strip()[:150] or "document"


def _pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ExtractError("Password-protected PDFs are not supported.")
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_text(filename: str, data: bytes) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractError("Only PDF, DOCX and TXT files are allowed.")
    try:
        if ext == ".pdf":
            text = _pdf(data)
        elif ext == ".docx":
            text = _docx(data)
        else:
            text = data.decode("utf-8", errors="ignore")
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError("Could not read this file. Is it corrupted?") from exc

    text = text.strip()
    if not text:
        raise ExtractError("No text found in the document (scanned PDFs are not supported).")
    return text
