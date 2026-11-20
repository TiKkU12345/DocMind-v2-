from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import settings
from app.core import embedding, storage, vector_store
from app.deps import get_session_id
from app.services.extract import ExtractError, extract_text, safe_filename
from app.services.text_splitter import split_text

router = APIRouter()

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "samples"
SAMPLE_NAME = "sample_service_agreement.txt"


def _index(session_id: str, filename: str, text: str) -> dict:
    chunks = split_text(text)
    vectors = embedding.get_embeddings(chunks)

    try:
        total = vector_store.add_document(session_id, filename, chunks, vectors)
    except vector_store.SessionFullError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except storage.StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "filename": filename,
        "chunks_created": len(chunks),
        "total_chunks": total,
        "status": "indexed",
    }


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...), session_id: str = Depends(get_session_id)
):
    filename = safe_filename(file.filename or "")
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB.")

    try:
        text = extract_text(filename, data)
    except ExtractError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _index(session_id, filename, text)


@router.post("/sample")
def load_sample(session_id: str = Depends(get_session_id)):
    path = SAMPLE_DIR / SAMPLE_NAME
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample document is not available.")
    return _index(session_id, SAMPLE_NAME, path.read_text(encoding="utf-8"))


@router.get("/documents")
def documents(session_id: str = Depends(get_session_id)):
    try:
        return {"documents": vector_store.list_documents(session_id)}
    except storage.StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/session")
def clear_session(session_id: str = Depends(get_session_id)):
    try:
        vector_store.clear(session_id)
    except storage.StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"status": "cleared"}
