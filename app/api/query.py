from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import embedding, generator, storage, vector_store
from app.deps import get_session_id

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/query")
def query_documents(request: QueryRequest, session_id: str = Depends(get_session_id)):
    try:
        store = vector_store.load(session_id)
    except storage.StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if store is None:
        raise HTTPException(
            status_code=404, detail="No documents indexed yet. Upload a document first."
        )

    query_vector = embedding.get_embeddings([request.query])
    results = vector_store.search(session_id, query_vector, request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    try:
        answer = generator.generate_answer(request.query, results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    return {"query": request.query, "answer": answer, "sources": results}
