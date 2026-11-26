from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import embedding, generator, storage, vector_store, verifier
from app.deps import get_session_id

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    verify: bool = False


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

    abstained = generator.is_abstention(answer)
    verification = None
    if request.verify and not abstained:
        try:
            verification = verifier.verify_answer(answer, results)
        except Exception as exc:
            verification = {"status": "failed", "error": str(exc)[:200]}

    return {
        "query": request.query,
        "answer": answer,
        "abstained": abstained,
        "sources": results,
        "verification": verification,
    }
