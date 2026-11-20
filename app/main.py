import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core import embedding, storage
from app.api.query import router as query_router
from app.api.upload import router as upload_router


def _background_startup() -> None:
    try:
        embedding.warmup()
    except Exception as exc:
        print(f"[startup] embedding warmup failed: {exc}")
    try:
        removed = storage.cleanup_old_sessions(settings.session_ttl_days)
        print(f"[startup] cleanup removed {removed} stale session(s)")
    except Exception as exc:
        print(f"[startup] cleanup failed: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_background_startup, daemon=True).start()
    yield


app = FastAPI(
    title="DocMind API",
    description="RAG-based AI Knowledge Assistant with per-session persistent indexes",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "DocMind backend running", "storage": storage.mode()}
