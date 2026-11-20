import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

import faiss
import numpy as np

from app.config import settings
from app.core import storage

CACHE_SIZE = 20


class SessionFullError(Exception):
    pass


@dataclass
class Store:
    index: faiss.Index
    chunks: list[str]
    sources: list[str]


_cache: "OrderedDict[str, Store]" = OrderedDict()
_cache_lock = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}


def _lock_for(session_id: str) -> threading.Lock:
    with _cache_lock:
        return _session_locks.setdefault(session_id, threading.Lock())


def _cache_put(session_id: str, store: Store) -> None:
    with _cache_lock:
        _cache[session_id] = store
        _cache.move_to_end(session_id)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def load(session_id: str) -> Store | None:
    with _cache_lock:
        cached = _cache.get(session_id)
        if cached is not None:
            _cache.move_to_end(session_id)
            return cached

    index_bytes = storage.load_bytes(session_id, "index.faiss")
    meta_bytes = storage.load_bytes(session_id, "meta.json")
    if index_bytes is None or meta_bytes is None:
        return None

    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
    meta = json.loads(meta_bytes)
    if index.ntotal != len(meta["chunks"]) or len(meta["chunks"]) != len(meta["sources"]):
        return None

    store = Store(index=index, chunks=meta["chunks"], sources=meta["sources"])
    _cache_put(session_id, store)
    return store


def _persist(session_id: str, store: Store) -> None:
    index_bytes = faiss.serialize_index(store.index).tobytes()
    meta_bytes = json.dumps(
        {"chunks": store.chunks, "sources": store.sources}, ensure_ascii=False
    ).encode("utf-8")
    storage.save_bytes(session_id, "index.faiss", index_bytes)
    storage.save_bytes(session_id, "meta.json", meta_bytes)


def add_document(
    session_id: str, filename: str, chunks: list[str], embeddings: np.ndarray
) -> int:
    """Adds a document. Re-uploading the same filename replaces its old chunks."""
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]

    with _lock_for(session_id):
        existing = load(session_id)
        index = faiss.IndexFlatIP(dim)
        kept_chunks: list[str] = []
        kept_sources: list[str] = []

        if existing is not None:
            if existing.index.d != dim:
                raise ValueError("Embedding dimension changed; clear the session and re-upload.")
            keep = [i for i, s in enumerate(existing.sources) if s != filename]
            if keep:
                old_vectors = existing.index.reconstruct_n(0, existing.index.ntotal)
                index.add(np.ascontiguousarray(old_vectors[keep], dtype="float32"))
                kept_chunks = [existing.chunks[i] for i in keep]
                kept_sources = [existing.sources[i] for i in keep]

        if len(kept_chunks) + len(chunks) > settings.max_chunks_per_session:
            raise SessionFullError(
                f"Session limit of {settings.max_chunks_per_session} chunks reached."
            )

        index.add(vectors)
        store = Store(
            index=index,
            chunks=kept_chunks + chunks,
            sources=kept_sources + [filename] * len(chunks),
        )
        _persist(session_id, store)
        _cache_put(session_id, store)
        return len(store.chunks)


def search(session_id: str, query_vector: np.ndarray, top_k: int) -> list[dict]:
    store = load(session_id)
    if store is None:
        return []

    query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query)
    k = max(1, min(top_k, store.index.ntotal))
    scores, ids = store.index.search(query, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append(
            {
                "chunk": store.chunks[idx],
                "source": store.sources[idx],
                "score": round(float(score), 4),
            }
        )
    return results


def list_documents(session_id: str) -> list[dict]:
    store = load(session_id)
    if store is None:
        return []
    counts: dict[str, int] = {}
    for source in store.sources:
        counts[source] = counts.get(source, 0) + 1
    return [{"filename": name, "chunks": n} for name, n in counts.items()]


def clear(session_id: str) -> None:
    with _lock_for(session_id):
        storage.delete_session(session_id)
        with _cache_lock:
            _cache.pop(session_id, None)
