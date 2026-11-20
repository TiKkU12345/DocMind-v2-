import threading

import faiss
import numpy as np

from app.config import settings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                _model = TextEmbedding(MODEL_NAME, cache_dir=settings.embed_cache_dir)
    return _model


def warmup() -> None:
    list(_get_model().embed(["warmup"]))


def get_embeddings(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(_get_model().embed(texts, batch_size=32)), dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors
