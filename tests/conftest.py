import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import vector_store


def fake_embed(texts):
    dim = 64
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, sum(map(ord, word)) % dim] += 1.0
    return out


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    vector_store.clear_cache()
    yield
    vector_store.clear_cache()


@pytest.fixture
def client(monkeypatch):
    from app.api import query as query_api
    from app.api import upload as upload_api
    from app.core import embedding

    monkeypatch.setattr(embedding, "get_embeddings", fake_embed)
    monkeypatch.setattr(
        query_api.generator,
        "generate_answer",
        lambda q, chunks: "ANSWER from " + ", ".join(sorted({c["source"] for c in chunks})),
    )
    from app.main import app

    return TestClient(app)
