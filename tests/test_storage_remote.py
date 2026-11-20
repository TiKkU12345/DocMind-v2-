import requests

from app.core import storage, vector_store
from tests.conftest import fake_embed

SID = "session_aaaaaaaaaaaaaaaa"


class FakeResp:
    def __init__(self, status=200, content=b"", json_data=None):
        self.status_code = status
        self.content = content
        self.text = content.decode("utf-8", errors="ignore") if content else ""
        self._json = json_data

    def json(self):
        return self._json


def install_fake_supabase(monkeypatch):
    objects = {}
    prefix = "https://proj.supabase.co/storage/v1/object/docmind-indexes/"

    def post(url, data=None, json=None, headers=None, timeout=None):
        if url.startswith(prefix):
            assert headers["x-upsert"] == "true"
            assert headers["Authorization"] == "Bearer svc-key"
            objects[url[len(prefix):]] = data
            return FakeResp(200)
        raise AssertionError(url)

    def get(url, headers=None, timeout=None):
        key = url[len(prefix):]
        return FakeResp(200, objects[key]) if key in objects else FakeResp(400, b"not found")

    def delete(url, json=None, headers=None, timeout=None):
        for path in json["prefixes"]:
            objects.pop(path, None)
        return FakeResp(200)

    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key")
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "delete", delete)
    return objects


def test_remote_roundtrip_survives_restart(monkeypatch):
    objects = install_fake_supabase(monkeypatch)
    assert storage.mode() == "supabase"

    chunks = ["cats purr softly", "stock markets crashed"]
    vector_store.add_document(SID, "a.txt", chunks, fake_embed(chunks))
    assert set(objects) == {f"{SID}/index.faiss", f"{SID}/meta.json"}

    vector_store.clear_cache()
    results = vector_store.search(SID, fake_embed(["cats"]), 1)
    assert results[0]["chunk"] == "cats purr softly"


def test_remote_missing_session_returns_none(monkeypatch):
    install_fake_supabase(monkeypatch)
    assert vector_store.load(SID) is None


def test_remote_delete(monkeypatch):
    objects = install_fake_supabase(monkeypatch)
    vector_store.add_document(SID, "a.txt", ["hello"], fake_embed(["hello"]))
    vector_store.clear(SID)
    assert objects == {}
