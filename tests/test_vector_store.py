from app.core import vector_store
from tests.conftest import fake_embed

SID = "session_aaaaaaaaaaaaaaaa"
OTHER = "session_bbbbbbbbbbbbbbbb"


def add(sid, name, chunks):
    return vector_store.add_document(sid, name, chunks, fake_embed(chunks))


def test_search_finds_relevant_chunk():
    add(SID, "a.txt", ["cats purr softly", "stock markets crashed today"])
    results = vector_store.search(SID, fake_embed(["cats purr"]), 1)
    assert results[0]["chunk"] == "cats purr softly"
    assert results[0]["source"] == "a.txt"


def test_persists_after_cache_cleared():
    add(SID, "a.txt", ["alpha beta gamma"])
    vector_store.clear_cache()
    assert vector_store.list_documents(SID) == [{"filename": "a.txt", "chunks": 1}]
    assert vector_store.search(SID, fake_embed(["alpha"]), 1)[0]["chunk"] == "alpha beta gamma"


def test_reupload_same_filename_replaces_chunks():
    add(SID, "a.txt", ["old text one", "old text two"])
    total = add(SID, "a.txt", ["fresh text"])
    assert total == 1
    assert vector_store.list_documents(SID) == [{"filename": "a.txt", "chunks": 1}]


def test_second_document_is_added_not_replaced():
    add(SID, "a.txt", ["first doc"])
    total = add(SID, "b.txt", ["second doc"])
    assert total == 2


def test_sessions_are_isolated():
    add(SID, "a.txt", ["secret contract terms"])
    assert vector_store.load(OTHER) is None
    assert vector_store.search(OTHER, fake_embed(["secret"]), 3) == []


def test_clear_removes_everything():
    add(SID, "a.txt", ["something"])
    vector_store.clear(SID)
    vector_store.clear_cache()
    assert vector_store.load(SID) is None


def test_session_limit(monkeypatch):
    monkeypatch.setenv("MAX_CHUNKS_PER_SESSION", "2")
    add(SID, "a.txt", ["one", "two"])
    try:
        add(SID, "b.txt", ["three"])
        raised = False
    except vector_store.SessionFullError:
        raised = True
    assert raised
