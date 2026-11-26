import io

from docx import Document

SID = "session_aaaaaaaaaaaaaaaa"
OTHER = "session_bbbbbbbbbbbbbbbb"


def hdr(sid=SID):
    return {"X-Session-ID": sid}


def upload(client, name, data, sid=SID):
    return client.post("/upload", files={"file": (name, data)}, headers=hdr(sid))


def make_docx(text):
    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["storage"] == "local"


def test_missing_session_header_rejected(client):
    assert client.get("/documents").status_code == 400


def test_bad_session_id_rejected(client):
    resp = client.get("/documents", headers={"X-Session-ID": "../../etc/passwd"})
    assert resp.status_code == 400


def test_upload_txt_and_query(client):
    resp = upload(client, "notes.txt", b"The tenant must pay rent on the first of every month.")
    assert resp.status_code == 200
    assert resp.json()["status"] == "indexed"

    q = client.post("/query", json={"query": "when is rent due"}, headers=hdr())
    assert q.status_code == 200
    assert q.json()["answer"] == "ANSWER from notes.txt"
    assert q.json()["sources"][0]["source"] == "notes.txt"


def test_upload_docx(client):
    resp = upload(client, "Contract.DOCX", make_docx("Termination requires thirty days notice."))
    assert resp.status_code == 200
    docs = client.get("/documents", headers=hdr()).json()["documents"]
    assert docs[0]["filename"] == "Contract.DOCX"


def test_uppercase_extension_accepted(client):
    assert upload(client, "NOTES.TXT", b"some text here").status_code == 200


def test_bad_extension_rejected(client):
    assert upload(client, "malware.exe", b"x").status_code == 400
    assert upload(client, "old.doc", b"x").status_code == 400


def test_path_traversal_filename_is_flattened(client):
    resp = upload(client, "../../etc/passwd.txt", b"hello world text")
    assert resp.status_code == 200
    assert resp.json()["filename"] == "passwd.txt"


def test_empty_file_rejected(client):
    assert upload(client, "empty.txt", b"   ").status_code == 400


def test_corrupted_docx_rejected(client):
    assert upload(client, "broken.docx", b"not a zip").status_code == 400


def test_oversized_file_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    resp = upload(client, "big.txt", b"a" * (1024 * 1024 + 10))
    assert resp.status_code == 413


def test_query_before_upload_is_404(client):
    resp = client.post("/query", json={"query": "anything"}, headers=hdr())
    assert resp.status_code == 404


def test_sessions_do_not_leak(client):
    upload(client, "private.txt", b"confidential salary information")
    resp = client.post("/query", json={"query": "salary"}, headers=hdr(OTHER))
    assert resp.status_code == 404
    assert client.get("/documents", headers=hdr(OTHER)).json()["documents"] == []


def test_delete_session(client):
    upload(client, "a.txt", b"some content to index")
    assert client.delete("/session", headers=hdr()).status_code == 200
    assert client.get("/documents", headers=hdr()).json()["documents"] == []


def test_query_validation(client):
    upload(client, "a.txt", b"some content to index")
    assert client.post("/query", json={"query": ""}, headers=hdr()).status_code == 422
    assert client.post("/query", json={"query": "x", "top_k": 99}, headers=hdr()).status_code == 422


def test_sample_loads_and_is_queryable(client):
    resp = client.post("/sample", headers=hdr())
    assert resp.status_code == 200
    assert resp.json()["filename"] == "sample_service_agreement.txt"
    assert resp.json()["chunks_created"] >= 4

    q = client.post("/query", json={"query": "payment due"}, headers=hdr())
    assert q.status_code == 200
    assert q.json()["sources"][0]["source"] == "sample_service_agreement.txt"


def test_sample_twice_does_not_duplicate(client):
    first = client.post("/sample", headers=hdr()).json()["total_chunks"]
    second = client.post("/sample", headers=hdr()).json()["total_chunks"]
    assert first == second


def test_sample_requires_session(client):
    assert client.post("/sample").status_code == 400


def _verifier_payload(supported=True):
    import json

    return json.dumps(
        {"claims": [{"claim": "c", "supported": supported, "chunks": [1] if supported else []}]}
    )


def test_query_without_verify_has_no_verification(client):
    upload(client, "a.txt", b"Payment is due within 30 days of invoice.")
    body = client.post("/query", json={"query": "payment"}, headers=hdr()).json()
    assert body["verification"] is None
    assert body["abstained"] is False


def test_query_with_verify_returns_claims(client, monkeypatch):
    from app.core import generator

    upload(client, "a.txt", b"Payment is due within 30 days of invoice.")
    monkeypatch.setattr(generator, "complete", lambda *a, **k: _verifier_payload(True))
    body = client.post("/query", json={"query": "payment", "verify": True}, headers=hdr()).json()
    assert body["verification"]["status"] == "ok"
    assert body["verification"]["faithfulness"] == 1.0


def test_verifier_failure_does_not_break_query(client, monkeypatch):
    from app.core import generator

    upload(client, "a.txt", b"Payment is due within 30 days of invoice.")
    monkeypatch.setattr(generator, "complete", lambda *a, **k: "garbage")
    resp = client.post("/query", json={"query": "payment", "verify": True}, headers=hdr())
    assert resp.status_code == 200
    assert resp.json()["answer"].startswith("ANSWER")
    assert resp.json()["verification"]["status"] == "failed"


def test_abstention_skips_verification(client, monkeypatch):
    from app.api import query as query_api

    upload(client, "a.txt", b"Payment is due within 30 days of invoice.")
    monkeypatch.setattr(
        query_api.generator,
        "generate_answer",
        lambda q, c: "I don't have enough information to answer this.",
    )
    called = []
    monkeypatch.setattr(query_api.verifier, "verify_answer", lambda *a: called.append(1))
    body = client.post("/query", json={"query": "refund", "verify": True}, headers=hdr()).json()
    assert body["abstained"] is True
    assert body["verification"] is None
    assert called == []
