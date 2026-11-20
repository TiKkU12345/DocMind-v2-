import os
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")


def get_api_base() -> str:
    try:
        base = st.secrets["API_BASE"]
    except Exception:
        base = os.getenv("API_BASE", "http://localhost:8000")
    return base.rstrip("/")


API_BASE = get_api_base()


def get_session_id() -> str:
    sid = st.query_params.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        st.query_params["sid"] = sid
    return sid


SID = get_session_id()


def call(method: str, path: str, timeout: int = 120, **kwargs) -> requests.Response:
    headers = {"X-Session-ID": SID}
    return requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=timeout, **kwargs)


def error_text(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def wake_server() -> bool:
    if st.session_state.get("server_ready"):
        return True
    with st.spinner("Waking up the server. On the free tier this can take up to a minute..."):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=90)
            ok = resp.status_code == 200
        except requests.RequestException:
            ok = False
    st.session_state["server_ready"] = ok
    return ok


def refresh_documents() -> None:
    try:
        resp = call("GET", "/documents", timeout=30)
        st.session_state["documents"] = resp.json().get("documents", []) if resp.ok else []
    except requests.RequestException:
        st.session_state["documents"] = []


SAMPLE_NAME = "sample_service_agreement.txt"
SAMPLE_QUESTIONS = [
    "When is payment due and what happens if it is late?",
    "How can this agreement be terminated?",
    "What is the liability cap?",
    "What is the refund policy?",
]


def load_sample() -> None:
    with st.spinner("Indexing the sample agreement..."):
        try:
            resp = call("POST", "/sample")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            return
    if resp.ok:
        refresh_documents()
        st.rerun()
    else:
        st.error(error_text(resp))


st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

st.title("DocMind")
st.caption("Upload documents once and ask questions about them. Your documents stay linked to this page's URL.")

if not wake_server():
    st.error("Backend is not reachable right now. Refresh in a minute.")
    st.stop()

if "documents" not in st.session_state:
    refresh_documents()

with st.sidebar:
    st.subheader("Your documents")
    docs = st.session_state["documents"]
    if docs:
        for d in docs:
            st.write(f"{d['filename']}  ({d['chunks']} chunks)")
    else:
        st.write("Nothing indexed yet.")

    files = st.file_uploader(
        "Add PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    if files and st.button("Index selected files", type="primary"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    resp = call(
                        "POST", "/upload", files={"file": (f.name, f.getvalue())}
                    )
                except requests.RequestException as exc:
                    st.error(f"{f.name}: request failed ({exc})")
                    continue
            if resp.ok:
                st.success(f"{f.name}: {resp.json()['chunks_created']} chunks")
            else:
                st.error(f"{f.name}: {error_text(resp)}")
        st.session_state["uploader_key"] += 1
        refresh_documents()
        st.rerun()

    if docs and st.button("Delete all my documents"):
        try:
            call("DELETE", "/session", timeout=30)
        except requests.RequestException:
            pass
        st.session_state["messages"] = []
        refresh_documents()
        st.rerun()

    st.caption("Bookmark this page. The URL is your key to these documents. Data is deleted after 7 days of inactivity.")

has_sample = any(d["filename"] == SAMPLE_NAME for d in st.session_state["documents"])

if not st.session_state["documents"]:
    st.info("No documents yet. Upload your own in the sidebar, or try a ready-made one.")
    if st.button("Try a sample document", type="primary"):
        load_sample()
elif has_sample and not st.session_state["messages"]:
    st.caption("Sample agreement loaded. Try a question (the last one is deliberately not covered by the document):")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"sq_{i}"):
            st.session_state["pending_question"] = q
            st.rerun()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        for src in message.get("sources", []):
            with st.expander(f"{src['source']}  (score {src['score']})"):
                st.write(src["chunk"])

question = st.chat_input("Ask something about your documents")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = call("POST", "/query", json={"query": question, "top_k": 3})
            except requests.RequestException as exc:
                resp = None
                answer, sources = f"Request failed: {exc}", []
        if resp is not None:
            if resp.ok:
                data = resp.json()
                answer, sources = data["answer"], data["sources"]
            else:
                answer, sources = error_text(resp), []
        st.write(answer)
        for src in sources:
            with st.expander(f"{src['source']}  (score {src['score']})"):
                st.write(src["chunk"])

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
