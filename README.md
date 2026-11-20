# DocMind v2

RAG document Q&A. FastAPI + FAISS + fastembed (all-MiniLM-L6-v2) + Groq (default model `openai/gpt-oss-120b`, override with `GROQ_MODEL`), Streamlit frontend.

What changed from v1:

- Backend runs on Render free tier instead of Railway.
- Embeddings use fastembed (ONNX) instead of sentence-transformers, so no torch and much lower RAM. Same model.
- FAISS index and chunk metadata are stored in Supabase Storage, so documents survive restarts and sleep. Users do not re-upload.
- Per-session isolation. Each browser session has its own index, identified by `sid` in the page URL.
- DOCX support. Re-uploading a file with the same name replaces its old chunks instead of duplicating them.
- Chunk metadata is JSON, not pickle. Raw uploads are never written to disk.
- "Try a sample document" button (fictional agreement in `samples/`) so visitors can test without uploading anything.
- Sessions untouched for 7 days are deleted on server startup.

## Layout

    app/            FastAPI backend
    samples/        demo document loaded by the sample button
    frontend/       Streamlit app (own requirements.txt)
    scripts/        build-time model download
    tests/          pytest suite
    render.yaml     Render blueprint

## 1. Supabase

1. Create a project (or use an existing one).
2. Storage, New bucket, name `docmind-indexes`, keep it **private**.
3. Project settings, API, copy the Project URL and the `service_role` key.
   The service_role key goes only on the backend. Never put it in the Streamlit app or in git.

## 2. Backend on Render

1. Push this folder to GitHub.
2. Render, New, Blueprint, pick the repo. It reads `render.yaml`.
3. Set the three secrets when asked: `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.
4. After deploy, open `https://<your-service>.onrender.com/health`. It should say `"storage": "supabase"`.
   If it says `local`, the Supabase env vars are missing and data will vanish on restart.

## 3. Frontend on Streamlit Cloud

1. New app, same repo, main file path `frontend/streamlit_app.py`.
2. Advanced settings, Secrets:

       API_BASE = "https://<your-service>.onrender.com"

## Run locally

    pip install -r requirements-dev.txt
    cp .env.example .env        # leave Supabase vars empty to use local disk
    python -m pytest
    uvicorn app.main:app --reload
    API_BASE=http://localhost:8000 streamlit run frontend/streamlit_app.py

## Known limits

- Free Render services sleep after inactivity. First request after sleep takes a while. The frontend shows a wake-up spinner.
- Single backend instance assumed (in-process cache and locks).
- Anyone with a session URL can read that session's documents. Treat the URL as a secret.
- No rate limiting. Someone can burn your Groq quota. Add limits before sharing widely.
- Scanned PDFs (images) are not supported, there is no OCR.
