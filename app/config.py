import os

from dotenv import load_dotenv

load_dotenv()


class _Settings:
    """Reads env vars at access time, so load_dotenv order can never bite again."""

    @property
    def groq_api_key(self) -> str:
        return os.getenv("GROQ_API_KEY", "")

    @property
    def groq_model(self) -> str:
        return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    @property
    def supabase_url(self) -> str:
        return os.getenv("SUPABASE_URL", "").rstrip("/")

    @property
    def supabase_key(self) -> str:
        return os.getenv("SUPABASE_SERVICE_KEY", "")

    @property
    def supabase_bucket(self) -> str:
        return os.getenv("SUPABASE_BUCKET", "docmind-indexes")

    @property
    def data_dir(self) -> str:
        return os.getenv("DATA_DIR", "data")

    @property
    def embed_cache_dir(self) -> str:
        return os.getenv("EMBED_CACHE_DIR", "models_cache")

    @property
    def max_upload_mb(self) -> int:
        return int(os.getenv("MAX_UPLOAD_MB", "10"))

    @property
    def max_chunks_per_session(self) -> int:
        return int(os.getenv("MAX_CHUNKS_PER_SESSION", "3000"))

    @property
    def session_ttl_days(self) -> int:
        return int(os.getenv("SESSION_TTL_DAYS", "7"))

    @property
    def allowed_origins(self) -> list[str]:
        raw = os.getenv("ALLOWED_ORIGINS", "*")
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = _Settings()
