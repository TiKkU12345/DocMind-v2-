import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from app.config import settings

TIMEOUT = 20
FILES = ("index.faiss", "meta.json")


class StorageError(Exception):
    pass


def remote_enabled() -> bool:
    return bool(settings.supabase_url and settings.supabase_key)


def mode() -> str:
    return "supabase" if remote_enabled() else "local"


def _local_dir(session_id: str) -> Path:
    return Path(settings.data_dir) / "sessions" / session_id


def _headers(extra: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {settings.supabase_key}",
        "apikey": settings.supabase_key,
    }
    if extra:
        headers.update(extra)
    return headers


def _object_url(path: str) -> str:
    return f"{settings.supabase_url}/storage/v1/object/{settings.supabase_bucket}/{path}"


def save_bytes(session_id: str, name: str, data: bytes) -> None:
    if remote_enabled():
        try:
            resp = requests.post(
                _object_url(f"{session_id}/{name}"),
                data=data,
                headers=_headers(
                    {"x-upsert": "true", "Content-Type": "application/octet-stream"}
                ),
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise StorageError(f"Supabase upload failed: {exc}") from exc
        if resp.status_code >= 300:
            raise StorageError(f"Supabase upload failed ({resp.status_code}): {resp.text[:200]}")
        return

    folder = _local_dir(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.tmp"
    tmp.write_bytes(data)
    tmp.replace(folder / name)


def load_bytes(session_id: str, name: str) -> bytes | None:
    if remote_enabled():
        try:
            resp = requests.get(
                _object_url(f"{session_id}/{name}"), headers=_headers(), timeout=TIMEOUT
            )
        except requests.RequestException as exc:
            raise StorageError(f"Supabase download failed: {exc}") from exc
        if resp.status_code in (400, 404):
            return None
        if resp.status_code >= 300:
            raise StorageError(f"Supabase download failed ({resp.status_code}): {resp.text[:200]}")
        return resp.content

    path = _local_dir(session_id) / name
    return path.read_bytes() if path.exists() else None


def delete_session(session_id: str) -> None:
    if remote_enabled():
        try:
            resp = requests.delete(
                f"{settings.supabase_url}/storage/v1/object/{settings.supabase_bucket}",
                json={"prefixes": [f"{session_id}/{name}" for name in FILES]},
                headers=_headers(),
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise StorageError(f"Supabase delete failed: {exc}") from exc
        if resp.status_code >= 300:
            raise StorageError(f"Supabase delete failed ({resp.status_code}): {resp.text[:200]}")
        return

    shutil.rmtree(_local_dir(session_id), ignore_errors=True)


def _list_remote(prefix: str) -> list[dict]:
    resp = requests.post(
        f"{settings.supabase_url}/storage/v1/object/list/{settings.supabase_bucket}",
        json={"prefix": prefix, "limit": 1000, "offset": 0},
        headers=_headers(),
        timeout=TIMEOUT,
    )
    if resp.status_code >= 300:
        raise StorageError(f"Supabase list failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()


def _parse_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def cleanup_old_sessions(max_age_days: int) -> int:
    """Deletes sessions untouched for max_age_days. Returns how many were removed."""
    cutoff = time.time() - max_age_days * 86400
    removed = 0

    if remote_enabled():
        for entry in _list_remote(""):
            session_id = entry.get("name")
            if not session_id or entry.get("id") is not None:
                continue
            files = _list_remote(session_id)
            stamps = [_parse_ts(f["updated_at"]) for f in files if f.get("updated_at")]
            if stamps and max(stamps) < cutoff:
                delete_session(session_id)
                removed += 1
        return removed

    root = Path(settings.data_dir) / "sessions"
    if not root.exists():
        return 0
    for folder in root.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed
