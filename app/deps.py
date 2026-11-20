import re

from fastapi import Header, HTTPException

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def get_session_id(x_session_id: str = Header(default="")) -> str:
    if not SESSION_RE.match(x_session_id):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Session-ID header (16-64 chars, letters, digits, _ or -).",
        )
    return x_session_id
