"""Client identity resolution: (frontend, session_id) from each request."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Custom header that lets any frontend pin its own session id explicitly.
SESSION_HEADER = "x-kv-session"

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def sanitize(name: str, fallback: str = "default") -> str:
    """Make a frontend/session/model name safe to use as a directory component."""
    name = (name or "").strip()
    if not name:
        return fallback
    if _SAFE_NAME_RE.match(name):
        return name
    # keep it short and filesystem-safe; hash the tail so distinct names stay distinct
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:32]
    import hashlib

    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{cleaned}-{digest}" if cleaned else fallback


@dataclass(frozen=True)
class Identity:
    frontend: str
    session: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.frontend, self.session)


def resolve_identity(
    headers: dict[str, str],
    body: dict | None,
    keys: dict[str, str],
) -> Identity:
    """Resolve (frontend, session) for a request.

    Frontend: the first configured API key present in Authorization wins
    (Bearer or raw); otherwise "default".
    Session: X-KV-Session header, else body "user" field, else "default".
    """
    frontend = "default"
    auth = headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else auth.strip()
    if token:
        for name, key in keys.items():
            if token == key:
                frontend = sanitize(name)
                break

    session_raw = headers.get(SESSION_HEADER, "")
    if not session_raw and isinstance(body, dict):
        user = body.get("user")
        session_raw = str(user) if user else ""
    session = sanitize(session_raw)

    return Identity(frontend=frontend, session=session)


def model_of(body: dict | None) -> str:
    """Extract the model id from an OpenAI-style request body."""
    if isinstance(body, dict):
        m = body.get("model")
        if m:
            return str(m)
    return ""
