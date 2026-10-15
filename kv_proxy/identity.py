"""Client identity resolution: (frontend, session_id) from each request."""

from __future__ import annotations

import hashlib
import json
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


def _msg_seed(msg: dict) -> str:
    """Canonical string for one message (role + content), for hashing."""
    content = msg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return f"{msg.get('role', '')}:{content}"


def _opening_seed(body: dict) -> str:
    """The conversation's fixed opening: leading system prompt + first user turn.

    Both are constant for the life of a chat (later turns only append), so a
    hash of them is a stable per-chat identifier.
    """
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for msg in messages:  # first system message, if any
            if isinstance(msg, dict) and msg.get("role") == "system":
                parts.append(_msg_seed(msg))
                break
        for msg in messages:  # first non-system message (the defining user turn)
            if isinstance(msg, dict) and msg.get("role") != "system":
                parts.append(_msg_seed(msg))
                break
        if parts:
            return "\x00".join(parts)
    # legacy /completion: `prompt` may be a string or list of strings. The
    # full prompt grows every turn, so only the part before the first
    # assistant reply is stable for the life of the chat.
    prompt = body.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        cut = prompt.find("Assistant:")
        return prompt[: cut if cut > 0 else 512]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str) and prompt[0].strip():
        return prompt[0][:512]
    return ""


def dump_opening(body: dict, path: str) -> None:
    """Append the full opening messages to a debug file (KV_PROXY_DEBUG_OPENING)."""
    import datetime

    messages = body.get("messages")
    if not (isinstance(messages, list) and messages):
        return
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "system": None,
        "first": None,
    }
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if rec["system"] is None and msg.get("role") == "system":
            rec["system"] = msg.get("content")
        if rec["first"] is None and msg.get("role") != "system":
            rec["first"] = msg.get("content")
        if rec["system"] is not None and rec["first"] is not None:
            break
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def seed_components(body: dict) -> str:
    """Compact per-component digest of the opening seed (for debugging).

    Shows which part of the opening (system prompt vs first user turn) a
    session hash is built from, so per-request logs can reveal what changes
    between turns of the same chat.
    """
    messages = body.get("messages")
    if not (isinstance(messages, list) and messages):
        return ""
    sys_hash = ""
    first_hash = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system" and not sys_hash:
            sys_hash = hashlib.sha1(_msg_seed(msg).encode("utf-8")).hexdigest()[:12]
        elif msg.get("role") != "system" and not first_hash:
            first_hash = hashlib.sha1(_msg_seed(msg).encode("utf-8")).hexdigest()[:12]
        if sys_hash and first_hash:
            break
    return f"sys={sys_hash or '-'} first={first_hash or '-'} n={len(messages)}"


def _content_session(body: dict) -> str:
    """Derive a stable per-chat session id from the conversation's opening.

    Finer-grained than a `user` id (which lumps all of one user's chats into a
    single blob that keeps overwriting itself) and works with zero client
    configuration. Chats that genuinely share the same opening share the same
    KV prefix anyway, so sharing a cache entry there is harmless.
    """
    seed = _opening_seed(body)
    if not seed:
        return ""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"c-{digest}"


def _session_from_body(body: dict) -> str:
    """Extract a session id from the request body, if the client left one.

    First non-empty wins, most specific first:

      body["session"]            - explicit, e.g. SillyTavern "chat" field
                                   renamed via its advanced "Body" editor
      body["id"]                 - OpenAI "completion id" field
      body["messages"][*]["metadata"] / ["id"]  - per-message hints
      content-derived            - stable per-chat hash of the opening turn
                                   (automatic, zero-config per-chat isolation)
      body["user"]               - legacy per-user grouping (coarser)
    """
    if isinstance(body.get("session"), (str, int)) and str(body["session"]).strip():
        return str(body["session"]).strip()
    if isinstance(body.get("id"), (str, int)) and str(body["id"]).strip():
        return str(body["id"]).strip()

    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            for field in ("metadata", "id"):
                value = msg.get(field)
                if isinstance(value, (str, int)) and str(value).strip():
                    return str(value).strip()

    # automatic per-chat isolation: hash the conversation's opening
    content = _content_session(body)
    if content:
        return content

    user = body.get("user")
    if user:
        return str(user)
    return ""


def resolve_identity(
    headers: dict[str, str],
    body: dict | None,
    keys: dict[str, str],
) -> Identity:
    """Resolve (frontend, session) for a request.

    Frontend: the first configured API key present in Authorization wins
    (Bearer or raw); otherwise "default".
    Session: X-KV-Session header, else body hints (session / id / message
    metadata), else a per-chat hash of the conversation's opening, else the
    body "user" field, else "default".
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
        session_raw = _session_from_body(body)
    session = sanitize(session_raw)

    return Identity(frontend=frontend, session=session)


def model_of(body: dict | None) -> str:
    """Extract the model id from an OpenAI-style request body."""
    if isinstance(body, dict):
        m = body.get("model")
        if m:
            return str(m)
    return ""
