"""Configuration loading and validation for kv-proxy."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)?\s*$", re.IGNORECASE)
_UNIT_MULTIPLIERS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


def parse_size(value: str | int) -> int:
    """Parse a human size like '20GB' or a plain byte count into bytes."""
    if isinstance(value, int):
        return value
    m = _SIZE_RE.match(str(value))
    if not m:
        raise ValueError(f"invalid size: {value!r} (expected e.g. '20GB')")
    number, unit = m.groups()
    return int(float(number) * _UNIT_MULTIPLIERS[(unit or "B").upper()])


@dataclass
class Config:
    listen: str = "0.0.0.0:8080"
    llama_swap_url: str = "http://127.0.0.1:8080"
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".kv-cache")
    slot_save_path: Path = field(
        default_factory=lambda: Path.home() / ".kv-cache" / "slots"
    )
    max_cache_size: int = 20 * 1024**3
    wait_for_slot_timeout: float = 120.0
    model_poll_interval: float = 2.0
    save_after_request: bool = True
    direct_upstream: dict[str, str] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)

    @property
    def host(self) -> str:
        return self.listen.rsplit(":", 1)[0] or "0.0.0.0"

    @property
    def port(self) -> int:
        try:
            return int(self.listen.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return 8080

    def validate(self) -> None:
        """Validate config and ensure directories exist."""
        if not self.llama_swap_url.startswith(("http://", "https://")):
            raise ValueError(
                f"llama_swap_url must be an http(s) URL, got {self.llama_swap_url!r}"
            )
        self.cache_dir = _validate_path(self.cache_dir)
        self.slot_save_path = _validate_path(self.slot_save_path)
        if self.slot_save_path == self.cache_dir:
            raise ValueError(
                "slot_save_path must be a subdirectory distinct from cache_dir "
                "(llama.cpp writes staging files there; kv-proxy moves them out)"
            )
        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def load_config(path: str | Path | None = None) -> Config:
    """Load config from TOML file, with environment variable overrides.

    Lookup order for the file: explicit path, $KV_PROXY_CONFIG, ./config.toml.
    A missing file is fine — defaults apply (useful for tests).
    """
    data: dict = {}
    candidates = [path] if path else []
    candidates.append(_env("KV_PROXY_CONFIG"))
    candidates.append(Path("config.toml"))

    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.is_file():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            break

    cfg = Config()

    if v := data.get("listen") or _env("KV_PROXY_LISTEN"):
        cfg.listen = str(v)
    if v := data.get("llama_swap_url") or _env("KV_PROXY_LLAMA_SWAP_URL"):
        cfg.llama_swap_url = str(v).rstrip("/")
    if v := data.get("cache_dir") or _env("KV_PROXY_CACHE_DIR"):
        cfg.cache_dir = Path(str(v)).expanduser()
    if v := data.get("slot_save_path"):
        cfg.slot_save_path = Path(str(v)).expanduser()
    else:
        # default relative to cache dir so the two never drift apart
        cfg.slot_save_path = cfg.cache_dir / "slots"
    if v := data.get("max_cache_size") or _env("KV_PROXY_MAX_CACHE_SIZE"):
        cfg.max_cache_size = parse_size(str(v))
    if v := data.get("wait_for_slot_timeout"):
        cfg.wait_for_slot_timeout = float(v)
    if v := data.get("model_poll_interval"):
        cfg.model_poll_interval = float(v)
    if v := data.get("save_after_request"):
        cfg.save_after_request = bool(v)

    du = data.get("direct_upstream") or {}
    if isinstance(du, dict):
        cfg.direct_upstream = {str(k): str(val) for k, val in du.items()}

    keys = data.get("keys") or {}
    if isinstance(keys, dict):
        cfg.keys = {str(k): str(val) for k, val in keys.items() if str(val)}

    cfg.validate()
    return cfg


def _validate_path(p: Path) -> Path:
    p = p.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.mkdir(parents=True, exist_ok=True)
    return p
