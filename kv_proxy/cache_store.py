"""On-disk KV cache store: ~/.kv-cache/<frontend>/<session>/<model>/state.bin.

Layout per cached session+model:
    state.bin   - the slot KV blob (moved out of llama.cpp's --slot-save-path)
    .meta.json  - {"size": int, "last_access": float, "n_saved": int}

Eviction is global LRU across all frontends/sessions/models, capped at
config.max_cache_size bytes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("kv_proxy.cache")

META_NAME = ".meta.json"


@dataclass
class CacheEntry:
    frontend: str
    session: str
    model: str
    path: Path  # dir containing state.bin
    size: int
    last_access: float
    n_saved: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.frontend, self.session, self.model)


class CacheStore:
    def __init__(self, cache_dir: Path, max_size: int):
        self.cache_dir = Path(cache_dir)
        self.max_size = int(max_size)
        self._entries: dict[tuple[str, str, str], CacheEntry] = {}
        self._scan()

    # ------------------------------------------------------------------ scan

    def _scan(self) -> None:
        """Rebuild the in-memory index from disk (survives proxy restarts)."""
        self._entries.clear()
        if not self.cache_dir.is_dir():
            return
        for frontend in sorted(self.cache_dir.iterdir()):
            if not frontend.is_dir():
                continue
            for session in sorted(frontend.iterdir()):
                if not session.is_dir():
                    continue
                for model in sorted(session.iterdir()):
                    if not model.is_dir():
                        continue
                    self._add_entry(CacheEntry(
                        frontend=frontend.name,
                        session=session.name,
                        model=model.name,
                        path=model,
                        size=self._size_of(model),
                        last_access=self._meta_last_access(model),
                        n_saved=self._meta_n_saved(model),
                    ))

    def _add_entry(self, entry: CacheEntry) -> None:
        if entry.path.is_dir():
            self._entries[entry.key] = entry

    @staticmethod
    def _size_of(model_dir: Path) -> int:
        blob = model_dir / "state.bin"
        return blob.stat().st_size if blob.exists() else 0

    @staticmethod
    def _meta_last_access(model_dir: Path) -> float:
        meta = model_dir / META_NAME
        if meta.exists():
            try:
                return float(json.loads(meta.read_text()).get("last_access", 0.0))
            except (json.JSONDecodeError, ValueError):
                pass
        blob = model_dir / "state.bin"
        return blob.stat().st_mtime if blob.exists() else 0.0

    @staticmethod
    def _meta_n_saved(model_dir: Path) -> int:
        meta = model_dir / META_NAME
        if meta.exists():
            try:
                return int(json.loads(meta.read_text()).get("n_saved", 0))
            except (json.JSONDecodeError, ValueError):
                pass
        return 0

    # ----------------------------------------------------------------- query

    @property
    def total_size(self) -> int:
        return sum(e.size for e in self._entries.values())

    def get(self, frontend: str, session: str, model: str) -> CacheEntry | None:
        return self._entries.get((frontend, session, model))

    def list(self, frontend: str | None = None) -> list[CacheEntry]:
        entries = [e for e in self._entries.values()
                   if frontend is None or e.frontend == frontend]
        return sorted(entries, key=lambda e: -e.last_access)

    # ------------------------------------------------------------------ save

    def stage(self, frontend: str, session: str, model: str) -> Path:
        """Prepare the destination dir for a new blob."""
        d = self.cache_dir / frontend / session / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def finalize(
        self,
        frontend: str,
        session: str,
        model: str,
        staged_file: Path,
        n_saved: int = 0,
    ) -> CacheEntry | None:
        """Atomically move a freshly saved blob into the store.

        Returns the updated entry, or None if the staged file is missing.
        """
        d = self.stage(frontend, session, model)
        blob = d / "state.bin"
        tmp = blob.with_suffix(f".tmp-{os.getpid()}")
        try:
            shutil.move(str(staged_file), str(tmp))
            os.replace(tmp, blob)
        except OSError as exc:
            log.warning("finalize %s/%s/%s failed: %s", frontend, session, model, exc)
            tmp.unlink(missing_ok=True)
            return None

        size = blob.stat().st_size
        now = time.time()
        entry = self._entries.get((frontend, session, model))
        if entry is not None:
            entry.size = size
            entry.last_access = now
            entry.n_saved = n_saved or entry.n_saved
        else:
            entry = CacheEntry(frontend, session, model, d, size, now, n_saved)
            self._entries[(frontend, session, model)] = entry

        self._write_meta(entry)
        log.info(
            "cached %s/%s/%s (%d bytes, n_saved=%d)",
            frontend, session, model, size, entry.n_saved,
        )
        return entry

    def _write_meta(self, entry: CacheEntry) -> None:
        meta = entry.path / META_NAME
        try:
            meta.write_text(json.dumps({
                "size": entry.size,
                "last_access": entry.last_access,
                "n_saved": entry.n_saved,
            }))
        except OSError as exc:  # meta is best-effort; blob is the source of truth
            log.warning("writing meta for %s failed: %s", entry.key, exc)

    def touch(self, frontend: str, session: str, model: str) -> None:
        """Mark a cache entry as used (restored from it)."""
        entry = self._entries.get((frontend, session, model))
        if entry is not None:
            entry.last_access = time.time()
            self._write_meta(entry)

    # ------------------------------------------------------------------ evict

    def delete(self, frontend: str, session: str | None = None,
               model: str | None = None) -> int:
        """Delete cache entries. Returns number of bytes freed."""
        freed = 0
        for key in [k for k in self._entries
                    if k[0] == frontend
                    and (session is None or k[1] == session)
                    and (model is None or k[2] == model)]:
            freed += self._delete_entry(self._entries[key])
        return freed

    def _delete_entry(self, entry: CacheEntry) -> int:
        try:
            shutil.rmtree(entry.path)
        except OSError as exc:
            log.warning("evict %s failed: %s", entry.key, exc)
        self._entries.pop(entry.key, None)
        # prune now-empty session/frontend dirs
        parent = entry.path.parent
        for d in (parent, parent.parent):
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                break
        size = entry.size
        log.info("evicted %s (%d bytes)", entry.key, size)
        return size

    def enforce_max(self) -> int:
        """Evict oldest entries until total size is under the cap. Returns bytes freed."""
        freed = 0
        while self.total_size > self.max_size and self._entries:
            oldest = min(self._entries.values(), key=lambda e: e.last_access)
            freed += self._delete_entry(oldest)
        return freed
