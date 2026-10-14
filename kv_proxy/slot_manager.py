"""Per-model slot state machine.

Invariants:
  * at most one session is resident per slot (per model)
  * a request for session S on model M uses the slot S already occupies, or a
    free slot, or an idle slot holding another session (which gets saved+evicted),
    or waits until wait_for_slot_timeout expires

Model swap awareness: when llama-swap loads/unloads a model (manual swap) or a
session switches models, resident state for the affected model is dropped —
the in-memory KV is gone, but disk blobs survive and get restored on demand.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .cache_store import CacheStore
from .llama_client import LlamaClient, LlamaError

log = logging.getLogger("kv_proxy.slots")


@dataclass
class SlotInfo:
    id: int
    busy: bool  # is_processing
    resident: str | None = None  # (frontend, session) key string, or None


@dataclass
class Allocation:
    slot_id: int
    restored: bool = False
    evicted: tuple[str, str] | None = None  # (frontend, session) pushed out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class SlotManager:
    def __init__(self, client: LlamaClient, store: CacheStore,
                 slot_save_path, wait_timeout: float = 120.0,
                 poll_interval: float = 2.0):
        self.client = client
        self.store = store
        self.slot_save_path = slot_save_path
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self._lock = asyncio.Lock()  # serializes allocation decisions per model
        self._resident: dict[str, dict[int, tuple[str, str]]] = {}  # model -> slot -> (fe, sess)
        self._session_slot: dict[tuple[str, str, str], int] = {}  # (fe, sess, model) -> slot
        self._watcher_task: asyncio.Task | None = None

    # ------------------------------------------------------------- lifecycle

    def ensure_watcher(self) -> None:
        """Idempotent start of the /running poll loop on the active event loop.

        Call from request handlers (a loop is guaranteed running there).
        A dead task is restarted; a healthy one is left alone.
        """
        if self.poll_interval <= 0:
            return
        if self._watcher_task is not None and not self._watcher_task.done():
            return
        self._watcher_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watcher_task = None

    async def _watch_loop(self) -> None:
        """Poll llama-swap /running; drop resident state for vanished models."""
        last_seen: set[str] | None = None
        while True:
            try:
                running = set(await self.client.running_models())
                if last_seen is not None and running != last_seen:
                    gone = last_seen - running
                    added = running - last_seen
                    for model in gone:
                        self.invalidate_model(model, reason="unloaded")
                    for model in added:
                        log.info("model %s loaded (manual swap); caches will restore on demand", model)
                last_seen = running
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - watcher must never die
                log.warning("running-models poll failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    def invalidate_model(self, model: str, reason: str = "") -> None:
        """Drop all resident state for a model (its in-memory KV is gone)."""
        self._resident.pop(model, None)
        for key in [k for k in self._session_slot if k[2] == model]:
            del self._session_slot[key]
        if reason:
            log.info("invalidated resident state for model %s (%s)", model, reason)

    # ------------------------------------------------------------ slot state

    async def refresh(self, model: str) -> dict[int, SlotInfo]:
        """Read live slot states from llama-server and merge with our residency map."""
        slots: dict[int, SlotInfo] = {}
        try:
            raw = await self.client.get_slots(model)
        except LlamaError as exc:
            log.warning("GET /slots for %s failed: %s", model, exc)
            return slots
        resident = self._resident.setdefault(model, {})
        for s in raw:
            sid = int(s.get("id", -1))
            if sid < 0:
                continue
            busy = bool(s.get("is_processing"))
            owner = resident.get(sid)
            # busy = server is actively processing. A slot that merely holds a
            # resident session's KV is still evictable (we save it first).
            slots[sid] = SlotInfo(
                id=sid,
                busy=busy,
                resident=f"{owner[0]}/{owner[1]}" if owner else None,
            )
        return slots

    def _pick_free(self, slots: dict[int, SlotInfo]) -> int | None:
        for s in sorted(slots.values(), key=lambda x: x.id):
            if not s.busy and s.resident is None:
                return s.id
        return None

    def _pick_evictable(self, slots: dict[int, SlotInfo], exclude: tuple[str, str]) -> int | None:
        """An idle slot holding a *different* session that we can save+evict.

        A slot is evictable when the server reports it idle (no active task).
        It may still hold our tracked residency — saving first preserves that KV.
        """
        for s in sorted(slots.values(), key=lambda x: x.id):
            if s.busy:
                continue
            if s.resident is None or s.resident == f"{exclude[0]}/{exclude[1]}":
                continue
            return s.id
        return None

    # ------------------------------------------------------------- allocation

    async def allocate(
        self,
        model: str,
        frontend: str,
        session: str,
        *,
        must_restore: bool = True,
    ) -> Allocation:
        """Pick (and prepare) a slot for this session on this model.

        If the session has a saved blob and must_restore is set, the chosen
        slot is restored before use. Raises TimeoutError if no slot frees up
        within wait_timeout.
        """
        key = (frontend, session, model)
        t_start = time.monotonic()

        async with self._lock:
            while True:
                slots = await self.refresh(model)
                resident = self._resident.setdefault(model, {})

                # 1. reuse our own slot whenever it exists (busy or idle): the
                #    session owns its KV there; llama.cpp will process the new
                #    prompt on top of it (or wait for the task to finish).
                my_slot = self._session_slot.get(key)
                if my_slot is not None and my_slot in slots:
                    if must_restore and self.store.get(frontend, session, model):
                        await self._restore_into(model, my_slot, frontend, session)
                    return self._claim(model, key, my_slot, restored=must_restore)

                # 2. any idle slot (unowned preferred; a resident session's KV
                #    is only evicted when there is no other option — save first).
                free = self._pick_free(slots)
                if free is not None:
                    if must_restore and self.store.get(frontend, session, model):
                        await self._restore_into(model, free, frontend, session)
                    return self._claim(model, key, free, restored=must_restore)

                # 3. no free slot: evict an idle slot holding another session
                victim = self._pick_evictable(slots, (frontend, session))
                if victim is not None:
                    owner_str = slots[victim].resident or ""
                    owner = owner_str.split("/", 1) if owner_str else ("", "")
                    if owner[0]:
                        await self._save_and_release(model, victim, owner[0], owner[1])
                    if must_restore and self.store.get(frontend, session, model):
                        await self._restore_into(model, victim, frontend, session)
                    return self._claim(model, key, victim,
                                       restored=must_restore,
                                       evicted=(owner[0], owner[1]) if owner[0] else None)

                # 4. all busy: wait
                elapsed = time.monotonic() - t_start
                if self.wait_timeout > 0 and elapsed < self.wait_timeout:
                    await asyncio.sleep(min(0.5, max(0.05, self.wait_timeout - elapsed)))
                    continue
                raise TimeoutError(
                    f"no free slot for {frontend}/{session} on {model} "
                    f"after {elapsed:.1f}s (wait_for_slot_timeout={self.wait_timeout})"
                )

    def _claim(self, model: str, key: tuple[str, str, str], slot_id: int,
               restored: bool = False, evicted: tuple[str, str] | None = None) -> Allocation:
        self._resident.setdefault(model, {})[slot_id] = (key[0], key[1])
        self._session_slot[key] = slot_id
        log.debug("allocated slot %d for %s on %s (restored=%s evicted=%s)",
                  slot_id, key, model, restored, evicted)
        return Allocation(slot_id=slot_id, restored=restored, evicted=evicted)

    def release(self, model: str, frontend: str, session: str) -> None:
        """Mark the session's slot as no longer resident (KV still in memory)."""
        key = (frontend, session, model)
        slot = self._session_slot.pop(key, None)
        if slot is not None:
            self._resident.get(model, {}).pop(slot, None)

    def forget(self, model: str, frontend: str, session: str) -> None:
        """Fully drop tracking for a session (e.g. after its cache is evicted)."""
        key = (frontend, session, model)
        slot = self._session_slot.pop(key, None)
        if slot is not None:
            self._resident.get(model, {}).pop(slot, None)

    # ------------------------------------------------------- save / restore

    async def _restore_into(self, model: str, slot_id: int,
                            frontend: str, session: str) -> bool:
        entry = self.store.get(frontend, session, model)
        if entry is None or not (entry.path / "state.bin").exists():
            return False
        filename = f"kv-{frontend}-{session}.bin"
        # the blob must be inside llama.cpp's --slot-save-path to be restorable
        staged = self.slot_save_path / filename
        try:
            import shutil
            if not staged.exists():
                shutil.copy2(entry.path / "state.bin", staged)
            result = await self.client.restore_slot(model, slot_id, filename)
            self.store.touch(frontend, session, model)
            log.info("restored %s/%s into slot %d of %s (%s tokens)",
                     frontend, session, slot_id, model, result.get("n_restored", "?"))
            return True
        except LlamaError as exc:
            log.warning("restore into slot %d failed: %s", slot_id, exc)
            return False

    async def _save_and_release(self, model: str, slot_id: int,
                                frontend: str, session: str) -> None:
        """Save a victim slot's KV to disk, then release the slot."""
        filename = f"kv-{frontend}-{session}.bin"
        try:
            result = await self.client.save_slot(model, slot_id, filename)
            staged = self.slot_save_path / filename
            if staged.exists():
                self.store.finalize(frontend, session, model, staged,
                                    n_saved=int(result.get("n_saved", 0)))
            else:
                log.warning("save for %s/%s reported success but file missing", frontend, session)
        except LlamaError as exc:
            log.warning("save of slot %d (%s/%s) failed: %s — discarding KV",
                        slot_id, frontend, session, exc)
        self.release(model, frontend, session)

    async def save_session(self, model: str, slot_id: int,
                           frontend: str, session: str) -> bool:
        """Save a session's current slot KV to disk (background, best-effort)."""
        filename = f"kv-{frontend}-{session}.bin"
        try:
            result = await self.client.save_slot(model, slot_id, filename)
            staged = self.slot_save_path / filename
            if staged.exists():
                self.store.finalize(frontend, session, model, staged,
                                    n_saved=int(result.get("n_saved", 0)))
                return True
            log.warning("save for %s/%s reported success but file missing at %s",
                        frontend, session, staged)
        except LlamaError as exc:
            log.warning("background save of %s/%s failed: %s", frontend, session, exc)
        return False

    # ------------------------------------------------------------- reporting

    def status(self) -> dict:
        return {
            "resident": {
                model: {str(slot): f"{fe}/{sess}" for slot, (fe, sess) in slots.items()}
                for model, slots in self._resident.items()
            },
            "session_slots": {
                f"{fe}/{sess} ({model})": slot
                for (fe, sess, model), slot in self._session_slot.items()
            },
        }
