"""Slot manager: allocation matrix, eviction with save, model-swap invalidation."""

import asyncio

import pytest

from kv_proxy.cache_store import CacheStore
from kv_proxy.llama_client import LlamaClient
from kv_proxy.slot_manager import SlotManager


def make(tmp_path, fake, wait_timeout=1.0):
    store = CacheStore(tmp_path / "cache", max_size=1024**3)
    import httpx

    client = LlamaClient("http://swap", direct_upstream={}, timeout=5.0)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake.handle), base_url="http://swap"
    )
    mgr = SlotManager(client, store, slot_save_path=fake.save_dir, wait_timeout=wait_timeout)
    return mgr


async def close_client(mgr):
    await mgr.client.aclose()


async def test_allocate_free_slot(tmp_path, fake):
    fake.load_model("m1", n_slots=2)
    mgr = make(tmp_path, fake)
    try:
        alloc = await mgr.allocate("m1", "fe", "s1")
        assert alloc.slot_id in (0, 1)
        # second session gets a different slot
        alloc2 = await mgr.allocate("m1", "fe", "s2")
        assert alloc2.slot_id != alloc.slot_id
    finally:
        await close_client(mgr)


async def test_same_session_reuses_slot(tmp_path, fake):
    fake.load_model("m1", n_slots=2)
    mgr = make(tmp_path, fake)
    try:
        a1 = await mgr.allocate("m1", "fe", "s1")
        a2 = await mgr.allocate("m1", "fe", "s1")
        assert a1.slot_id == a2.slot_id
    finally:
        await close_client(mgr)


async def test_evict_idle_slot_saves_first(tmp_path, fake):
    fake.load_model("m1", n_slots=1)
    mgr = make(tmp_path, fake)
    try:
        a1 = await mgr.allocate("m1", "fe", "s1")
        # simulate s1 having processed a prompt (KV in the slot)
        fake.slots["m1"][a1.slot_id]["kv"] = 50
        # NOTE: do NOT release s1 — it stays resident, so s2 must save+evict it

        # new session must evict s1's slot; save should run first
        a2 = await mgr.allocate("m1", "fe", "s2")
        assert a2.slot_id == a1.slot_id
        assert a2.evicted == ("fe", "s1")
        # saved blob exists in the store
        entry = mgr.store.get("fe", "s1", "m1")
        assert entry is not None and entry.size == 100  # b"KV"*50
    finally:
        await close_client(mgr)


async def test_restore_on_reuse_after_eviction(tmp_path, fake):
    fake.load_model("m1", n_slots=1)
    mgr = make(tmp_path, fake)
    try:
        a1 = await mgr.allocate("m1", "fe", "s1")
        fake.slots["m1"][a1.slot_id]["kv"] = 30
        # s1 stays resident; s2's allocation must save+evict it

        # evict s1 (saves its KV)
        a2 = await mgr.allocate("m1", "fe", "s2")
        assert mgr.store.get("fe", "s1", "m1") is not None

        # now s1 comes back: should restore into the (evicted) slot
        a3 = await mgr.allocate("m1", "fe", "s1")
        assert a3.restored is True
        # slot KV restored from disk
        assert fake.slots["m1"][a3.slot_id]["kv"] == 30
    finally:
        await close_client(mgr)


async def test_wait_times_out_when_all_busy(tmp_path, fake):
    fake.load_model("m1", n_slots=1)
    mgr = make(tmp_path, fake, wait_timeout=0.2)
    try:
        a1 = await mgr.allocate("m1", "fe", "s1")
        fake.slots["m1"][a1.slot_id]["busy"] = True  # actively generating
        # do not release: slot is busy AND resident to s1, so s2 can't evict it

        start = asyncio.get_event_loop().time()
        with pytest.raises(TimeoutError):
            await mgr.allocate("m1", "fe", "s2")
        assert asyncio.get_event_loop().time() - start >= 0.15
    finally:
        await close_client(mgr)


async def test_busy_slot_not_taken_by_other(tmp_path, fake):
    """A busy slot (actively generating for another session) is not evicted;
    the other session waits and times out."""
    fake.load_model("m1", n_slots=1)
    mgr = make(tmp_path, fake, wait_timeout=0.3)
    try:
        a1 = await mgr.allocate("m1", "fe", "s1")
        fake.slots["m1"][a1.slot_id]["busy"] = True  # s1 actively generating

        with pytest.raises(TimeoutError):
            await mgr.allocate("m1", "fe", "s2")

        # generation finishes; s2 can now evict s1 (save first) and proceed
        fake.slots["m1"][a1.slot_id]["busy"] = False
        a3 = await mgr.allocate("m1", "fe", "s2")
        assert a3.evicted == ("fe", "s1")
    finally:
        await close_client(mgr)


async def test_same_session_reuses_busy_slot(tmp_path, fake):
    """A session always gets its own slot back, even while it's busy."""
    fake.load_model("m1", n_slots=2)
    mgr = make(tmp_path, fake)
    try:
        a1 = await mgr.allocate("m1", "fe", "s1")
        fake.slots["m1"][a1.slot_id]["busy"] = True
        a2 = await mgr.allocate("m1", "fe", "s1")
        assert a2.slot_id == a1.slot_id
    finally:
        await close_client(mgr)


async def test_invalidate_model_drops_residency(tmp_path, fake):
    fake.load_model("m1", n_slots=2)
    mgr = make(tmp_path, fake)
    try:
        await mgr.allocate("m1", "fe", "s1")
        assert mgr._session_slot.get(("fe", "s1", "m1")) is not None

        mgr.invalidate_model("m1", reason="unloaded")
        assert ("fe", "s1", "m1") not in mgr._session_slot

        # disk cache survives the swap
        fake.slots["m1"][0]["kv"] = 20
        await mgr.save_session("m1", 0, "fe", "s1")
        assert mgr.store.get("fe", "s1", "m1") is not None

        # model reloaded: next allocate restores from disk
        fake.load_model("m1", n_slots=2)
        a = await mgr.allocate("m1", "fe", "s1")
        assert a.restored is True
    finally:
        await close_client(mgr)


async def test_watcher_invalidates_on_unload(tmp_path, fake):
    fake.load_model("m1", n_slots=2)
    mgr = make(tmp_path, fake)
    mgr.poll_interval = 0.05
    mgr.ensure_watcher()
    try:
        await asyncio.sleep(0.15)  # let the watcher take its first snapshot
        await mgr.allocate("m1", "fe", "s1")
        assert ("fe", "s1", "m1") in mgr._session_slot

        # manual unload via llama-swap (remove from /running AND slots state)
        with fake.running_lock:
            fake.running.remove("m1")
        fake.slots.pop("m1", None)
        for _ in range(80):
            if ("fe", "s1", "m1") not in mgr._session_slot:
                break
            await asyncio.sleep(0.05)
        # watcher runs every 0.05s; give it a couple of cycles after detection
        assert ("fe", "s1", "m1") not in mgr._session_slot, \
            f"watcher did not invalidate residency within 4s: {mgr._session_slot}"
    finally:
        await mgr.stop()
        await close_client(mgr)
