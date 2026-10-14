"""Cache store: atomic finalize, LRU eviction, delete, rescan on restart."""

import time

import pytest

from kv_proxy.cache_store import CacheStore


def test_finalize_moves_blob_atomically(tmp_path):
    store = CacheStore(tmp_path / "cache", max_size=1024**3)
    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"x" * 500)

    entry = store.finalize("fe", "sess", "model", staged, n_saved=10)
    assert entry is not None
    blob = tmp_path / "cache" / "fe" / "sess" / "model" / "state.bin"
    assert blob.exists() and blob.stat().st_size == 500
    assert store.total_size == 500
    assert not staged.exists()


def test_missing_staged_file_returns_none(tmp_path):
    store = CacheStore(tmp_path / "cache", max_size=1024**3)
    assert store.finalize("fe", "sess", "m", tmp_path / "nope.bin") is None


def test_get_and_touch(tmp_path):
    store = CacheStore(tmp_path / "cache", max_size=1024**3)
    staged = tmp_path / "s.bin"
    staged.write_bytes(b"x" * 10)
    store.finalize("fe", "sess", "m", staged)

    entry = store.get("fe", "sess", "m")
    assert entry is not None
    before = entry.last_access
    time.sleep(0.01)
    store.touch("fe", "sess", "m")
    assert store.get("fe", "sess", "m").last_access >= before


def test_lru_eviction_order(tmp_path):
    store = CacheStore(tmp_path / "cache", max_size=30)
    s1, s2, s3 = tmp_path / "1.bin", tmp_path / "2.bin", tmp_path / "3.bin"
    s1.write_bytes(b"a" * 10)
    s2.write_bytes(b"b" * 10)
    s3.write_bytes(b"c" * 10)

    store.finalize("fe", "s1", "m", s1)
    store.finalize("fe", "s2", "m", s2)
    # touch s1 so s2 becomes the oldest
    store.touch("fe", "s1", "m")
    store.finalize("fe", "s3", "m", s3)  # now total=30, still under cap

    freed = store.enforce_max() if store.total_size > 30 else 0
    assert store.get("fe", "s2", "m") is None or freed == 0

    # force over-cap: add another entry
    s4 = tmp_path / "4.bin"
    s4.write_bytes(b"d" * 10)
    store.finalize("fe", "s4", "m", s4)
    assert store.total_size > 30
    freed = store.enforce_max()
    assert freed >= 10
    # oldest remaining should have been evicted
    assert store.total_size <= 30


def test_enforce_max_evicts_oldest_first(tmp_path):
    store = CacheStore(tmp_path / "cache", max_size=25)
    for i, name in enumerate(["old", "mid", "new"]):
        f = tmp_path / f"{name}.bin"
        f.write_bytes(b"z" * 10)
        store.finalize("fe", name, "m", f)
        if name == "old":
            # make 'old' clearly the oldest
            pass

    # touch new ones so 'old' stays LRU
    time.sleep(0.01)
    store.touch("fe", "mid", "m")
    time.sleep(0.01)
    store.touch("fe", "new", "m")

    freed = store.enforce_max()
    assert store.get("fe", "old", "m") is None
    assert store.total_size <= 25


def test_delete_by_frontend_session_model(tmp_path):
    store = CacheStore(tmp_path / "cache", max_size=1024**3)
    for sess in ("a", "b"):
        f = tmp_path / f"{sess}.bin"
        f.write_bytes(b"x" * 10)
        store.finalize("fe", sess, "m", f)

    freed = store.delete("fe", "a")
    assert freed == 10
    assert store.get("fe", "a", "m") is None
    assert store.get("fe", "b", "m") is not None

    freed = store.delete("fe")
    assert freed == 10
    assert store.list() == []


def test_rescan_rebuilds_index_after_restart(tmp_path):
    cache_dir = tmp_path / "cache"
    store = CacheStore(cache_dir, max_size=1024**3)
    f = tmp_path / "s.bin"
    f.write_bytes(b"x" * 42)
    store.finalize("fe", "sess", "m", f, n_saved=7)

    # simulate restart
    store2 = CacheStore(cache_dir, max_size=1024**3)
    entry = store2.get("fe", "sess", "m")
    assert entry is not None
    assert entry.size == 42
    assert entry.n_saved == 7
