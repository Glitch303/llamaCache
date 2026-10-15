"""App-level endpoint tests with a mocked llama-swap transport."""

import json

import httpx
import pytest
from httpx import ASGITransport

from kv_proxy.config import Config
from kv_proxy.endpoints import Proxy, create_app
from kv_proxy.llama_client import LlamaClient
from kv_proxy.cache_store import CacheStore
from kv_proxy.slot_manager import SlotManager


def build(tmp_path, fake, keys=None):
    cfg = Config(
        llama_swap_url="http://swap",
        cache_dir=tmp_path / "cache",
        slot_save_path=fake.save_dir,
        max_cache_size=1024**3,
        wait_for_slot_timeout=1.0,
        model_poll_interval=0.05,
        keys=keys or {},
    )
    client = LlamaClient("http://swap")
    import httpx as _h

    client._client = _h.AsyncClient(transport=_h.MockTransport(fake.handle), base_url="http://swap")
    store = CacheStore(cfg.cache_dir, cfg.max_cache_size)
    slots = SlotManager(client, store, slot_save_path=fake.save_dir,
                        wait_timeout=1.0, poll_interval=cfg.model_poll_interval)
    proxy = Proxy(client, store, slots, save_after_request=True)
    app = create_app(proxy)
    app.state.config = cfg
    return app


@pytest.fixture
def client_factory(tmp_path, fake):
    def _make(keys=None):
        app = build(tmp_path, fake, keys=keys)
        transport = ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://t")

    return _make


async def test_chat_nonstream_full_cycle(client_factory, fake, tmp_path):
    fake.load_model("m1", n_slots=2)
    client = client_factory()
    async with client:
        r = await client.post("/v1/chat/completions", json={
            "model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "hi"

        # request was pinned to a slot
        sent = fake.requests[-1]["body"]
        assert "id_slot" in sent and sent["id_slot"] in (0, 1)

        # background save should have produced a cache entry
        await _drain(client)
        entries = [e for e in _store_of(client).list()]
        assert len(entries) == 1
        assert entries[0].frontend == "default"


async def test_second_request_restores(client_factory, fake):
    fake.load_model("m1", n_slots=2)
    client = client_factory()
    async with client:
        await client.post("/v1/chat/completions", json={
            "model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": False,
        })
        await _drain(client)

        # wipe in-memory slot KV to prove restore came from disk
        for s in fake.slots["m1"].values():
            s["kv"] = 0

        # same chat, next turn: the opening message is unchanged, so the
        # content-derived session is stable and the saved blob gets restored
        r2 = await client.post("/v1/chat/completions", json={
            "model": "m1",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello!"},
                {"role": "user", "content": "hi again"},
            ],
            "stream": False,
        })
        assert r2.status_code == 200
        # slot used for the 2nd request must have KV restored from disk
        sent = fake.requests[-1]["body"]
        assert fake.slots["m1"][sent["id_slot"]]["kv"] > 0


async def test_identity_from_api_key(client_factory, fake):
    fake.load_model("m1", n_slots=2)
    client = client_factory(keys={"sillytavern": "sk-st-1"})
    async with client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "m1", "messages": [{"role": "user", "content": "x"}], "stream": False},
            headers={"Authorization": "Bearer sk-st-1"},
        )
        assert r.status_code == 200
        await _drain(client)
        entries = _store_of(client).list()
        assert len(entries) == 1 and entries[0].frontend == "sillytavern"


async def test_missing_model_400(client_factory, fake):
    client = client_factory()
    async with client:
        r = await client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 400


async def test_slot_timeout_503(client_factory, fake):
    # single busy slot: s1 holds it actively generating, s2 times out -> 503
    fake.load_model("m1", n_slots=1)
    client = client_factory()
    async with client:
        r1 = await client.post("/v1/chat/completions", json={
            "model": "m1", "messages": [{"role": "user", "content": "a"}], "stream": False,
        }, headers={"X-KV-Session": "s1"})
        assert r1.status_code == 200
        # simulate s1's generation still in flight on the only slot
        sent_slot = fake.requests[-1]["body"]["id_slot"]
        fake.slots["m1"][sent_slot]["busy"] = True

        import asyncio

        start = asyncio.get_event_loop().time()
        r2 = await client.post("/v1/chat/completions", json={
            "model": "m1", "messages": [{"role": "user", "content": "b"}], "stream": False,
        }, headers={"X-KV-Session": "s2"})
        assert asyncio.get_event_loop().time() - start >= 0.9  # waited ~wait_timeout
        assert r2.status_code == 503
        assert "no free slot" in r2.json()["error"]["message"]


async def test_streaming_sse(client_factory, fake):
    fake.load_model("m1", n_slots=2)
    sse_body = b"data: {\"delta\":{}}\n\ndata: [DONE]\n\n"
    fake.chat_replies["m1"] = httpx.Response(200, content=sse_body,
                                             headers={"content-type": "text/event-stream"})
    client = client_factory()
    async with client:
        r = await client.post("/v1/chat/completions", json={
            "model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        })
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert b"[DONE]" in r.content


async def test_models_and_health_passthrough(client_factory, fake):
    fake.load_model("m1", n_slots=1)
    client = client_factory()
    async with client:
        r = await client.get("/v1/models")
        assert r.status_code == 200
        assert any(m["id"] == "m1" for m in r.json()["data"])

        h = await client.get("/health")
        assert h.status_code == 200


async def test_status_and_cache_delete(client_factory, fake):
    fake.load_model("m1", n_slots=2)
    client = client_factory(keys={"hermes": "sk-h-1"})
    async with client:
        await client.post("/v1/chat/completions", json={
            "model": "m1", "messages": [{"role": "user", "content": "x"}], "stream": False,
        }, headers={"Authorization": "Bearer sk-h-1"})
        await _drain(client)

        st = await client.get("/status")
        assert st.status_code == 200
        body = st.json()
        assert body["total_size_bytes"] > 0
        assert any(e["frontend"] == "hermes" for e in body["sessions"])

        d = await client.delete("/cache", params={"frontend": "hermes"})
        assert d.status_code == 200
        assert d.json()["freed_bytes"] > 0

        st2 = (await client.get("/status")).json()
        assert st2["total_size_bytes"] == 0


# ------------------------------------------------------------- test helpers

async def _drain(client, n=10):
    import asyncio

    for _ in range(n):
        await asyncio.sleep(0.01)


def _store_of(client):
    return client._transport.app.state.proxy.store
