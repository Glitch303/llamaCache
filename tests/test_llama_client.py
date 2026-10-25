"""LlamaClient unit tests: /running parsing and upstream auth header."""

import httpx

from kv_proxy.llama_client import LlamaClient


def _client_with(response: httpx.Response, api_key: str | None = None):
    captured: dict = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        return response

    client = LlamaClient("http://swap", api_key=api_key)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://swap",
    )
    return client, captured


async def test_running_models_llamaswap_shape():
    # real llama-swap /running: {"running": [{"model": "...", ...}]}
    client, _ = _client_with(httpx.Response(200, json={
        "running": [{"model": "m1", "state": "ready"}, {"model": "m2"}],
    }))
    async with client:
        assert await client.running_models() == ["m1", "m2"]


async def test_running_models_plain_list_shape():
    client, _ = _client_with(httpx.Response(200, json=["m1", {"id": "m2"}]))
    async with client:
        assert await client.running_models() == ["m1", "m2"]


async def test_running_models_sends_auth():
    client, captured = _client_with(
        httpx.Response(200, json={"running": [{"model": "m1"}]}),
        api_key="ls-k",
    )
    async with client:
        assert await client.running_models() == ["m1"]
    assert captured["auth"] == "Bearer ls-k"


async def test_no_key_means_no_auth_header():
    client, captured = _client_with(httpx.Response(200, json={"running": []}))
    async with client:
        assert await client.running_models() == []
    assert captured["auth"] is None


async def test_forward_json_auth_wins_over_forwarded_headers():
    client, captured = _client_with(httpx.Response(200, json={"ok": True}),
                                    api_key="ls-k")
    async with client:
        status, data = await client.forward_json(
            "/v1/chat/completions", {"model": "m1"},
            headers={"Authorization": "Bearer client-attribution-key"},
        )
    assert status == 200 and data == {"ok": True}
    assert captured["auth"] == "Bearer ls-k"
