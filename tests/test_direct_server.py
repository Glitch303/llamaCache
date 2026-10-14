"""Direct llama-server mode (no llama-swap): slot ops hit /slots directly."""

import httpx
import pytest

from kv_proxy.llama_client import LlamaClient, LlamaError


class FakeLlamaServer:
    """Stands in for a plain llama-server (no /upstream, no /running)."""

    def __init__(self):
        self.model = "direct-model"
        self.slots = {0: {"busy": False, "kv": 0}, 1: {"busy": False, "kv": 0}}
        self.save_dir_files: dict[str, bytes] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        import json

        path = request.url.path
        body = json.loads(request.content.decode() or "{}") if request.method == "POST" else {}

        if path == "/health":
            return httpx.Response(200, text="ok")
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": self.model}]})
        if path == "/slots" and request.method == "GET":
            return httpx.Response(200, json=[
                {"id": i, "is_processing": s["busy"]} for i, s in sorted(self.slots.items())
            ])
        if path.startswith("/slots/"):
            sid = int(path.split("/")[2])
            action = request.url.params.get("action", "")
            if action == "save":
                fname = body["filename"]
                self.save_dir_files[fname] = b"KV" * max(self.slots[sid]["kv"], 1)
                return httpx.Response(200, json={"id_slot": sid, "n_saved": self.slots[sid]["kv"]})
            if action == "restore":
                fname = body["filename"]
                if fname not in self.save_dir_files:
                    return httpx.Response(404, json={"error": "no file"})
                self.slots[sid]["kv"] = len(self.save_dir_files[fname]) // 2
                return httpx.Response(200, json={"id_slot": sid, "n_restored": self.slots[sid]["kv"]})
        if path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "direct"}}]})
        # /upstream/... should NOT be used in direct mode
        return httpx.Response(404, json={"error": f"unexpected route {path}"})


@pytest.fixture
def server():
    return FakeLlamaServer()


@pytest.fixture
def client(server):
    c = LlamaClient("http://srv", direct_llama_server=True)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(server.handle), base_url="http://srv")
    return c


async def test_slots_direct_path(client, server):
    slots = await client.get_slots("direct-model")
    assert [s["id"] for s in slots] == [0, 1]


async def test_save_restore_direct(client, server):
    server.slots[0]["kv"] = 25
    r = await client.save_slot("direct-model", 0, "kv-a-b.bin")
    assert r["n_saved"] == 25

    server.slots[1]["kv"] = 0
    r2 = await client.restore_slot("direct-model", 1, "kv-a-b.bin")
    assert r2["n_restored"] == 25
    assert server.slots[1]["kv"] == 25


async def test_running_models_from_v1_models(client):
    models = await client.running_models()
    assert models == ["direct-model"]
