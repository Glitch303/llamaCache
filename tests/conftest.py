"""Shared fixtures: a fake llama-swap server (httpx.MockTransport) + app wiring."""

from __future__ import annotations

import json
import time

import httpx
import pytest


class FakeSwap:
    """In-memory stand-in for llama-swap + one llama-server.

    Tracks slots per model, running models, and a --slot-save-path dir where
    save/restore blobs live (real files on disk).
    """

    def __init__(self, tmp_path):
        self.save_dir = tmp_path / "slots"
        self.save_dir.mkdir()
        self.running: list[str] = []
        self.running_lock = __import__("threading").Lock()  # tests mutate from the event loop thread; keep reads atomic
        # model -> {slot_id: {"busy": bool, "kv": int}}
        self.slots: dict[str, dict[int, dict]] = {}
        self.requests: list[dict] = []  # recorded forwarded chat requests
        self.chat_replies: dict[str, object] = {}  # model -> reply body

    def load_model(self, model: str, n_slots: int = 2):
        if model not in self.running:
            self.running.append(model)
        self.slots.setdefault(model, {i: {"busy": False, "kv": 0} for i in range(n_slots)})

    # ------------------------------------------------------------- handlers

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode() or "{}") if request.method == "POST" else {}

        if path == "/running":
            with self.running_lock:
                snapshot = list(self.running)
            return httpx.Response(200, json=snapshot)

        if path == "/health":
            return httpx.Response(200, text="ok")

        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.running]})

        # /upstream/{model}/...
        parts = path.split("/")
        if len(parts) >= 2 and parts[1] == "upstream":
            model = parts[2]
            rest = "/".join(parts[3:])
            return self._handle_upstream(model, request, rest, body)

        # OpenAI endpoints proxied by llama-swap
        if path in ("/v1/chat/completions", "/completion"):
            model = body.get("model", "")
            if model not in self.running:
                return httpx.Response(404, json={"error": f"model {model} not loaded"})
            self.requests.append({"path": path, "body": body})
            reply = self.chat_replies.get(model, {"choices": [{"message": {"content": "hi"}}]})
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200, json=reply)

        return httpx.Response(404, json={"error": f"fake: no route {path}"})

    def _handle_upstream(self, model: str, request: httpx.Request, rest: str, body: dict):
        if model not in self.running:
            return httpx.Response(503, json={"error": f"model {model} not loaded"})
        slots = self.slots.setdefault(model, {})

        if rest == "health":
            return httpx.Response(200, text="ok")

        if rest == "slots" and request.method == "GET":
            out = [
                {"id": sid, "is_processing": s["busy"], "n_ctx": 4096}
                for sid, s in sorted(slots.items())
            ]
            return httpx.Response(200, json=out)

        # slots/{id}?action=...
        if rest.startswith("slots/"):
            try:
                sid = int(rest.split("/")[1])
            except (IndexError, ValueError):
                return httpx.Response(400, json={"error": "bad slot id"})
            action = request.url.params.get("action", "")
            if sid not in slots:
                return httpx.Response(404, json={"error": f"no slot {sid}"})

            if action == "save":
                fname = body.get("filename", "slot.bin")
                kv = slots[sid]["kv"]
                (self.save_dir / fname).write_bytes(b"KV" * max(kv, 1))
                return httpx.Response(200, json={
                    "id_slot": sid, "filename": fname,
                    "n_saved": kv, "n_written": 2 * max(kv, 1),
                })
            if action == "restore":
                fname = body.get("filename", "slot.bin")
                f = self.save_dir / fname
                if not f.exists():
                    return httpx.Response(404, json={"error": f"no file {fname}"})
                kv = len(f.read_bytes()) // 2
                slots[sid]["kv"] = kv
                return httpx.Response(200, json={
                    "id_slot": sid, "filename": fname,
                    "n_restored": kv, "n_read": len(f.read_bytes()),
                })
            if action == "erase":
                slots[sid]["kv"] = 0
                return httpx.Response(200, json={"id_slot": sid, "n_erased": 0})
            return httpx.Response(400, json={"error": f"bad action {action}"})

        if rest == "slots":
            return httpx.Response(405, json={"error": "method not allowed"})

        return httpx.Response(404, json={"error": f"fake upstream: no route {rest}"})


@pytest.fixture
def fake(tmp_path):
    return FakeSwap(tmp_path)
