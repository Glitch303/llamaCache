"""Async HTTP client for llama-swap (and, optionally, direct llama-server).

llama-swap proxies the OpenAI endpoints but NOT /slots. Slot operations go
through its /upstream/{model}/... passthrough (or a configured direct upstream
host:port as an escape hatch).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("kv_proxy.llama")


class LlamaError(Exception):
    """Upstream returned an error for a slot operation."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"upstream {status}: {detail}")
        self.status = status
        self.detail = detail


class LlamaClient:
    def __init__(self, swap_url: str, direct_upstream: dict[str, str] | None = None,
                 timeout: float = 30.0, direct_llama_server: bool = False):
        self.swap_url = swap_url.rstrip("/")
        self.direct_upstream = {k: v.rstrip("/") for k, v in (direct_upstream or {}).items()}
        # direct_llama_server: upstream is a plain llama-server, not llama-swap.
        # Slot ops then hit /slots directly; chat requests still use the body's
        # model field (llama-server accepts and ignores it).
        self.direct_llama_server = direct_llama_server
        self._client = httpx.AsyncClient(
            base_url=self.swap_url,
            timeout=httpx.Timeout(timeout, read=3600.0),  # long reads: generations
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- slot ops

    def _slot_path(self, model: str, sub: str = "") -> str:
        """Path (relative to base_url) for /slots operations.

        llama-swap mode:  /upstream/{model}/slots[/sub]
        direct mode:      /slots[/sub]
        per-model override via [direct_upstream]: absolute host:port + /slots
        """
        if self.direct_llama_server:
            return f"/slots{('/' + sub) if sub else ''}"
        return f"/upstream/{model}/slots{('/' + sub) if sub else ''}"

    async def get_slots(self, model: str) -> list[dict]:
        r = await self._client.get(self._slot_path(model))
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    async def save_slot(self, model: str, slot_id: int, filename: str) -> dict:
        r = await self._client.post(
            self._slot_path(model, f"{slot_id}"),
            params={"action": "save"},
            json={"filename": filename},
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    async def restore_slot(self, model: str, slot_id: int, filename: str) -> dict:
        r = await self._client.post(
            self._slot_path(model, f"{slot_id}"),
            params={"action": "restore"},
            json={"filename": filename},
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    async def erase_slot(self, model: str, slot_id: int) -> dict:
        r = await self._client.post(
            self._slot_path(model, f"{slot_id}"),
            params={"action": "erase"},
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    # ------------------------------------------------------------- discovery

    async def running_models(self) -> list[str]:
        """Model ids currently loaded.

        llama-swap: GET /running. Direct llama-server: the single model it serves
        (read from /v1/models).
        """
        if self.direct_llama_server:
            try:
                data = await self.get_models()
                ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                return [str(i) for i in ids if i]
            except (httpx.HTTPError, ValueError):
                return []
        try:
            r = await self._client.get("/running")
            if r.status_code != 200:
                return []
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return []
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id") or item.get("model") or item.get("name")
                    if mid:
                        out.append(str(mid))
            return out
        if isinstance(data, dict):
            # tolerate {"models": [...]} or {model_id: {...}} shapes
            if "models" in data and isinstance(data["models"], list):
                return [str(m) for m in data["models"]]
            return list(data.keys())
        return []

    async def health(self, model: str | None = None) -> bool:
        """True when llama-swap (and the model's server, if given) is ready."""
        try:
            r = await self._client.get("/health")
            if r.status_code != 200:
                return False
            if model:
                # /upstream passthrough to the model's own health endpoint
                r2 = await self._client.get(f"/upstream/{model}/health")
                return r2.status_code == 200
            return True
        except httpx.HTTPError:
            return False

    async def get_models(self) -> Any:
        r = await self._client.get("/v1/models")
        return r.json() if r.status_code == 200 else {"data": []}

    # ---------------------------------------------------------- forwarding

    async def forward_json(
        self, path: str, body: dict, headers: dict[str, str] | None = None
    ) -> tuple[int, dict]:
        """Non-streaming forward. Returns (status_code, parsed_json)."""
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        r = await self._client.post(path, json=body, headers=h)
        try:
            data = r.json()
        except ValueError:
            data = {"error": {"message": r.text[:500]}}
        return r.status_code, data

    async def forward_stream(
        self, path: str, body: dict, headers: dict[str, str] | None = None
    ) -> AsyncIterator[tuple[int, dict, bytes]]:
        """Streaming forward. Yields (status, response_headers, chunk) tuples.

        The first yielded item carries the status line and headers; subsequent
        items carry raw body chunks. Caller is responsible for closing the
        response context (use as an async context manager around aiter_bytes).
        """
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if headers:
            h.update(headers)
        req = self._client.build_request("POST", path, json=body, headers=h)
        resp = await self._client.send(req, stream=True)
        try:
            yield resp.status_code, dict(resp.headers), b""
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield resp.status_code, {}, chunk
        finally:
            await resp.aclose()
