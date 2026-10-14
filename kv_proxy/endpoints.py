"""HTTP endpoints: the core chat path plus passthroughs and cache controls."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .cache_store import CacheStore
from .identity import Identity, model_of, resolve_identity
from .llama_client import LlamaClient, LlamaError
from .slot_manager import SlotManager

log = logging.getLogger("kv_proxy.http")


class Proxy:
    """Wires config + client + store + slot manager together."""

    def __init__(self, client: LlamaClient, store: CacheStore, slots: SlotManager,
                 save_after_request: bool = True):
        self.client = client
        self.store = store
        self.slots = slots
        self.save_after_request = save_after_request


def create_app(proxy: Proxy) -> FastAPI:
    app = FastAPI(title="kv-proxy", version="0.1.0")
    app.state.proxy = proxy

    # ------------------------------------------------------------------ core

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle_chat(request, path="/v1/chat/completions")

    @app.post("/completion")
    async def completion(request: Request):
        return await _handle_chat(request, path="/completion")

    # ------------------------------------------------------------- passthrough

    @app.get("/v1/models")
    async def models():
        return JSONResponse(await proxy.client.get_models())

    @app.get("/health")
    async def health():
        ok = await proxy.client.health()
        return JSONResponse({"status": "ok" if ok else "degraded"},
                            status_code=200 if ok else 503)

    # ------------------------------------------------------------ cache admin

    @app.get("/status")
    async def status():
        p: Proxy = app.state.proxy
        entries = p.store.list()
        return JSONResponse({
            "sessions": [
                {
                    "frontend": e.frontend,
                    "session": e.session,
                    "model": e.model,
                    "size_bytes": e.size,
                    "n_saved": e.n_saved,
                    "last_access": e.last_access,
                }
                for e in entries
            ],
            "total_size_bytes": p.store.total_size,
            "max_cache_size_bytes": p.store.max_size,
            "slots": p.slots.status(),
        })

    @app.delete("/cache")
    async def delete_cache(request: Request):
        p: Proxy = app.state.proxy
        frontend = request.query_params.get("frontend")
        session = request.query_params.get("session")
        model = request.query_params.get("model")
        if not frontend:
            return JSONResponse({"error": "frontend query param required"}, status_code=400)
        freed = p.store.delete(frontend, session, model)
        p.store.enforce_max()
        return JSONResponse({"freed_bytes": freed})

    @app.exception_handler(LlamaError)
    async def llama_error_handler(request: Request, exc: LlamaError):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "upstream_error"}},
            status_code=exc.status if 400 <= exc.status < 600 else 502,
        )

    return app


async def _handle_chat(request: Request, path: str):
    proxy: Proxy = request.app.state.proxy
    proxy.slots.ensure_watcher()  # idempotent; spawns on the active loop if deferred
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    identity: Identity = resolve_identity(
        {k.lower(): v for k, v in request.headers.items()}, body, _keys(request)
    )
    model = model_of(body)
    if not model:
        return JSONResponse({"error": {"message": "missing 'model' field"}}, status_code=400)

    fe, sess = identity.frontend, identity.session

    try:
        alloc = await proxy.slots.allocate(model, fe, sess, must_restore=True)
    except LlamaError as exc:
        log.warning("slot allocation failed for %s/%s on %s: %s", fe, sess, model, exc)
        return JSONResponse(
            {"error": {"message": str(exc), "type": "upstream_error"}},
            status_code=exc.status if 400 <= exc.status < 600 else 502,
        )
    except TimeoutError as exc:
        return JSONResponse({"error": {"message": str(exc), "type": "slot_timeout"}},
                            status_code=503)

    # pin the request to the allocated slot
    body["id_slot"] = alloc.slot_id
    headers = _forward_headers(request)
    stream = bool(body.get("stream"))

    if not stream:
        try:
            status_code, data = await proxy.client.forward_json(path, body, headers)
        except LlamaError as exc:
            # slot may have been released by the server; drop stale residency
            proxy.slots.forget(model, fe, sess)
            raise
        finally:
            proxy.slots.release(model, fe, sess)
        if status_code == 200 and proxy.save_after_request:
            _schedule_save(proxy, model, alloc.slot_id, fe, sess)
        return JSONResponse(data, status_code=status_code)

    async def sse():
        status_code = 200
        try:
            agen = proxy.client.forward_stream(path, body, headers)
            first = True
            async for code, rheaders, chunk in agen:
                if first:
                    first = False
                    status_code = code
                    yield _sse_start(code, rheaders)
                if chunk:
                    yield chunk
        except LlamaError as exc:
            log.warning("stream error for %s/%s on %s: %s", fe, sess, model, exc)
            proxy.slots.forget(model, fe, sess)
            payload = json.dumps({"error": {"message": str(exc)}}).encode()
            yield f"data: {payload}\n\n".encode()
        finally:
            proxy.slots.release(model, fe, sess)
            if status_code == 200 and proxy.save_after_request:
                _schedule_save(proxy, model, alloc.slot_id, fe, sess)

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


def _keys(request: Request) -> dict[str, str]:
    cfg = getattr(request.app.state, "config", None)
    return getattr(cfg, "keys", {}) if cfg is not None else {}


def _forward_headers(request: Request) -> dict[str, str]:
    """Forward the client's auth so llama-swap can validate its own keys too."""
    out = {}
    auth = request.headers.get("authorization")
    if auth:
        out["Authorization"] = auth
    return out


def _sse_start(status_code: int, rheaders: dict[str, str]) -> bytes:
    """Emit non-200 upstream errors as a JSON SSE event so clients see them."""
    if status_code == 200:
        return b""
    payload = json.dumps({"error": {"message": f"upstream returned {status_code}"}}).encode()
    return f"data: {payload}\n\n".encode()


def _schedule_save(proxy: Proxy, model: str, slot_id: int, fe: str, sess: str) -> None:
    """Fire-and-forget background save of the session's slot KV.

    Failures are logged inside SlotManager.save_session (best-effort by design).
    """
    import asyncio

    async def _run():
        try:
            await proxy.slots.save_session(model, slot_id, fe, sess)
        except Exception as exc:  # noqa: BLE001 - never crash a bg task
            log.warning("background save crashed for %s/%s: %s", fe, sess, exc)

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:  # no running loop (sync test context): skip, next request retries
        pass
