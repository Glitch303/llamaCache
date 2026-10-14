"""Application entrypoint for kv-proxy."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from .config import Config, load_config
from .endpoints import Proxy, create_app
from .llama_client import LlamaClient
from .cache_store import CacheStore
from .slot_manager import SlotManager

log = logging.getLogger("kv_proxy")


def build_app(config: Config | None = None):
    """Build the FastAPI app with all collaborators wired up. Returns (app, proxy)."""
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app):
        log.info(
            "kv-proxy starting: upstream=%s cache_dir=%s max_cache=%dMB",
            cfg.llama_swap_url, cfg.cache_dir, cfg.max_cache_size // (1024 * 1024),
        )
        proxy.slots.ensure_watcher()
        try:
            yield
        finally:
            await proxy.slots.stop()
            await proxy.client.aclose()

    client = LlamaClient(
        cfg.llama_swap_url, cfg.direct_upstream,
        direct_llama_server=cfg.direct_llama_server,
    )
    store = CacheStore(cfg.cache_dir, cfg.max_cache_size)
    slots = SlotManager(
        client, store,
        slot_save_path=cfg.slot_save_path,
        wait_timeout=cfg.wait_for_slot_timeout,
        poll_interval=cfg.model_poll_interval,
    )
    proxy = Proxy(client, store, slots, save_after_request=cfg.save_after_request)
    app = create_app(proxy)
    app.state.config = cfg
    app.router.lifespan_context = lifespan
    return app, proxy


def cli() -> None:
    parser = argparse.ArgumentParser(prog="kv-proxy", description=__doc__)
    parser.add_argument("--config", default=None, help="path to TOML config file")
    parser.add_argument("--listen", default=None, help="override listen address (host:port)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    cfg = load_config(args.config)
    if args.listen:
        cfg.listen = args.listen
        cfg.validate()

    app, _ = build_app(cfg)

    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    cli()
