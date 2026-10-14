# kv-proxy

API-level KV cache manager: a single OpenAI-compatible proxy in front of
**llama-swap** (→ `llama-server`) that transparently saves and restores each
conversation's KV cache to disk, per `<frontend>/<session>/<model>`.

```
SillyTavern ──┐
Hermes        ├──▶ kv-proxy :8080 ──▶ llama-swap :<swap-port> ──▶ llama-server (dynamic port, per model)
Open WebUI    ─┘         │
                         └──▶ ~/.kv-cache/<frontend>/<session>/<model>/state.bin
```

Frontends need **no code changes** — they just point their llama.cpp endpoint at
the proxy and (optionally) use a per-frontend API key. No frontend knows or
cares about KV caching.

## How it works

1. **Identify** each request: `Authorization: Bearer <key>` → frontend name
   (from config); session id from the `X-KV-Session` header, else the OpenAI
   `user` field, else `default`. Model from the body's `model` field.
2. **Allocate a slot** on the model's llama-server (via llama-swap's
   `/upstream/{model}/slots` passthrough): reuse the session's own slot, else
   an idle slot, else save+evict another session's idle slot, else queue until
   `wait_for_slot_timeout`.
3. **Restore** the session's saved KV into that slot *before* forwarding the
   request (with `id_slot` injected so llama.cpp uses exactly that slot).
4. **Forward** the request — SSE streaming passes through byte-for-byte.
5. **Save** the slot's KV to disk in the background after each successful
   response: llama.cpp writes to `--slot-save-path`, kv-proxy moves the blob
   into `~/.kv-cache/<frontend>/<session>/<model>/state.bin` (atomic rename).
6. **Evict** oldest sessions by last-access when total cache exceeds
   `max_cache_size`.

Model swaps are handled both ways: per-request `model` changes re-restore that
model's cache inline, and a background poller on llama-swap's `/running`
detects manual loads/unloads (UI or API) so cached state is re-synced after any
swap. This fixes the documented race in llama-swap where the first request
arrives before a restore could happen ([#615](https://github.com/mostlygeek/llama-swap/discussions/615)).

## Requirements

- Python 3.10+
- llama-swap (any recent version) exposing `/upstream/{model}` passthrough and `/running`
- llama.cpp build with `--slot-save-path` support (current master; the proxy
  fails fast with a clear error on first use if your build lacks it)

## Setup

### 1. Add flags to llama-swap's config.yaml (per model entry)

Do **not** point this at anything else — add to each model's `cmd`:

```yaml
models:
  my-model:
    cmd: llama-server --port ${PORT} --model /path/to/model.gguf \
      --slot-save-path ~/.kv-cache/slots -sps 0.0 -np 4
```

| Flag | Why |
|---|---|
| `--slot-save-path ~/.kv-cache/slots` | **Required.** Staging dir where llama.cpp writes slot blobs; kv-proxy moves them into the per-session layout. Must match `slot_save_path` in kv-proxy's config. |
| `-np 4` | Slot count. ≥ your expected *concurrent* generations. More sessions than slots is fine — that's what save/restore is for; simultaneous generations beyond `-np` queue. |
| `-sps 0.0` | Disables llama.cpp's prompt-similarity slot picker so it doesn't fight the proxy's explicit `id_slot` routing. |

Restart llama-swap after editing.

**Not using llama-swap?** kv-proxy can point directly at a plain `llama-server`
instead. Set `llama_swap_url` to the server's address:

```toml
llama_swap_url = "http://127.0.0.1:8081"   # your llama-server, any port
```

Everything else works identically — slot save/restore go straight to the
server's `/slots` endpoint (no `/upstream/{model}` prefix needed), and the
`--slot-save-path` / `-sps 0.0` flags above still apply to how you launch
llama-server. The only things you lose are llama-swap features: model hot-swapping
and the `/running` swap watcher (with a single always-loaded model there's
nothing to watch).

### 2. Install and configure kv-proxy

```sh
cd llamaCache
python3 -m venv .venv && .venv/bin/pip install -e .
cp config.example.toml config.toml   # edit keys, ports, sizes
.venv/bin/kv-proxy --config config.toml
```

Or with uv: `uv pip install -e .` then `kv-proxy`.

### 3. Point frontends at the proxy

- **Base URL**: `http://<host>:8080/v1` (or `/completion` for legacy clients)
- **API key** (per frontend, optional but recommended): set each frontend's
  llama.cpp API key to its configured key (`sillytavern`, `hermes`, `openwebui`
  in `[keys]`). Requests without a recognized key are cached under the
  `default` frontend.
- **Session id**: frontends that let you set a custom header can send
  `X-KV-Session: <id>`; otherwise the OpenAI `user` field is used; otherwise
  everything shares one session per frontend (one cache each).

#### Adding keys for more frontends

Each entry in `[keys]` maps **frontend name → API key**. The name becomes the
directory under `~/.kv-cache/`, and the key is what that frontend puts in its
"llama.cpp / OpenAI API key" setting. Add as many as you have frontends — just
append a line:

```toml
[keys]
sillytavern = "sk-st-change-me"
hermes = "sk-hermes-change-me"
openwebui = "sk-owu-change-me"

# adding a fourth frontend? one more line is all it takes:
koboldcpp = "sk-kobold-change-me"
novelai   = "sk-nai-change-me"
```

Then in that new frontend's settings, set base URL to the proxy and API key to
`sk-kobold-change-me`. Its sessions will be cached under
`~/.kv-cache/koboldcpp/<session>/...` automatically — no other config changes.

Rules of thumb:

- **Keys must be unique** across frontends (the first match wins).
- Names are filesystem-safe: letters, digits, `._-` (anything else gets
  sanitized + hashed).
- No key / unknown key → everything lands in the shared `default` frontend.
- Edit `config.toml`, then **restart kv-proxy** (`kv-proxy --config config.toml`).

Example curl:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-st-change-me" \
  -H "X-KV-Session: session-abc123" \
  -d '{"model": "my-model", "messages": [{"role":"user","content":"hi"}], "stream": false}'
```

## Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /v1/chat/completions` | Core path (identify → allocate → restore → forward SSE/JSON → background save) |
| `POST /completion` | Legacy llama.cpp endpoint, same logic |
| `GET /v1/models`, `GET /health` | Passthrough to llama-swap |
| `GET /status` | Dashboard: sessions per frontend/model, sizes, resident slots |
| `DELETE /cache?frontend=X[&session=Y][&model=Z]` | Manually drop cached KV |

## Configuration reference (config.toml)

| Key | Default | Description |
|---|---|---|
| `listen` | `0.0.0.0:8080` | Bind address (`KV_PROXY_LISTEN`) |
| `llama_swap_url` | `http://127.0.0.1:<swap-port>` | The only upstream kv-proxy knows (`KV_PROXY_LLAMA_SWAP_URL`) |
| `cache_dir` | `~/.kv-cache` | Per-session blob root (`KV_PROXY_CACHE_DIR`) |
| `slot_save_path` | `<cache_dir>/slots` | Must match llama-server's `--slot-save-path` |
| `max_cache_size` | `20GB` | Global cap; LRU-evicts oldest beyond this (`KV_PROXY_MAX_CACHE_SIZE`) |
| `wait_for_slot_timeout` | `120` | Seconds to queue when all slots busy; 0 = fail fast with 503 |
| `model_poll_interval` | `2.0` | `/running` poll cadence (seconds); 0 disables the watcher |
| `save_after_request` | `true` | Background-save KV after each successful request |
| `[direct_upstream]` | — | Escape hatch: `model = "host:port"` for slot ops if your llama-swap lacks `/upstream` |
| `[keys]` | — | `frontend_name = "api-key"` mappings |

## Architecture notes

- **Why explicit identity, not prefix similarity**: the popular [proxycache](https://github.com/airnsk/proxycache)
  routes by prompt-similarity matching. That's nondeterministic — two
  conversations sharing a system prompt can collide. kv-proxy routes by
  `(frontend, session, model)` identity: a session always maps to its own
  cache. llama.cpp's own `cache_prompt` still reuses the restored slot within
  a request.
- **Slot ops via `/upstream/{model}`**: llama-swap doesn't proxy `/slots`
  itself ([#626](https://github.com/mostlygeek/llama-swap/discussions/626)),
  but its `/upstream/{model_id}` passthrough reaches the real server, so
  kv-proxy has exactly one upstream address and never needs to know llama-swap's
  dynamic `${PORT}` ports.
- **Concurrency**: per-session async lock (one slot serves at most one session
  at a time) + a global allocation lock per model. A session's own busy slot is
  never stolen; other sessions queue.
- **Saves are best-effort**: a slow disk save never blocks user-visible
  latency; failures are logged and retried on the next request.

## References & inspiration

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the inference server,
  including the slot persistence API this project builds on: `--slot-save-path`
  with `POST /slots/{id}?action=save|restore` and `GET /slots`. See also the
  official [KV cache reuse tutorial](https://github.com/ggml-org/llama.cpp/discussions/13606).
- [llama-swap](https://github.com/mostlygeek/llama-swap) — the model-swapping
  proxy kv-proxy sits in front of. Slot caching is not a llama-swap feature
  (see its discussions [#615](https://github.com/mostlygeek/llama-swap/discussions/615) and
  [#626](https://github.com/mostlygeek/llama-swap/discussions/626)); this project
  fills that gap, and relies on llama-swap's `/upstream/{model}` passthrough and
  `/running` endpoint.
- [proxycache](https://github.com/airnsk/proxycache) — prior art for a smart
  OpenAI-compatible proxy managing llama.cpp slots with save/restore. kv-proxy
  takes a different routing strategy: explicit `(frontend, session, model)`
  identity instead of prefix-similarity matching (see Architecture notes).

## Development

```sh
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest            # 34 tests, all mocked — no live model needed
.venv/bin/pytest -x -q
```

Tests use a `FakeSwap` (in-memory llama-swap + llama-server stand-in) covering
identity resolution, LRU eviction, the slot-allocation matrix (free / evict /
wait-timeout / busy-own-slot), model-swap invalidation, and endpoint behavior
including SSE streaming.

## Known limitations

- `id_slot` must be honored by your llama.cpp build for deterministic routing
  (current master does; with `-sps 0.0` it's the only routing mechanism).
- Slot count `-np` bounds *concurrent* generations, not total sessions.
- If a model is unloaded mid-generation by llama-swap, in-flight requests may
  fail once; the next request re-restores cleanly.
