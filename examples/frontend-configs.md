# Frontend wiring examples

All frontends: point the llama.cpp / OpenAI-compatible endpoint at
`http://<proxy-host>:8080/v1` instead of llama-swap directly.

## SillyTavern

- **Connection**: Settings → Connections → add a "llama.cpp (OpenAI)" connection,
  base URL `http://localhost:8080/v1`.
- **API key**: set to the `sillytavern` key from kv-proxy's `[keys]`.
- **Session id**: SillyTavern doesn't expose arbitrary custom headers on the
  OpenAI connector by default. Two options:
  - Leave it — all chats share one cached session per model (fine for single
    user; each chat still reuses its own conversation via `cache_prompt`).
  - Or use a reverse proxy in front that injects `X-KV-Session` from the
    character/chat id.

## Open WebUI

- **Admin → Settings → Inference Base URL**: `http://localhost:8080/v1`.
- **API key**: the `openwebui` key.
- Open WebUI sends a `user` field in chat requests (the logged-in user), which
  kv-proxy uses as the session id — so each Open WebUI user gets an isolated
  cache automatically.

## Hermes / generic OpenAI clients

- Base URL: `http://localhost:8080/v1`
- API key: the `hermes` key (or any configured key).
- Session id: send `X-KV-Session: <your-session-id>` header if your client
  allows custom headers; otherwise set `"user": "<session-id>"` in the request
  body; otherwise it falls back to the shared `default` session.

## curl smoke test

```sh
# first request: cold (full prompt processing)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-st-change-me" \
  -H "X-KV-Session: demo" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"}],"stream":false}' | jq .timings

# second request, same session: timings.cache_n should be > 0 (restored KV)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-st-change-me" \
  -H "X-KV-Session: demo" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi!"},{"role":"user","content":"and?"}],"stream":false}' | jq .timings

# inspect the cache
curl -s http://localhost:8080/status | jq .sessions
```
