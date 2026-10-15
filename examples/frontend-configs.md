# Frontend wiring examples

All frontends: point the llama.cpp / OpenAI-compatible endpoint at
`http://<proxy-host>:8080/v1` instead of llama-swap directly.

## SillyTavern

- **Connection**: Settings → Connections → add a connection with the **Custom**
  source, URL `http://localhost:8080/v1` (your proxy's /v1).
- **API key**: set to the `sillytavern` key from kv-proxy's `[keys]`.
- **Session id**: leave the *Additional Parameters* fields (the button next to
  the URL field on the Custom source) **empty**. kv-proxy then derives a stable
  per-chat session id from the conversation's opening messages (system prompt
  + first user turn), so every chat gets its own cache automatically — no
  client configuration needed.
- **Important**: ST has **no per-chat id macro** — `{{chat}}`, `{{chatid}}`,
  `{{chatName}}` etc. do not exist (verified against the 1.19.0 macro list).
  Unknown macros are sent **verbatim**, so `{"session": "{{chat}}"}` in the
  additional body makes *every* chat send the same literal session id, and
  kv-proxy's `body["session"]` takes priority over the content hash — all
  chats end up sharing one cache and overwriting each other. If you do pin an
  explicit id, use something that actually resolves per conversation, e.g.
  per-character `{"session": "{{char}}"}` (all chats with one character share
  one cache), or a per-chat global variable via `{{getvar::...}}`.
  A reverse proxy injecting `X-KV-Session: <chat-id>` works too.

## Open WebUI

- **Admin → Settings → Inference Base URL**: `http://localhost:8080/v1`.
- **API key**: the `openwebui` key.
- Per-chat isolation is automatic: kv-proxy derives the session id from the
  conversation's opening messages, so each chat gets its own cache. (The
  `user` field, if sent, is only a coarser fallback.)

## Hermes / generic OpenAI clients

- Base URL: `http://localhost:8080/v1`
- API key: the `hermes` key (or any configured key).
- Session id (first non-empty wins): `X-KV-Session` header → `body["session"]`
  → `body["id"]` → `messages[*]["metadata"]` / `messages[*]["id"]` →
  **automatic per-chat hash of the opening messages** → `body["user"]` →
  shared `default` session. In practice the opening-message hash means every
  chat gets its own cache with no configuration.

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
