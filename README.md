# cortexlayer-hermes-plugin

A native [Hermes](https://hermes-agent.nousresearch.com/) `MemoryProvider` plugin for
[CortexLayer](https://www.cortexlayer.net) — the deeper integration surface Hermes calls
into every turn (prefetch before the LLM call, sync after), instead of the generic MCP tool
registration ([`cortex-backend`](https://github.com/Cortex-Layer/cortexlayer-backend)'s
`mcp_server.py`, which stays available and unaffected for MCP-generic clients like the
ChatGPT connector).

This is a thin adapter over [`cortexlayer.CortexClient`](https://github.com/Cortex-Layer/cortexlayer-python)
— CortexLayer's hosted API client, API-key authenticated — no storage/retrieval logic lives
here.

For the shorter setup walkthrough, see [docs.cortexlayer.net/docs/agents/hermes](https://docs.cortexlayer.net/docs/agents/hermes)
(also covers the alternative MCP tool-registration route). This README is the fuller,
developer-facing reference — design decisions, what's unverified, how to run the tests.

## Status

**Platform mode only (MVP decision, 2026-09-26)** — no local/embedded ("OSS") mode, unlike
mem0's Hermes integration which offers both plus a self-hosted-dashboard option. Needs a
CortexLayer API key; there is no keyless local path in this plugin today. Implements the
required lifecycle, the 5 memory tools, and the `on_session_end`/`on_pre_compress` hooks (see
"Design decisions" below). In Hermes' plugin catalog as of
[#124341](https://github.com/NousResearch/hermes-agent/pull/124341) (merged 2026-09-27);
verified end to end against a live Hermes instance the same day.

## Install

```
hermes plugins install cortexlayer
hermes plugins enable cortexlayer
```

`install` resolves this plugin's Python dependency (`cortexlayer`) automatically — no separate
`pip install` step needed (confirmed live 2026-09-27). No `[local]` extra either way: Platform
mode only talks HTTP via `CortexClient`, which depends on nothing but `httpx`. The embedded
engine's heavier deps, Chroma + spaCy, aren't pulled in.

Then run `hermes memory setup`, select `cortexlayer` as the provider, and paste an API key
(create one at [cortexlayer.net](https://www.cortexlayer.net) under Keys).

(A manual directory-plugin install still works if you're running against a fork/local checkout
of this repo instead of the catalog: clone it and copy `plugins/memory/cortexlayer/` into
`$HERMES_HOME/plugins/cortexlayer/` — the directory name must match `cortexlayer` exactly,
since that's what `memory.provider: cortexlayer` in config.yaml resolves against.)

## Configuration

- `api_key` (**required**, secret): your CortexLayer API key. Marked `secret: True`, so per
  the Hermes dev guide it's written to `~/.hermes/.env` as `CORTEX_API_KEY` rather than
  handled by this plugin's `save_config`. `CortexClient` (and this plugin's `is_available()`)
  both read that same env var — the same convention `cortexlayer-python` already uses
  outside Hermes.

No other configuration for the MVP: no `base_url` override (self-hosted `cortex-backend`
deployments aren't a supported mode here yet), no identity/user-mapping config — the API key
*is* the identity (see below).

## Design decisions (task 0088, locked 2026-09-26)

Full reasoning lives in `cortex-layer`'s `tasks/0088-hermes-memory-provider-plugin.md`.
Summary:

- **Distribution**: this repo, installed via Hermes' plugin catalog (git-clone at a pinned
  commit SHA), not a pip package — Hermes itself isn't pip-installable, and a pip entry
  point would need `cortexlayer` installed into Hermes' own (often Dockerized) Python
  environment.
- **Platform mode only, no identity mapping needed**: unlike mem0's Hermes integration
  (Platform/self-hosted-dashboard/OSS — the first two need a key, OSS doesn't), this plugin
  implements Platform only for the MVP. That retires an earlier design pass entirely: task
  0088 originally locked a `user_id` → `user_id_alt` → `user_name` precedence chain for
  mapping Hermes' gateway identity onto Cortex's `user_id` (for a since-dropped embedded-
  engine mode). It doesn't apply to Platform mode — per cortex-backend's own locked rule
  (task 0026), *"Key-derived user authoritative; conflicting `user_id` param → error"*, and
  `CortexClient`'s methods don't even take a `user_id` argument. One configured API key *is*
  the identity; Hermes' gateway fields (`user_id`, `gateway_session_key`, `chat_id`, etc.)
  play no role. Multi-tenant Hermes setups get isolation for free from different profiles
  configuring different keys, not from anything this plugin does.
- **`on_session_end`**: implemented — triggers a final `relink` pass.
- **`on_pre_compress`**: implemented via Hermes' v2 checkpoint API (normalized direct
  evidence messages) — this is the direct answer to Cortex's pitch of not losing what
  Hermes' own context compaction would otherwise drop. Idempotent by content digest, stored
  under `$HERMES_HOME/cortex/checkpoint_digests.json`.
- **`system_prompt_block`**: explicitly deferred, not silently skipped — no compelling
  static prompt content yet.
- **Agent/session provenance tagging** (`agent_identity`/`chat_id` as metadata on stored
  pages, never a retrieval filter): a good idea, but needs a small `cortexlayer` engine
  change. Filed separately as `cortex-layer` task 0090 (written against the embedded-engine
  mode; would need revisiting for Platform mode's REST API). Not part of this plugin's v1.

## Development

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Tests run against a `CortexClient` wired to `httpx.MockTransport` (fake HTTP responses
matching `cortex-backend`'s actual `/v1/*` shapes — no real network, no live server needed)
through a local test double for Hermes' `agent.memory_provider`/`agent.memory_manager` (see
`tests/stub_hermes/` — **not** the real Hermes package; Hermes ships via a shell installer,
not PyPI, so there is nothing to depend on for CI). That stub reproduces the dev guide's
documented contract closely enough to exercise this plugin's own logic; real verification
against Hermes' actual base class and the real hosted API happens in a live end-to-end pass,
done 2026-09-26 (manual directory install) and again 2026-09-27 (real catalog install).

## License

Apache-2.0, matching `cortexlayer-python`.
