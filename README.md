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
"Design decisions" below). Catalog PR open, not yet merged (Cortex-Layer/cortexlayer-hermes-plugin
against NousResearch/hermes-agent, #124341); not yet verified
against a live Hermes instance end to end.

## Install

Once published to Hermes' plugin catalog:

```
hermes plugins install cortexlayer
```

Until then, install manually as a directory plugin: clone this repo's
`plugins/memory/cortexlayer/` into `$HERMES_HOME/plugins/cortexlayer/` (or
`./.hermes/plugins/cortexlayer/` for a project-local install — the directory name must match
`cortexlayer` exactly, since that's what `memory.provider: cortexlayer` in config.yaml
actually resolves against; confirmed live via `hermes memory status`, which reports "Plugin:
NOT installed" if the folder name doesn't match), and make sure `cortexlayer` is importable
in whatever Python environment Hermes runs in:

```
pip install cortexlayer
```

(No `[local]` extra needed — Platform mode only talks HTTP via `CortexClient`, which depends
on nothing but `httpx`. The embedded engine's heavier deps, Chroma + spaCy, aren't pulled in.)

**Unverified**: whether `hermes plugins install` resolves a plugin's own Python dependencies
automatically. Confirm this against a real Hermes instance before assuming the manual
`pip install` step above is (or isn't) necessary.

Then run `hermes memory setup`, select `cortexlayer` as the provider, and paste an API key
(create one at [cortexlayer.net](https://www.cortexlayer.net) under Keys).

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
documented contract closely enough to exercise this plugin's own logic, but real
verification against Hermes' actual base class, and against the real hosted API, only
happens in a live end-to-end pass — not yet done for this repo.

## License

Apache-2.0, matching `cortexlayer-python`.
