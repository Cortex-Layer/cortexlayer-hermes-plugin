# cortexlayer-hermes-plugin

A native [Hermes](https://hermes-agent.nousresearch.com/) `MemoryProvider` plugin for
[CortexLayer](https://www.cortexlayer.net) — the deeper integration surface Hermes calls
into every turn (prefetch before the LLM call, sync after), instead of the generic MCP tool
registration ([`cortex-backend`](https://github.com/Cortex-Layer/cortexlayer-backend)'s
`mcp_server.py`, which stays available and unaffected for MCP-generic clients like the
ChatGPT connector).

This is a thin adapter over [`cortexlayer.Memory`](https://github.com/Cortex-Layer/cortexlayer-python)
(the embedded engine, facts backend) — no storage/retrieval logic lives here.

## Status

Alpha (v0.1.0) — implements the required lifecycle, the 5 memory tools, and the
`on_session_end`/`on_pre_compress` hooks (see "Design decisions" below). Not yet submitted
to Hermes' plugin catalog; not yet verified against a live Hermes instance end to end.

## Install

Once published to Hermes' plugin catalog:

```
hermes plugins install cortex
```

Until then, install manually as a directory plugin: clone this repo's `plugins/memory/cortex/`
into `$HERMES_HOME/plugins/cortex/` (or `./.hermes/plugins/cortex/` for a project-local
install), and make sure `cortexlayer[local]` is importable in whatever Python environment
Hermes runs in:

```
pip install "cortexlayer[local]"
```

**Unverified**: whether `hermes plugins install` resolves a plugin's own Python dependencies
automatically. Confirm this against a real Hermes instance before assuming the manual
`pip install` step above is (or isn't) necessary.

Then run `hermes memory setup` and select `cortex` as the provider.

## Configuration

- `default_user_id` (optional, not secret): the Cortex identity to use when Hermes gives no
  gateway `user_id`/`user_id_alt`/`user_name` (e.g. a bare CLI session with no messaging
  gateway). Required for those platforms — the provider refuses to activate rather than
  guessing an identity if this is unset and no gateway identity is present.

## Design decisions (task 0088, locked 2026-09-26)

Full reasoning lives in `cortex-layer`'s `tasks/0088-hermes-memory-provider-plugin.md`.
Summary:

- **Distribution**: this repo, installed via Hermes' plugin catalog (git-clone at a pinned
  commit SHA), not a pip package — Hermes itself isn't pip-installable, and a pip entry
  point would need `cortexlayer` installed into Hermes' own (often Dockerized) Python
  environment.
- **Identity mapping**: Cortex's `user_id` resolves from Hermes' `user_id` → `user_id_alt` →
  `user_name` (person-identity fields, in that order), sanitized to Cortex's user_id
  charset. **Never** `gateway_session_key`/`chat_id`/`session_id` — those identify a
  conversation, not a person, and using them would silently fragment one person's memory
  into a new "user" every session (modeled on how mem0 keeps `user_id`/`agent_id`/`run_id`
  as separate axes rather than inferring identity from a session handle).
- **`on_session_end`**: implemented — triggers a final `Memory.relink` pass.
- **`on_pre_compress`**: implemented via Hermes' v2 checkpoint API (normalized direct
  evidence messages) — this is the direct answer to Cortex's pitch of not losing what
  Hermes' own context compaction would otherwise drop. Idempotent by content digest, stored
  under `$HERMES_HOME/cortex/checkpoint_digests.json`.
- **`system_prompt_block`**: explicitly deferred, not silently skipped — no compelling
  static prompt content yet.
- **Agent/session provenance tagging** (`agent_identity`/`chat_id` as metadata on stored
  pages, never a retrieval filter): a good idea, but needs a small `cortexlayer` engine
  change (the `Page` schema has no generic tag field today). Filed separately as
  `cortex-layer` task 0090; not part of this plugin's v1.

## Development

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Tests run against the `raw` cortexlayer backend (no LLM/network needed) through a local test
double for Hermes' `agent.memory_provider`/`agent.memory_manager` (see
`tests/stub_hermes/` — **not** the real Hermes package; Hermes ships via a shell installer,
not PyPI, so there is nothing to depend on for CI). That stub reproduces the dev guide's
documented contract closely enough to exercise this plugin's own logic, but real
verification against Hermes' actual base class only happens in a live end-to-end pass — not
yet done for this repo.

## License

Apache-2.0, matching `cortexlayer-python`.
