"""CortexLayer memory-provider plugin for Hermes (task 0088) — Platform mode.

Wraps ``cortexlayer.CortexClient`` (the hosted API client, API-key
authenticated, default ``https://api.cortexlayer.net``) as a Hermes
``MemoryProvider`` — the deeper integration surface Hermes calls into every
turn (prefetch before the LLM call, sync after), separate from the generic
MCP tool registration Cortex's own server (``cortex-backend``'s
``mcp_server.py``) already exposes for MCP-generic clients. That server is
untouched by this plugin.

**MVP scope decision (2026-09-26, Miguel)**: Platform mode only — no
embedded/local-engine ("OSS") mode, matching one of mem0's three Hermes
modes (Platform/self-hosted-dashboard/OSS) rather than all three. An earlier
version of this file wrapped the embedded ``cortexlayer.Memory`` engine
directly (no key, fully local); that's gone for now, not merged in.

This also retires task 0088's original Q2 identity-mapping design entirely
(session→user_id precedence, sanitization, etc. — see git history if that
ever needs reviving for an OSS mode). It doesn't apply here: per
cortex-backend's own locked rule (task 0026, API-key addendum), *"Key-derived
user authoritative; conflicting user_id param → error"* — ``CortexClient``'s
methods don't even take a ``user_id`` parameter. One configured API key *is*
the identity; Hermes' gateway identity fields (user_id/user_id_alt/
gateway_session_key/etc.) play no role here. Multi-tenant Hermes setups get
isolation for free from different profiles configuring different keys, not
from anything this plugin does.

No new storage/retrieval logic lives here — every method is a thin
translation onto ``CortexClient.add``/``search``/``relink``/``update``/
``delete``.

Unverified without a live Hermes instance (flagged, not assumed):
  - the exact ``get_tool_schemas()``/``handle_tool_call`` return shape (no
    example is published in the dev guide; this follows the near-universal
    ``{name, description, parameters: <json-schema>}`` convention also used
    by cortex-backend's own MCP tools)
  - whether a ``secret: True`` config field's value reliably reaches this
    process as the ``env_var`` name given in ``get_config_schema`` (assumed
    here, matching the guide's own ``os.environ.get(...)`` example for
    ``is_available()``) before ``hermes plugins install`` resolves this
    repo's declared ``cortexlayer`` dependency automatically, or whether
    that needs a manual ``pip install`` into Hermes' own environment
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, spawn_context_thread

from cortexlayer import CortexClient
from cortexlayer.client import API_KEY_ENV
from cortexlayer.errors import CortexConfigError, CortexError, NotFoundError

_CHECKPOINT_FILENAME = "checkpoint_digests.json"


class CortexProviderError(RuntimeError):
    """Raised instead of silently proceeding without a usable client."""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _message_text(message: Any) -> str:
    """Best-effort text extraction from a v2 pre-compress message (Hermes'
    normalized shape: role + text content, tool calls already filtered)."""
    if isinstance(message, dict):
        role = message.get("role", "")
        content = message.get("content", "")
        if isinstance(content, list):  # some frameworks send content parts
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return f"{role}: {content}".strip() if content else ""
    return str(message) if message else ""


class CortexMemoryProvider(MemoryProvider):
    """Adapter over ``cortexlayer.CortexClient`` (Platform/hosted mode) for
    Hermes' memory-provider slot."""

    #: Displayed/catalog identifier (what `hermes memory setup` lists and
    #: `hermes plugins install <name>` resolves) — deliberately NOT the same
    #: as this directory's on-disk name (`plugins/memory/cortex/`). If
    #: Hermes' loader ever adds `plugins/memory/` to sys.path and imports by
    #: directory name, a directory literally named `cortexlayer` would
    #: collide with this file's own `from cortexlayer import CortexClient`
    #: below — it could shadow the real engine package with itself. The
    #: directory stays `cortex`; everything user-facing (this name,
    #: plugin.yaml, catalog entry, README) says `cortexlayer`.
    name = "cortexlayer"

    #: Opts into Hermes' v2 pre-compress checkpoint API (normalized direct
    #: evidence messages, not raw OpenAI-style ones) — see on_pre_compress.
    pre_compress_checkpoint_api_version = 2

    def __init__(self) -> None:
        self._client: Optional[CortexClient] = None
        self._data_dir: Optional[Path] = None
        self._sync_thread: Optional[threading.Thread] = None

    # --- lifecycle -------------------------------------------------------

    def is_available(self) -> bool:
        """No network calls — mirrors the dev guide's own example: check the
        configured key exists, don't validate it (that's a network call)."""
        return bool(os.environ.get(API_KEY_ENV))

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise CortexProviderError(
                "cortex memory provider: initialize() got no hermes_home — refusing to "
                "fall back to a hardcoded path (breaks profile isolation; see the Hermes "
                "plugin guide's 'Wrong' example)."
            )
        self._data_dir = Path(hermes_home) / "cortex"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = self._build_client()
        except CortexConfigError as e:
            raise CortexProviderError(
                f"cortex memory provider: {e} Run `hermes memory setup` and configure a "
                "CortexLayer API key (create one at https://www.cortexlayer.net under Keys)."
            ) from e

    def _build_client(self) -> CortexClient:
        """Separate method so tests can override it with a mocked transport
        instead of hitting the real hosted API."""
        return CortexClient()

    def shutdown(self) -> None:
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=10)
            self._sync_thread = None
        if self._client is not None:
            self._client.close()

    # --- config ------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "CortexLayer API key (create one at cortexlayer.net under Keys).",
                "secret": True,
                "required": True,
                "env_var": API_KEY_ENV,
                "url": "https://www.cortexlayer.net",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Nothing non-secret to persist for this MVP — the only config
        field (``api_key``) is ``secret: True``, so per the dev guide Hermes
        writes it to ``.env`` itself rather than handing it here."""

    # --- tools (mirrors cortex-backend's 5 MCP tools, task 0006) -----------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_add",
                "description": "Save a fact worth remembering long-term.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "timestamp": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "memory_retrieve",
                "description": "Search remembered facts relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 4},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_relink",
                "description": "Re-run link discovery across recent memories.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "memory_update",
                "description": "Correct a previously stored memory (page_id from search/retrieve).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["page_id", "text"],
                },
            },
            {
                "name": "memory_delete",
                "description": "Remove a previously stored memory (page_id from search/retrieve).",
                "parameters": {
                    "type": "object",
                    "properties": {"page_id": {"type": "string"}},
                    "required": ["page_id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        client = self._require_client()
        try:
            if tool_name == "memory_add":
                result = client.add(args["text"], timestamp=args.get("timestamp"))
                return {"page_ids": result.page_ids}
            if tool_name == "memory_retrieve":
                hits = client.search(args["query"], limit=int(args.get("k", 4)))
                return {"passages": [asdict(h) for h in hits]}
            if tool_name == "memory_relink":
                return client.relink()
            if tool_name == "memory_update":
                client.update(args["page_id"], args["text"])
                return {"ok": True}
            if tool_name == "memory_delete":
                client.delete(args["page_id"])
                return {"ok": True}
        except CortexError as e:
            # Network/server errors surface to the agent as a tool result
            # rather than crashing the turn (RateLimitError, ServerError,
            # ConnectionError, etc. are all real possibilities now that this
            # is a hosted HTTP call, not an in-process one).
            return {"error": str(e)}
        raise CortexProviderError(f"cortex memory provider: unknown tool {tool_name!r}")

    # --- automatic per-turn hooks -------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> Optional[str]:
        """A real network call every turn now (Platform mode) — same
        latency tradeoff mem0's own cloud mode has."""
        client = self._require_client()
        try:
            hits = client.search(query, limit=4)
        except CortexError:
            return None  # best-effort: a failed prefetch shouldn't block the turn
        if not hits:
            return None
        return "\n".join(f"- {h.text or h.snippet}" for h in hits)

    def sync_turn(
        self,
        user: str,
        assistant: str,
        *,
        session_id: str = "",
        messages: Optional[list] = None,
    ) -> None:
        """Must be non-blocking — spawn via Hermes' spawn_context_thread
        (never a bare threading.Thread; preserves contextvars for profile
        isolation, per the dev guide's threading contract)."""
        client = self._require_client()
        text = f"User: {user}\nAssistant: {assistant}"

        def _sync() -> None:
            try:
                client.add(text)
            except CortexError:
                pass  # fire-and-forget: no channel back into the conversation

        self._sync_thread = spawn_context_thread(_sync, name="cortex-provider-sync")
        self._sync_thread.start()

    def on_session_end(self, messages: list) -> None:
        """Explicit decision (task 0088): implemented — final relink pass.
        Best-effort; a failed relink shouldn't crash session teardown."""
        client = self._require_client()
        try:
            client.relink()
        except CortexError:
            pass

    def on_pre_compress(self, messages: list, *, require_checkpoint: bool = False) -> str:
        """Explicit decision (task 0088): implemented via the v2 checkpoint
        API — this is the direct answer to Cortex's pitch of not losing what
        Hermes' own compaction would drop. Idempotent by content digest
        (guide's requirement) so retries don't double-write."""
        client = self._require_client()
        digest_path = self._data_dir / _CHECKPOINT_FILENAME  # type: ignore[union-attr]
        seen = set(_load_json(digest_path).get("digests", []))
        new_ids: List[str] = []
        for message in messages:
            text = _message_text(message)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            try:
                result = client.add(text)
            except CortexError:
                if require_checkpoint:
                    raise
                continue
            new_ids.extend(result.page_ids)
            seen.add(digest)
        digest_path.write_text(json.dumps({"digests": sorted(seen)}))
        return f"checkpoint: {new_ids}"

    def system_prompt_block(self) -> Optional[str]:
        """Explicit decision (task 0088): deferred, not silently skipped —
        no compelling static prompt content yet; revisit if a real need
        shows up (e.g. surfacing which CortexLayer account is active)."""
        return None

    # --- internals -----------------------------------------------------

    def _require_client(self) -> CortexClient:
        if self._client is None:
            raise CortexProviderError(
                "cortex memory provider: called before initialize() (or after shutdown())"
            )
        return self._client


def register(ctx: Any) -> None:
    """Hermes plugin discovery entry point."""
    ctx.register_memory_provider(CortexMemoryProvider())
