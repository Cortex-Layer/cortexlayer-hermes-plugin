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
modes (Platform/self-hosted-dashboard/OSS) rather than all three.

This also retires task 0088's original Q2 identity-mapping design entirely
(session→user_id precedence, sanitization, etc.). It doesn't apply here: per
cortex-backend's own locked rule (task 0026, API-key addendum), *"Key-derived
user authoritative; conflicting user_id param → error"* — ``CortexClient``'s
methods don't even take a ``user_id`` parameter. One configured API key *is*
the identity; Hermes' gateway identity fields play no role here.

No new storage/retrieval logic lives here — every method is a thin
translation onto ``CortexClient.add``/``search``/``relink``/``update``/
``delete``.

**Verified against a real Hermes install (2026-09-26)** — the base class and
manager were read directly from a live ``nousresearch/hermes-agent`` image,
not just the dev guide, which caught several real bugs the guide's summary
didn't surface:
  - ``handle_tool_call`` must return a **JSON string**, not a dict —
    ``MemoryManager.handle_tool_call`` passes the return value straight
    through with no conversion.
  - The dev guide's ``agent.memory_provider.spawn_context_thread`` does not
    exist in the real package (only a Honcho-plugin-local helper of the same
    name does) — importing it would have raised ``ImportError`` at plugin
    load. Not needed anyway: ``MemoryManager.sync_all`` already dispatches
    every provider's ``sync_turn`` on its own background executor (with
    contextvars propagated), so a provider's ``sync_turn`` can simply call
    the client inline.
  - ``system_prompt_block``/``prefetch`` default to ``""``, not ``None``
    (the manager's own callers guard against ``None`` too, so this wasn't
    a crash, just non-compliant).
  - ``initialize()`` also receives ``agent_context`` ("primary", "subagent",
    "cron", "flush") and ``platform``; the shipped Honcho plugin skips
    activation entirely for ``agent_context in {"cron", "flush"}`` or
    ``platform == "cron"`` ("cron system prompts would corrupt user
    representations") — copied that exact gate here.
  - Tool schema shape (bare ``{name, description, parameters}``) was
    confirmed correct as originally guessed.
  - **The on-disk directory name IS the activation key** —
    ``plugins/memory/__init__.py``'s ``find_provider_dir()`` does a literal
    ``directory / name`` join against ``memory.provider`` in config.yaml (and
    the entry-point path matches ``entry_point.name == name`` the same way);
    the class's ``name`` attribute plays no part in lookup. An earlier version
    of this file kept the directory named ``cortex`` while displaying
    ``cortexlayer`` as the class's ``name``, reasoning (wrongly, it turns out)
    that Hermes might naively ``sys.path``-import plugins by directory name
    and collide with this file's own ``from cortexlayer import CortexClient``.
    That's not how it loads plugins at all — ``plugins/memory/__init__.py``
    dynamically constructs a namespaced module (``_hermes_user_memory.<name>``
    via ``importlib.util.module_from_spec``), so there was never a collision
    risk. But it *did* mean `memory.provider: cortexlayer` couldn't find a
    directory literally named ``cortex`` — confirmed live via `hermes memory
    status` reporting "Plugin: NOT installed" while listing our plugin under
    its folder name. Fixed: this directory is now ``plugins/memory/
    cortexlayer/``, matching the catalog name everywhere.

Still unverified: the exact plugin-catalog submission mechanics beyond what
Hermes' docs describe, and whether ``hermes plugins install`` resolves this
repo's declared ``cortexlayer`` dependency automatically or needs a manual
``pip install``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus

from cortexlayer import CortexClient
from cortexlayer.client import API_KEY_ENV
from cortexlayer.errors import CortexConfigError, CortexError

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "checkpoint_digests.json"

#: Contexts in which writes (and activation at all) are skipped — matches
#: the bundled Honcho plugin's own gate exactly: system-triggered runs with
#: no real user on the other end shouldn't write to someone's account.
_SKIP_CONTEXTS = {"cron", "flush"}


class CortexProviderError(RuntimeError):
    """Raised only for a fundamentally malformed host call (missing
    ``hermes_home``) — never for "not configured", which degrades
    gracefully instead (see ``self._client is None`` handling below)."""


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

    #: Matches this directory's own name (`plugins/memory/cortexlayer/`) —
    #: confirmed live that the directory name, not this attribute, is what
    #: `memory.provider` in config.yaml actually resolves against (see the
    #: module docstring's "on-disk directory name IS the activation key"
    #: note). Kept in sync with the directory rather than treated as an
    #: independent display label.
    name = "cortexlayer"

    #: Opts into Hermes' v2 pre-compress checkpoint API (normalized direct
    #: evidence messages, not raw OpenAI-style ones) — see on_pre_compress.
    pre_compress_checkpoint_api_version = 2

    def __init__(self) -> None:
        self._client: Optional[CortexClient] = None
        self._data_dir: Optional[Path] = None
        self._last_recall_count: Optional[int] = None

    # --- lifecycle -------------------------------------------------------

    def is_available(self) -> bool:
        """No network calls — mirrors the dev guide's own example: check the
        configured key exists, don't validate it (that's a network call)."""
        return bool(os.environ.get(API_KEY_ENV))

    def unavailable_reason(self) -> str:
        if os.environ.get(API_KEY_ENV):
            return ""
        return (
            "No CortexLayer API key configured — run `hermes memory setup` and "
            "create one at https://www.cortexlayer.net under Keys."
        )

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

        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "cli")
        if agent_context in _SKIP_CONTEXTS or platform == "cron":
            logger.debug(
                "cortex memory provider skipped: %s context (agent_context=%s, platform=%s)",
                "system-triggered", agent_context, platform,
            )
            self._client = None
            return

        try:
            self._client = self._build_client()
        except CortexConfigError as e:
            # is_available() already gates registration on the key existing,
            # so this should be rare — but never crash agent init over it.
            logger.warning("cortex memory provider: failed to build client: %s", e)
            self._client = None

    def _build_client(self) -> CortexClient:
        """Separate method so tests can override it with a mocked transport
        instead of hitting the real hosted API."""
        return CortexClient()

    def shutdown(self) -> None:
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
        field (``api_key``) is ``secret: True`` with ``env_var`` set, so per
        the dev guide Hermes writes it to ``.env`` itself. Env-vars-only
        providers are explicitly allowed to leave this as a no-op."""

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

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Must return a JSON string — ``MemoryManager.handle_tool_call``
        passes this straight through with no conversion of its own."""
        if self._client is None:
            return json.dumps({
                "error": "CortexLayer memory provider is not active (no API key "
                         "configured, or this is a system-triggered run)."
            })
        client = self._client
        try:
            if tool_name == "memory_add":
                result = client.add(args["text"], timestamp=args.get("timestamp"))
                return json.dumps({"page_ids": result.page_ids})
            if tool_name == "memory_retrieve":
                hits = client.search(args["query"], limit=int(args.get("k", 4)))
                return json.dumps({"passages": [asdict(h) for h in hits]})
            if tool_name == "memory_relink":
                return json.dumps(client.relink())
            if tool_name == "memory_update":
                client.update(args["page_id"], args["text"])
                return json.dumps({"ok": True})
            if tool_name == "memory_delete":
                client.delete(args["page_id"])
                return json.dumps({"ok": True})
        except CortexError as e:
            # Network/server errors surface to the agent as a tool result
            # rather than crashing the turn (RateLimitError, ServerError,
            # ConnectionError, etc. are all real possibilities now that this
            # is a hosted HTTP call, not an in-process one).
            return json.dumps({"error": str(e)})
        raise CortexProviderError(f"cortex memory provider: unknown tool {tool_name!r}")

    # --- automatic per-turn hooks -------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """A real network call every turn now (Platform mode) — same
        latency tradeoff mem0's own cloud mode has. Hermes already runs
        this per-provider on its own background thread with an 8s timeout
        (skipping the turn if a prior call is still in flight), so a
        blocking call here is within the host's own contract."""
        if self._client is None:
            self._last_recall_count = None
            return ""
        try:
            hits = self._client.search(query, limit=4)
        except CortexError:
            self._last_recall_count = None
            return ""  # best-effort: a failed prefetch shouldn't block the turn
        self._last_recall_count = len(hits) or None
        if not hits:
            return ""
        return "\n".join(f"- {h.text or h.snippet}" for h in hits)

    def recall_status(self) -> Optional[RecallStatus]:
        if self._last_recall_count is None:
            return None
        return RecallStatus(provider_label="CortexLayer", count=self._last_recall_count)

    def sync_turn(
        self,
        user: str,
        assistant: str,
        *,
        session_id: str = "",
        messages: Optional[list] = None,
    ) -> None:
        """No extra threading needed here: ``MemoryManager.sync_all``
        already dispatches every provider's ``sync_turn`` on its own
        background worker (with contextvars propagated for profile
        isolation) — a provider calling the client inline already satisfies
        the "must be non-blocking" contract from the host's side."""
        if self._client is None:
            return
        text = f"User: {user}\nAssistant: {assistant}"
        try:
            self._client.add(text)
        except CortexError:
            pass  # fire-and-forget: no channel back into the conversation

    def on_session_end(self, messages: list) -> None:
        """Explicit decision (task 0088): implemented — final relink pass.
        Best-effort; a failed relink shouldn't crash session teardown."""
        if self._client is None:
            return
        try:
            self._client.relink()
        except CortexError:
            pass

    def on_pre_compress(self, messages: list, *, require_checkpoint: bool = False) -> str:
        """Explicit decision (task 0088): implemented via the v2 checkpoint
        API — this is the direct answer to Cortex's pitch of not losing what
        Hermes' own compaction would drop. Idempotent by content digest
        (guide's requirement) so retries don't double-write. Returns text
        for the compression summary prompt (or "" to contribute nothing —
        never a bare "checkpoint: []" when there's nothing new)."""
        if self._client is None:
            return ""
        client = self._client
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
        if not new_ids:
            return ""
        digest_path.write_text(json.dumps({"digests": sorted(seen)}))
        return f"checkpoint: {new_ids}"

    def system_prompt_block(self) -> str:
        """Explicit decision (task 0088): deferred, not silently skipped —
        no compelling static prompt content yet; revisit if a real need
        shows up (e.g. surfacing which CortexLayer account is active)."""
        return ""


def register(ctx: Any) -> None:
    """Hermes plugin discovery entry point."""
    ctx.register_memory_provider(CortexMemoryProvider())
