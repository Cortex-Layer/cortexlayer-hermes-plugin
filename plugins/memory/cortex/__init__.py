"""CortexLayer memory-provider plugin for Hermes (task 0088).

Wraps ``cortexlayer.Memory`` (facts backend) as a Hermes ``MemoryProvider`` —
the deeper integration surface Hermes calls into every turn (prefetch before
the LLM call, sync after), separate from the generic MCP tool registration
Cortex's own server (``cortex-backend``'s ``mcp_server.py``) already exposes
for MCP-generic clients. That server is untouched by this plugin; see the
"Coexistence" note in cortexlayer-layer's tasks/0088 for why both can exist
without conflict as long as one Hermes profile doesn't enable both at once.

No new storage/retrieval logic lives here — every method below is a thin
translation onto ``Memory.add``/``search``/``relink``/``update``/``delete``.

Two things this module cannot verify without a live Hermes instance (flagged
rather than silently assumed — confirm during the manual end-to-end pass):
  - the exact ``get_tool_schemas()``/``handle_tool_call`` return shape (no
    example is published in the dev guide; this follows the near-universal
    ``{name, description, parameters: <json-schema>}`` convention also used
    by cortex-backend's own MCP tools)
  - whether ``hermes plugins install`` resolves this repo's declared
    ``cortexlayer`` dependency automatically, or whether that needs a manual
    ``pip install`` into Hermes' own environment (see repo README)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, spawn_context_thread

from cortexlayer import Memory
from cortexlayer.errors import CortexError, NotFoundError

_USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}[A-Za-z0-9]?$")
_CONFIG_FILENAME = "config.json"
_CHECKPOINT_FILENAME = "checkpoint_digests.json"


class CortexProviderError(RuntimeError):
    """Raised instead of silently picking an identity, backend, or path."""


def sanitize_user_id(raw: str) -> Optional[str]:
    """Deterministically map an arbitrary Hermes identity string onto
    Cortex's ``user_id`` charset (``^[A-Za-z0-9][A-Za-z0-9_-]{0,62}
    [A-Za-z0-9]?$``, cortex-layer task 0026 Sec 8.4): strip disallowed
    characters, trim non-alnum edges, cap at 64 chars.

    Returns ``None`` if nothing usable remains, so the caller falls through
    to the next field in the identity precedence chain. This is a fixed,
    documented transform of an *opaque gateway identity Hermes hands us
    automatically* — not the "invalid user_id -> error, never silent
    fallback" rule from 0026, which is about a *caller-typed literal*
    ``user_id`` param a human or client explicitly asserts. Nothing here
    reintroduces the collision risk that rule guards against: the same raw
    string always sanitizes to the same output.
    """
    if not raw:
        return None
    kept = re.sub(r"[^A-Za-z0-9_-]", "", raw)[:64]
    trimmed = kept.strip("_-")
    assert not trimmed or _USER_ID_RE.fullmatch(trimmed), trimmed  # by construction
    return trimmed or None


def resolve_user_id(identity: Dict[str, Any], configured_default: Optional[str]) -> str:
    """Task 0088 design lock (Q2): person-identity fields only, in this
    precedence, never a session/chat handle (``gateway_session_key``,
    ``chat_id``, ``session_id`` are conversation identifiers, not a person —
    see the task file's mem0-comparison reasoning). Raises rather than
    inventing an identity when nothing resolves."""
    for field in ("user_id", "user_id_alt", "user_name"):
        candidate = sanitize_user_id(str(identity.get(field) or ""))
        if candidate:
            return candidate
    if configured_default:
        candidate = sanitize_user_id(str(configured_default))
        if candidate:
            return candidate
    raise CortexProviderError(
        "cortex memory provider: no usable identity for this session — Hermes gave no "
        "user_id/user_id_alt/user_name, and no default_user_id is configured for this "
        "profile. Set one via the plugin's config (get_config_schema) before using a "
        "platform with no gateway identity (e.g. bare CLI)."
    )


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
    """Adapter over ``cortexlayer.Memory`` (facts backend) for Hermes'
    memory-provider slot."""

    name = "cortex"

    #: Overridable by tests/subclasses to avoid needing a live LLM — v1 ships
    #: against "facts" per task 0088's acceptance criteria.
    _backend = "facts"

    #: Opts into Hermes' v2 pre-compress checkpoint API (normalized direct
    #: evidence messages, not raw OpenAI-style ones) — see on_pre_compress.
    pre_compress_checkpoint_api_version = 2

    def __init__(self) -> None:
        self._memory: Optional[Memory] = None
        self._uid: Optional[str] = None
        self._data_dir: Optional[Path] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._config: Dict[str, Any] = {}

    # --- lifecycle -------------------------------------------------------

    def is_available(self) -> bool:
        """No network calls — just confirm the local engine is importable."""
        try:
            import cortexlayer  # noqa: F401
        except ImportError:
            return False
        return True

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
        self._config = _load_json(self._data_dir / _CONFIG_FILENAME)
        self._uid = resolve_user_id(kwargs, self._config.get("default_user_id"))
        self._memory = Memory(data_dir=str(self._data_dir), backend=self._backend)

    def shutdown(self) -> None:
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=10)
            self._sync_thread = None

    # --- config ------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "default_user_id",
                "description": (
                    "Cortex identity to use when Hermes gives no gateway "
                    "user_id/user_id_alt/user_name (e.g. plain CLI sessions with no "
                    "messaging-gateway identity). Required for those platforms."
                ),
                "secret": False,
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        data_dir = Path(hermes_home) / "cortex"
        data_dir.mkdir(parents=True, exist_ok=True)
        current = _load_json(data_dir / _CONFIG_FILENAME)
        current.update({k: v for k, v in values.items() if v is not None})
        (data_dir / _CONFIG_FILENAME).write_text(json.dumps(current, indent=2))
        self._config = current

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
        mem = self._require_memory()
        try:
            if tool_name == "memory_add":
                result = mem.add(args["text"], user_id=self._uid, timestamp=args.get("timestamp"))
                return {"page_ids": result.page_ids}
            if tool_name == "memory_retrieve":
                hits = mem.search(args["query"], user_id=self._uid, limit=int(args.get("k", 4)))
                return {"passages": [asdict(h) for h in hits]}
            if tool_name == "memory_relink":
                return mem.relink(user_id=self._uid)
            if tool_name == "memory_update":
                mem.update(args["page_id"], args["text"], user_id=self._uid)
                return {"ok": True}
            if tool_name == "memory_delete":
                mem.delete(args["page_id"], user_id=self._uid)
                return {"ok": True}
        except NotFoundError as e:
            return {"error": str(e)}
        raise CortexProviderError(f"cortex memory provider: unknown tool {tool_name!r}")

    # --- automatic per-turn hooks -------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> Optional[str]:
        """Cheap by default (search, no LLM compression call) — this runs
        every turn, unlike the explicit memory_retrieve tool where the
        caller opts into ``compress``."""
        mem = self._require_memory()
        hits = mem.search(query, user_id=self._uid, limit=4)
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
        mem = self._require_memory()
        uid = self._uid
        text = f"User: {user}\nAssistant: {assistant}"

        def _sync() -> None:
            mem.add(text, user_id=uid)

        self._sync_thread = spawn_context_thread(_sync, name="cortex-provider-sync")
        self._sync_thread.start()

    def on_session_end(self, messages: list) -> None:
        """Explicit decision (task 0088): implemented — final relink pass.
        Best-effort; a failed relink shouldn't crash session teardown."""
        mem = self._require_memory()
        try:
            mem.relink(user_id=self._uid)
        except CortexError:
            pass

    def on_pre_compress(self, messages: list, *, require_checkpoint: bool = False) -> str:
        """Explicit decision (task 0088): implemented via the v2 checkpoint
        API — this is the direct answer to Cortex's pitch of not losing what
        Hermes' own compaction would drop. Idempotent by content digest
        (guide's requirement) so retries don't double-write."""
        mem = self._require_memory()
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
                result = mem.add(text, user_id=self._uid)
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
        shows up (e.g. surfacing the configured user_id for debugging)."""
        return None

    # --- internals -----------------------------------------------------

    def _require_memory(self) -> Memory:
        if self._memory is None:
            raise CortexProviderError(
                "cortex memory provider: called before initialize() (or after shutdown())"
            )
        return self._memory


def register(ctx: Any) -> None:
    """Hermes plugin discovery entry point."""
    ctx.register_memory_provider(CortexMemoryProvider())
