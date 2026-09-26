"""Unit tests for the Cortex Hermes memory-provider plugin.

Runs against the "raw" cortexlayer backend (no LLM/network needed) rather
than the "facts" backend the plugin ships with by default — fact-extraction
quality is cortex-backend's own test suite's concern, not this adapter's.
The MemoryManager integration test below follows the dev guide's exact
documented pattern (add_provider/initialize_all/handle_tool_call/sync_all/
shutdown_all).
"""

import json

import pytest

import cortex as plugin  # plugins/memory/cortex, see conftest.py's sys.path setup
from agent.memory_manager import MemoryManager  # test stub — see tests/stub_hermes


class _RawCortexMemoryProvider(plugin.CortexMemoryProvider):
    """Same adapter, raw backend — no LLM required for these tests."""

    _backend = "raw"


# --- sanitize_user_id / resolve_user_id (pure functions) -------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("alice", "alice"),
        ("Alice_92", "Alice_92"),
        ("alice@example.com", "aliceexamplecom"),
        ("-_alice-_", "alice"),
        ("___", None),
        ("", None),
        ("a" * 100, "a" * 64),
    ],
)
def test_sanitize_user_id(raw, expected):
    assert plugin.sanitize_user_id(raw) == expected


def test_resolve_user_id_precedence_user_id_wins():
    identity = {"user_id": "alice", "user_id_alt": "alice2", "user_name": "Alice Smith"}
    assert plugin.resolve_user_id(identity, configured_default=None) == "alice"


def test_resolve_user_id_falls_back_through_chain():
    assert plugin.resolve_user_id({"user_name": "Bob Jones"}, configured_default=None) == "BobJones"
    assert plugin.resolve_user_id({}, configured_default="fallback-user") == "fallback-user"


def test_resolve_user_id_never_uses_session_handles():
    """Task 0088 Q2 lock: gateway_session_key/chat_id/session_id are never
    identity sources, even when they're the only fields present."""
    identity = {
        "gateway_session_key": "chat-9182",
        "chat_id": "9182",
        "session_id": "sess-abc",
    }
    with pytest.raises(plugin.CortexProviderError):
        plugin.resolve_user_id(identity, configured_default=None)


def test_resolve_user_id_raises_without_any_identity():
    with pytest.raises(plugin.CortexProviderError):
        plugin.resolve_user_id({}, configured_default=None)


# --- lifecycle ---------------------------------------------------------


def test_initialize_requires_hermes_home():
    provider = _RawCortexMemoryProvider()
    with pytest.raises(plugin.CortexProviderError):
        provider.initialize("session-1", user_id="alice")


def test_initialize_and_shutdown(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="alice")
    assert provider._uid == "alice"
    assert (tmp_path / "cortex").is_dir()
    provider.shutdown()


def test_config_roundtrip(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.save_config({"default_user_id": "carol"}, hermes_home=str(tmp_path))
    stored = json.loads((tmp_path / "cortex" / "config.json").read_text())
    assert stored == {"default_user_id": "carol"}

    # A later initialize() with no gateway identity picks up the saved default.
    provider.initialize("session-2", hermes_home=str(tmp_path))
    assert provider._uid == "carol"
    provider.shutdown()


# --- tools ---------------------------------------------------------------


def test_tool_schemas_mirror_the_five_mcp_tools():
    provider = _RawCortexMemoryProvider()
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert names == {"memory_add", "memory_retrieve", "memory_relink", "memory_update", "memory_delete"}


def test_handle_tool_call_full_loop(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="dana")

    added = provider.handle_tool_call("memory_add", {"text": "Dana moved to Lisbon in March."})
    page_id = added["page_ids"][0]

    retrieved = provider.handle_tool_call("memory_retrieve", {"query": "Where did Dana move?"})
    assert any(p["id"] == page_id for p in retrieved["passages"])

    relinked = provider.handle_tool_call("memory_relink", {})
    assert "pages" in relinked

    updated = provider.handle_tool_call(
        "memory_update", {"page_id": page_id, "text": "Dana moved to Porto in March."}
    )
    assert updated == {"ok": True}

    deleted = provider.handle_tool_call("memory_delete", {"page_id": page_id})
    assert deleted == {"ok": True}

    missing = provider.handle_tool_call("memory_delete", {"page_id": page_id})
    assert "error" in missing

    provider.shutdown()


def test_handle_tool_call_unknown_tool_raises(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="dana")
    with pytest.raises(plugin.CortexProviderError):
        provider.handle_tool_call("not_a_real_tool", {})
    provider.shutdown()


# --- automatic per-turn hooks ------------------------------------------


def test_sync_turn_is_nonblocking_and_persists(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="erin")

    before = provider._memory.count(user_id="erin")
    provider.sync_turn("What's the capital of France?", "Paris.")
    provider.shutdown()  # joins the spawned thread
    after = provider._memory.count(user_id="erin")

    assert after > before


def test_prefetch_returns_none_on_empty_store(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="frank")
    assert provider.prefetch("anything") is None
    provider.shutdown()


def test_on_session_end_relinks_without_crashing(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="gina")
    provider.handle_tool_call("memory_add", {"text": "Gina adopted a cat named Mochi."})
    provider.on_session_end([])  # must not raise
    provider.shutdown()


def test_on_pre_compress_is_idempotent(tmp_path):
    provider = _RawCortexMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="hank")
    messages = [
        {"role": "user", "content": "I just got a new job at Acme Corp."},
        {"role": "assistant", "content": "Congratulations on the new job!"},
    ]

    first = provider.on_pre_compress(messages)
    before = provider._memory.count(user_id="hank")
    second = provider.on_pre_compress(messages)  # same content -> should not re-add
    after = provider._memory.count(user_id="hank")

    assert first.startswith("checkpoint:")
    assert after == before
    provider.shutdown()


def test_system_prompt_block_deferred_not_silent():
    """Task 0088: explicit decision to defer, not an accidental no-op."""
    provider = _RawCortexMemoryProvider()
    assert provider.system_prompt_block() is None


# --- MemoryManager integration (dev guide's documented test pattern) -----


def test_memory_manager_flow(tmp_path):
    mgr = MemoryManager()
    mgr.add_provider(_RawCortexMemoryProvider())
    mgr.initialize_all(session_id="test-1", platform="cli", hermes_home=str(tmp_path), user_id="ivy")

    result = mgr.handle_tool_call("memory_add", {"text": "Ivy's favorite color is teal."})
    assert result["page_ids"]

    mgr.sync_all("Tell me something else about Ivy.", "Noted.")
    mgr.on_session_end([])
    mgr.shutdown_all()
