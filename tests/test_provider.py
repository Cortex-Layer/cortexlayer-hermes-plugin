"""Unit tests for the Cortex Hermes memory-provider plugin (Platform mode).

Runs against a `CortexClient` wired to `httpx.MockTransport` — a fake HTTP
server matching cortex-backend's actual `/v1/*` response shapes, no real
network or live server needed. The MemoryManager integration test at the
bottom follows the dev guide's documented pattern (add_provider/
initialize_all/handle_tool_call/sync_all/shutdown_all), and the stub's
`MemoryProvider` base class mirrors the real `nousresearch/hermes-agent`
image's contract as read directly from a live container on 2026-09-26 (see
the plugin module's own docstring) — in particular, `handle_tool_call` must
return a JSON **string**, not a dict.
"""

import json
from typing import Dict

import httpx
import pytest

import cortexlayer_hermes_plugin as plugin  # plugins/memory/cortexlayer, loaded by conftest.py
from agent.memory_manager import MemoryManager  # test stub — see tests/stub_hermes
from cortexlayer import CortexClient


class _FakeCortexServer:
    """A minimal in-memory stand-in for cortex-backend's REST API — just
    enough of /v1/pages, /v1/search, /v1/relink to exercise this plugin's
    request/response handling, not a real search engine."""

    def __init__(self) -> None:
        self.pages: Dict[str, str] = {}
        self._next_id = 0
        self.fail_with: "httpx.Response | None" = None  # inject an error response

    def _new_id(self) -> str:
        self._next_id += 1
        return f"page-{self._next_id}"

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.fail_with is not None:
            return self.fail_with
        method, path = request.method, request.url.path
        body = json.loads(request.content) if request.content else {}

        if method == "POST" and path == "/v1/pages":
            page_id = self._new_id()
            self.pages[page_id] = body["text"]
            return httpx.Response(200, json={"page_ids": [page_id]})

        if method == "POST" and path == "/v1/search":
            results = [
                {"id": pid, "title": text[:40], "snippet": text[:200], "score": 1.0,
                 "via": "direct", "text": text}
                for pid, text in self.pages.items()
            ]
            return httpx.Response(200, json={"results": results})

        if method == "PATCH" and path.startswith("/v1/pages/"):
            page_id = path.rsplit("/", 1)[-1]
            if page_id not in self.pages:
                return httpx.Response(404, json={"error": f"page not found: {page_id}"})
            self.pages[page_id] = body["text"]
            return httpx.Response(200, json={})

        if method == "DELETE" and path.startswith("/v1/pages/"):
            page_id = path.rsplit("/", 1)[-1]
            if page_id not in self.pages:
                return httpx.Response(404, json={"error": f"page not found: {page_id}"})
            del self.pages[page_id]
            return httpx.Response(200, json={})

        if method == "POST" and path == "/v1/relink":
            return httpx.Response(200, json={"pages": len(self.pages), "links_written": 0})

        return httpx.Response(500, json={"error": f"unhandled {method} {path}"})


class _MockCortexMemoryProvider(plugin.CortexMemoryProvider):
    """Same adapter, `_build_client` overridden to point at a fake server
    instead of the real hosted API."""

    def __init__(self, server: _FakeCortexServer) -> None:
        super().__init__()
        self._server = server

    def _build_client(self) -> CortexClient:
        transport = httpx.MockTransport(self._server.handle)
        return CortexClient(api_key="test-key", http_client=httpx.Client(transport=transport))


# --- lifecycle ---------------------------------------------------------


def test_is_available_true_when_key_env_set(monkeypatch):
    monkeypatch.setenv("CORTEX_API_KEY", "ctx_test")
    assert plugin.CortexMemoryProvider().is_available() is True


def test_is_available_false_when_no_key(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    assert plugin.CortexMemoryProvider().is_available() is False


def test_unavailable_reason_explains_missing_key(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    assert "API key" in plugin.CortexMemoryProvider().unavailable_reason()
    monkeypatch.setenv("CORTEX_API_KEY", "ctx_test")
    assert plugin.CortexMemoryProvider().unavailable_reason() == ""


def test_initialize_requires_hermes_home():
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    with pytest.raises(plugin.CortexProviderError):
        provider.initialize("session-1")


def test_initialize_and_shutdown(tmp_path):
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize("session-1", hermes_home=str(tmp_path))
    assert (tmp_path / "cortex").is_dir()
    provider.shutdown()


def test_initialize_ignores_unrelated_hermes_identity_kwargs(tmp_path):
    """Platform mode doesn't map any of these onto anything — just confirms
    the extra kwargs Hermes will always pass don't break initialize()."""
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        user_id="alice",
        gateway_session_key="chat-9182",
        chat_id="9182",
        agent_identity="primary",
        platform="cli",
    )
    provider.shutdown()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_context": "cron"},
        {"agent_context": "flush"},
        {"platform": "cron"},
    ],
)
def test_initialize_skips_activation_for_system_contexts(tmp_path, kwargs):
    """Matches the bundled Honcho plugin's own gate exactly: system-
    triggered runs never activate the provider at all."""
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize("session-1", hermes_home=str(tmp_path), **kwargs)
    assert provider._client is None
    # Every method degrades gracefully rather than raising or crashing.
    assert provider.prefetch("anything") == ""
    provider.sync_turn("hi", "hello")  # no-op, must not raise
    provider.on_session_end([])  # no-op, must not raise
    assert provider.on_pre_compress([]) == ""
    assert "error" in json.loads(provider.handle_tool_call("memory_add", {"text": "x"}))
    provider.shutdown()


# --- config --------------------------------------------------------------


def test_get_config_schema_requires_api_key():
    schema = plugin.CortexMemoryProvider().get_config_schema()
    assert len(schema) == 1
    assert schema[0]["key"] == "api_key"
    assert schema[0]["secret"] is True
    assert schema[0]["required"] is True
    assert schema[0]["env_var"] == "CORTEX_API_KEY"


# --- tools ---------------------------------------------------------------


def test_tool_schemas_mirror_the_five_mcp_tools():
    provider = plugin.CortexMemoryProvider()
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert names == {"memory_add", "memory_retrieve", "memory_relink", "memory_update", "memory_delete"}


def test_handle_tool_call_full_loop(tmp_path):
    server = _FakeCortexServer()
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))

    added = json.loads(provider.handle_tool_call("memory_add", {"text": "Dana moved to Lisbon in March."}))
    page_id = added["page_ids"][0]

    retrieved = json.loads(provider.handle_tool_call("memory_retrieve", {"query": "Where did Dana move?"}))
    assert any(p["id"] == page_id for p in retrieved["passages"])

    relinked = json.loads(provider.handle_tool_call("memory_relink", {}))
    assert "pages" in relinked

    updated = json.loads(provider.handle_tool_call(
        "memory_update", {"page_id": page_id, "text": "Dana moved to Porto in March."}
    ))
    assert updated == {"ok": True}

    deleted = json.loads(provider.handle_tool_call("memory_delete", {"page_id": page_id}))
    assert deleted == {"ok": True}

    missing = json.loads(provider.handle_tool_call("memory_delete", {"page_id": page_id}))
    assert "error" in missing

    provider.shutdown()


def test_handle_tool_call_returns_a_json_string_not_a_dict(tmp_path):
    """The real MemoryManager.handle_tool_call forwards this value straight
    through with no conversion of its own — a dict would break the turn."""
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize("session-1", hermes_home=str(tmp_path))
    result = provider.handle_tool_call("memory_add", {"text": "anything"})
    assert isinstance(result, str)
    json.loads(result)  # must parse
    provider.shutdown()


def test_handle_tool_call_unknown_tool_raises(tmp_path):
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize("session-1", hermes_home=str(tmp_path))
    with pytest.raises(plugin.CortexProviderError):
        provider.handle_tool_call("not_a_real_tool", {})
    provider.shutdown()


def test_handle_tool_call_returns_error_json_on_server_failure(tmp_path):
    """A CortexError (rate limit, server error, etc.) surfaces as a tool
    result, not an unhandled exception that would crash the turn."""
    server = _FakeCortexServer()
    server.fail_with = httpx.Response(500, json={"error": "internal error"})
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))

    result = json.loads(provider.handle_tool_call("memory_add", {"text": "anything"}))
    assert "error" in result
    provider.shutdown()


# --- automatic per-turn hooks ------------------------------------------


def test_sync_turn_persists(tmp_path):
    server = _FakeCortexServer()
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))

    assert len(server.pages) == 0
    provider.sync_turn("What's the capital of France?", "Paris.")

    assert len(server.pages) == 1
    provider.shutdown()


def test_prefetch_returns_empty_string_on_empty_store(tmp_path):
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize("session-1", hermes_home=str(tmp_path))
    assert provider.prefetch("anything") == ""
    assert provider.recall_status() is None
    provider.shutdown()


def test_prefetch_returns_empty_string_on_server_error(tmp_path):
    server = _FakeCortexServer()
    server.fail_with = httpx.Response(503, json={"error": "unavailable"})
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))
    assert provider.prefetch("anything") == ""  # best-effort, doesn't raise
    provider.shutdown()


def test_recall_status_reflects_last_prefetch(tmp_path):
    server = _FakeCortexServer()
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))
    provider.handle_tool_call("memory_add", {"text": "Dana moved to Lisbon."})

    context = provider.prefetch("Where did Dana move?")
    assert context  # non-empty
    status = provider.recall_status()
    assert status is not None
    assert status.count == 1
    assert status.provider_label == "CortexLayer"
    provider.shutdown()


def test_on_session_end_relinks_without_crashing(tmp_path):
    server = _FakeCortexServer()
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))
    provider.handle_tool_call("memory_add", {"text": "Gina adopted a cat named Mochi."})
    provider.on_session_end([])  # must not raise
    provider.shutdown()


def test_on_pre_compress_is_idempotent(tmp_path):
    server = _FakeCortexServer()
    provider = _MockCortexMemoryProvider(server)
    provider.initialize("session-1", hermes_home=str(tmp_path))
    messages = [
        {"role": "user", "content": "I just got a new job at Acme Corp."},
        {"role": "assistant", "content": "Congratulations on the new job!"},
    ]

    first = provider.on_pre_compress(messages)
    before = len(server.pages)
    second = provider.on_pre_compress(messages)  # same content -> nothing new to contribute
    after = len(server.pages)

    assert first.startswith("checkpoint:")
    assert second == ""
    assert after == before
    provider.shutdown()


def test_on_pre_compress_empty_messages_contributes_nothing(tmp_path):
    provider = _MockCortexMemoryProvider(_FakeCortexServer())
    provider.initialize("session-1", hermes_home=str(tmp_path))
    assert provider.on_pre_compress([]) == ""
    provider.shutdown()


def test_system_prompt_block_deferred_not_silent():
    """Task 0088: explicit decision to defer, not an accidental no-op. Must
    return "" (not None) per the real base class default/contract."""
    assert plugin.CortexMemoryProvider().system_prompt_block() == ""


# --- MemoryManager integration (dev guide's documented test pattern) -----


def test_memory_manager_flow(tmp_path, monkeypatch):
    # MemoryManager.initialize_all only calls initialize() when is_available()
    # is true (the real Hermes contract — mirrors the dev guide's own
    # os.environ-based example) — set the env var the guide's is_available()
    # example checks, even though _build_client injects its own test key.
    monkeypatch.setenv("CORTEX_API_KEY", "ctx_test")
    server = _FakeCortexServer()
    mgr = MemoryManager()
    mgr.add_provider(_MockCortexMemoryProvider(server))
    mgr.initialize_all(session_id="test-1", platform="cli", hermes_home=str(tmp_path), user_id="ivy")

    result = json.loads(mgr.handle_tool_call("memory_add", {"text": "Ivy's favorite color is teal."}))
    assert result["page_ids"]

    mgr.sync_all("Tell me something else about Ivy.", "Noted.")
    mgr.on_session_end([])
    mgr.shutdown_all()
