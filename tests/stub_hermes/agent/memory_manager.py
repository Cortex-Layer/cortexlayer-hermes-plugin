"""Test double for Hermes' ``agent.memory_manager.MemoryManager`` — just
enough of the dev guide's documented test pattern (``add_provider``,
``initialize_all``, ``handle_tool_call``, ``sync_all``, ``shutdown_all``) to
exercise the plugin through it, the way the real Hermes runtime would. See
``memory_provider.py`` in this stub package for why this exists instead of a
real dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List


class MemoryManager:
    def __init__(self) -> None:
        self._providers: List[Any] = []

    def add_provider(self, provider: Any) -> None:
        self._providers.append(provider)

    def initialize_all(self, *, session_id: str = "", **kwargs: Any) -> None:
        for p in self._providers:
            if p.is_available():
                p.initialize(session_id, **kwargs)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> Any:
        for p in self._providers:
            if any(s["name"] == tool_name for s in p.get_tool_schemas()):
                return p.handle_tool_call(tool_name, args, **kwargs)
        raise KeyError(f"no provider exposes tool {tool_name!r}")

    def sync_all(self, user: str, assistant: str, **kwargs: Any) -> None:
        for p in self._providers:
            p.sync_turn(user, assistant, **kwargs)

    def on_session_end(self, messages: list) -> None:
        for p in self._providers:
            p.on_session_end(messages)

    def shutdown_all(self) -> None:
        for p in self._providers:
            p.shutdown()
