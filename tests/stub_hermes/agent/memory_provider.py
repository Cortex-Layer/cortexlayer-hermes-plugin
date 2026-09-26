"""Test double for Hermes' ``agent.memory_provider`` module.

NOT the real Hermes package — Hermes isn't pip-installable (it ships via a
shell installer, not PyPI), so there's nothing to depend on for CI. This
reproduces the documented/observed contract of the real
``nousresearch/hermes-agent`` image's ``agent/memory_provider.py`` (read
directly from a live container on 2026-09-26 to verify this plugin, not
just the dev guide's summary — see the plugin module's own docstring for
what that verification changed) closely enough to exercise this plugin's
own logic. It is a from-scratch reimplementation of the interface for
testing, not a copy of Hermes' source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RecallStatus:
    provider_label: str
    count: int
    glyph: str = "\U0001f9e0"


class MemoryProvider:
    name: str = ""

    def is_available(self) -> bool:
        raise NotImplementedError

    def unavailable_reason(self) -> str:
        return ""

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def recall_status(self) -> Optional[RecallStatus]:
        return None

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages: Optional[list] = None,
    ) -> None:
        pass

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        pass

    def on_session_end(self, messages: list) -> None:
        pass

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                           reset: bool = False, rewound: bool = False, **kwargs: Any) -> None:
        pass

    def on_pre_compress(self, messages: list) -> str:
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> None:
        pass

    def on_memory_write(self, action: str, target: str, content: str,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    def backup_paths(self) -> List[str]:
        return []

    def shutdown(self) -> None:
        pass
