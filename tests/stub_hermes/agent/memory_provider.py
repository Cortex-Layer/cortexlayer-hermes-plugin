"""Test double for Hermes' ``agent.memory_provider`` module.

NOT the real Hermes package — Hermes isn't pip-installable (it ships via a
shell installer, not PyPI; verified during task 0088's design pass), so
there's nothing to depend on for CI. This reproduces just enough of the
documented contract (dev guide: base class methods, ``spawn_context_thread``'s
contextvars-preserving behavior) to exercise the plugin's own logic.

Real verification against Hermes' actual base class happens in the separate
manual end-to-end pass (task 0088's own acceptance criterion) — nothing here
substitutes for that.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any, Dict, List, Optional


class MemoryProvider:
    name: str = ""

    def is_available(self) -> bool:
        raise NotImplementedError

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass

    def system_prompt_block(self) -> Optional[str]:
        return None

    def prefetch(self, query: str, *, session_id: str = "") -> Optional[str]:
        return None

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(
        self, user: str, assistant: str, *, session_id: str = "", messages: Optional[list] = None
    ) -> None:
        pass

    def on_session_end(self, messages: list) -> None:
        pass

    def on_pre_compress(self, messages: list, *, require_checkpoint: bool = False):
        return None

    def on_memory_write(self, action, target, content, metadata=None) -> None:
        pass

    def shutdown(self) -> None:
        pass


def spawn_context_thread(target, *, name: Optional[str] = None) -> threading.Thread:
    """Mirrors the guide's contract: preserves contextvars (profile
    isolation) instead of a bare ``threading.Thread``."""
    ctx = contextvars.copy_context()
    return threading.Thread(target=lambda: ctx.run(target), name=name)
