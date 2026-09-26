import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The stub `agent` package (test double for Hermes' own `agent.memory_provider`
# / `agent.memory_manager` — see tests/stub_hermes/agent/memory_provider.py for
# why this exists instead of a real dependency) must resolve before the
# plugin package does its `from agent.memory_provider import ...` at import
# time.
sys.path.insert(0, str(ROOT / "tests" / "stub_hermes"))
sys.path.insert(0, str(ROOT / "plugins" / "memory"))
