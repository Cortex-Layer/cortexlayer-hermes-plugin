import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The stub `agent` package (test double for Hermes' own `agent.memory_provider`
# / `agent.memory_manager` — see tests/stub_hermes/agent/memory_provider.py for
# why this exists instead of a real dependency) must resolve before the
# plugin package does its `from agent.memory_provider import ...` at import
# time.
sys.path.insert(0, str(ROOT / "tests" / "stub_hermes"))

# Load the plugin module WITHOUT putting plugins/memory/ on sys.path. Its
# directory is literally named `cortexlayer` (task 0088's later fix — the
# on-disk directory name is Hermes' actual activation key, confirmed live),
# same as the real `cortexlayer` pip package this plugin itself imports
# (`from cortexlayer import CortexClient`) — a bare `sys.path.insert` here
# would shadow that package with the plugin's own directory and break the
# plugin's own internal import. The real Hermes loader avoids exactly this
# by dynamically constructing a namespaced module
# (`_hermes_user_memory.<name>` via importlib.util.module_from_spec) instead
# of touching sys.path; mirror that here.
_plugin_path = ROOT / "plugins" / "memory" / "cortexlayer" / "__init__.py"
_spec = importlib.util.spec_from_file_location("cortexlayer_hermes_plugin", _plugin_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["cortexlayer_hermes_plugin"] = _module
_spec.loader.exec_module(_module)
