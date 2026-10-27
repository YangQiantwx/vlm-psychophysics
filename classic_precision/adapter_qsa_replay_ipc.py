"""Source-bound native adapter replay with by-value dynamic-module IPC only.

The reviewed entry and native scientific adapters stay byte-identical. Private
module instances add cloudpickle transport registration before engine creation;
native multiprocessing, generation, guards and numerical policies are unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from .storage import file_sha256

PARENT_ENTRY_SHA256 = "75cbfcc32a10cbf8ef94e2b3122486f8c6cbabd6d2828cf1bb89c6708b759b3c"
PARENT_NATIVE_SHA256 = "4719c8b3f18d985d755454d9c1ae405e8145665990a70653609dcac404cc14f6"
SCHEMA = "classic_precision_native_adapter_qsa_early_stop_ipc/v2"
_private_entry = None


def _require(condition, message):
    if not condition: raise RuntimeError(message)


def _private(name, path, expected):
    _require(file_sha256(path) == expected and name not in sys.modules, "Frozen private replay source/instance differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def register_native_module(module, expected_sha256):
    """Transmit original callbacks by value; never replace a native callable."""
    import cloudpickle
    _require(isinstance(module, ModuleType) and getattr(module, "__file__", None)
        and file_sha256(module.__file__) == expected_sha256, "Unqualified native IPC module source")
    present = sys.modules.get(module.__name__)
    _require(present is None or present is module, "Native IPC module name belongs to another instance")
    sys.modules[module.__name__] = module
    cloudpickle.register_pickle_by_value(module)
    result = [{"module": module.__name__, "source_sha256": expected_sha256}]
    # T1 multi imports its exact reviewed T4 helper without registering it in
    # sys.modules. Register that source-qualified nested module as well.
    nested = getattr(module, "t4", None)
    if isinstance(nested, ModuleType):
        _require(Path(nested.__file__).resolve() == Path(module.T4_PATH).resolve()
            and file_sha256(nested.__file__) == module.T4_SHA, "Nested native IPC helper source differs")
        result.extend(register_native_module(nested, module.T4_SHA))
    return result


def _entry():
    global _private_entry
    if _private_entry is None:
        directory = Path(__file__).resolve().parent
        entry = _private("classic_precision._adapter_qsa_ipc_private_entry", directory / "adapter_qsa_replay.py", PARENT_ENTRY_SHA256)
        native = _private("classic_precision._adapter_qsa_ipc_private_native", directory / "adapter_replay.py", PARENT_NATIVE_SHA256)
        original_load = native._load
        def load(name, path):
            module = original_load(name, path)  # original plan/source guards run before this loader
            register_native_module(module, file_sha256(path))
            return module
        native._load = load
        entry.native = native
        entry.SCHEMA = SCHEMA
        original_implementation = entry.implementation
        def implementation():
            return {**original_implementation(), "entry_sha256": file_sha256(__file__),
                "parent_entry_sha256": PARENT_ENTRY_SHA256,
                "native_ipc_transport": "cloudpickle.register_pickle_by_value/source-qualified-native-modules"}
        entry.implementation = implementation
        _private_entry = entry
    return _private_entry


def __getattr__(name):
    return getattr(_entry(), name)


def main(argv=None):
    return _entry().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
