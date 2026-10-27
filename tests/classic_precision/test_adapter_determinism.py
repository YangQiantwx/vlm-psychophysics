"""CPU-only source/guard/hash tests; no checkpoint or CUDA execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from runtime_compat import classic_precision_adapter_determinism as adapter
from runtime_compat import classic_precision_determinism as real_core


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(tmp_path, monkeypatch):
    base_path = tmp_path / "base.py"
    base_path.write_text('''class ModelSession:
    def _initialize_processor(self):
        return "native processor guard"
    def prepare(self, row):
        return "native video prepare"
    def generate(self, row):
        return "native video generate"
''')
    core_path = tmp_path / "core.py"
    qsa_path = tmp_path / "classic_precision_qsa_determinism.py"
    qsa_path.write_text('# qualified CPU fixture only\n')
    core_path.write_text('''from functools import wraps
import hashlib
from pathlib import Path
FAMILY = None
QSA_STATE = "applied"
FILL_UNINITIALIZED_MEMORY = {"nemo": True, "qwen": False, "cosmos": True, "gemma": True}
def launch_environment():
    return {"ADAPTER_TEST_ENV": "1"}
def status():
    return {"family": FAMILY, "qsa_installation": {"state": QSA_STATE, "helper_source_sha256": hashlib.sha256(Path(__file__).with_name("classic_precision_qsa_determinism.py").read_bytes()).hexdigest()}}
def tensor_fingerprint(tensor, name="pixels"):
    from runtime_compat.classic_precision_determinism import tensor_fingerprint
    return tensor_fingerprint(tensor, name)
def activate(family):
    global FAMILY
    FAMILY = family
    original_prepare = BASE.ModelSession.prepare
    original_generate = BASE.ModelSession.generate
    original_processor = BASE.ModelSession._initialize_processor
    @wraps(original_prepare)
    def prepare(*args, **kwargs):
        if original_prepare is None:
            return None
        raise AssertionError("incompatible video-only wrapper")
    @wraps(original_generate)
    def generate(*args, **kwargs):
        if original_generate is None:
            return None
        raise AssertionError("incompatible video-only wrapper")
    @wraps(original_processor)
    def processor(*args, **kwargs):
        return original_processor(*args, **kwargs)
    BASE.ModelSession.prepare = prepare
    BASE.ModelSession.generate = generate
    BASE.ModelSession._initialize_processor = processor
''')
    source = tmp_path / "native_adapter.py"
    source.write_text('''VALIDATED = []
def decode_t1(*args):
    return "native lossless source guard"
def _install_strong_probe(*args):
    return "native strong probe"
def _read_strong_probe(*args):
    return "native strong receipts"
def validate_post(post, family, base):
    if not post.get("guard_ok"):
        raise ValueError("native T1 guard")
class GuardStore:
    def __init__(self, raw, base):
        self.raw, self.base, self.session = raw, base, None
    def event(self, kind, **fields):
        if kind == "POST_OUTPUT_INPUT_RECEIPT":
            validate_post(fields, self.session.family, self.base)
        self.raw.event(kind, **fields)
def validate_gemma(raw, post):
    if not post.get("guard_ok"):
        raise ValueError("native Gemma guard")
    VALIDATED.append(raw)
class Session:
    def __init__(self, family, batch, store, model):
        self.family, self.batch, self.store, self.model = family, batch, store, model
        self.hash_deep, self.native_guard_count = True, 0
        self.fail_guard, self.skip_forward = False, False
    def prepare(self, row):
        return None, self.batch, "unchanged prompt", {"deep_check": True}
    def prepare_pair(self, row, deep=True):
        return None, self.batch, "unchanged prompt", {"deep_check": deep}, [1, 2]
    def prepare_images(self, row, deep=True):
        return None, self.batch, "unchanged prompt", {"deep_check": deep}, [1, 2]
    def prepare_image(self, row, deep=True):
        return None, self.batch, "unchanged prompt", {"deep_check": deep}
    def _finish(self, row, prepared, commit):
        self.native_guard_count += 1
        if self.fail_guard:
            raise ValueError("native scientific guard")
        if not self.skip_forward:
            self.vision(self.batch["pixel_values"].bfloat16())
        raw = {"sample_id": row["sample_id"], "raw_answer": "LEFT", "output_token_ids": [1, 2]}
        commit(raw)
        self.store.event("NATIVE_STRONG_GUARD_PASS", sample_id=row["sample_id"])
        return "native result"
    def generate(self, row, commit):
        return self._finish(row, self.prepare(row), commit)
    def generate_pair(self, row, commit, deep=True):
        return self._finish(row, self.prepare_pair(row, deep), commit)
    def generate_images(self, row, commit, deep=True):
        return self._finish(row, self.prepare_images(row, deep), commit)
    def generate_image(self, row, commit, deep=True):
        return self._finish(row, self.prepare_image(row, deep), commit)
class GemmaSession:
    def __init__(self, batch, store, model):
        self.family, self.batch, self.store, self.model = "gemma", batch, store, model
    def prepare_image(self, row, image, deep=True):
        from runtime_compat.classic_precision_determinism import tensor_fingerprint
        pixels = tensor_fingerprint(self.batch["pixel_values"].bfloat16(), "pixels")
        positions = tensor_fingerprint(self.batch["image_position_ids"], "pixel_position_ids")
        return self.batch, {"deep_check": deep, "actual_processor_tensor_shape": pixels["shape"],
                           "expected_BF16_GPU_sha256": pixels["sha256"], "image_position_ids_sha256": positions["sha256"]}
''')
    base = _load("adapter_fixture_base", base_path, monkeypatch)
    core = _load("adapter_fixture_core", core_path, monkeypatch)
    module = _load("adapter_fixture_native", source, monkeypatch)
    core.BASE = base
    coverage = {"schema": "classic_precision_input_path_coverage/v1",
        "frozen_strict_profile": {"implementation": {"determinism_helper_sha256": adapter._sha(core_path),
            "base_runner_sha256": adapter._sha(base_path), "qsa_helper_sha256": adapter._sha(qsa_path)}},
        "routes": [{"route_id": route, "implementation_files": [{"path": str(source), "sha256": adapter._sha(source)}]}
                   for route in ("video", "single", "multi", "pair")]}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))
    monkeypatch.setattr(adapter, "_state", {})
    monkeypatch.setattr(adapter, "_core", lambda: core)
    monkeypatch.setenv("ADAPTER_TEST_ENV", "1")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
        monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")
        monkeypatch.setattr(torch.backends.cudnn.rnn, "fp32_precision", "ieee")
    else:
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    def activate(family="cosmos"):
        # Fake core performs no import while the fresh-process precondition is
        # exercised. The real CPU Torch module is restored immediately.
        monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", core.FILL_UNINITIALIZED_MEMORY[family])
        with monkeypatch.context() as fresh:
            fresh.delitem(sys.modules, "torch")
            return adapter.activate(family, path, base)
    return SimpleNamespace(base=base, core=core, module=module, path=path, source=source,
                           activate=activate, monkeypatch=monkeypatch)


class Store:
    def __init__(self):
        self.events = []
    def event(self, kind, **fields):
        self.events.append({"event": kind, **fields})


def _session(native, family="cosmos", gemma_explicit=False):
    store, vision = Store(), torch.nn.Identity()
    if family == "gemma":
        model = SimpleNamespace(model=SimpleNamespace(vision_tower=vision))
    elif family == "qwen":
        model = SimpleNamespace(visual=vision)
    else:
        model = SimpleNamespace(model=SimpleNamespace(visual=vision))
    batch = {"input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
             "pixel_values": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)}
    if gemma_explicit:
        batch["image_position_ids"] = torch.tensor([[0, 1]], dtype=torch.int64)
    cls = native.module.GemmaSession if gemma_explicit else native.module.Session
    session = cls(batch, store, model) if gemma_explicit else cls(family, batch, store, model)
    session.vision = vision
    return session, store


def _guard_video(native, session):
    guard = native.module.GuardStore(session.store, native.base)
    guard.session, session.store = session, guard
    return guard


def _mock_gpu_capture(monkeypatch):
    # Only hook orchestration is simulated. Separate tests prove that production
    # _gpu_tensors rejects CPU data and tensor_fingerprint hashes actual bytes.
    monkeypatch.setattr(adapter, "_gpu_tensors", lambda module, args, kwargs, family:
                        [real_core.tensor_fingerprint(args[0], "pixels")])


def test_activation_retains_core_processor_and_restores_only_video_wrappers(native):
    prepare, generate = native.base.ModelSession.prepare, native.base.ModelSession.generate
    processor = native.base.ModelSession._initialize_processor
    native.activate()
    assert native.base.ModelSession.prepare is prepare and native.base.ModelSession.generate is generate
    assert native.base.ModelSession._initialize_processor is not processor
    assert native.base.ModelSession._initialize_processor.__wrapped__ is processor
    assert adapter.status()["certification"] == "NOT_VALIDATED"


@pytest.mark.parametrize("route,method", [("video", "generate"), ("single", "generate_image"),
                                           ("multi", "generate_images"), ("pair", "generate_pair")])
def test_native_guards_callbacks_and_results_are_preserved(native, monkeypatch, route, method):
    native.activate()
    _mock_gpu_capture(monkeypatch)
    session, store = _session(native)
    original = getattr(session, method)
    if route == "video":
        _guard_video(native, session)
    handle = adapter.instrument_session(session, route, native.module, native.path)
    raw = []
    assert getattr(session, method)({"sample_id": "original"}, raw.append) == "native result"
    assert session.native_guard_count == 1 and len(raw) == 1
    assert [e["stage"] for e in store.events if e["event"] == "REPRO_INPUT_HASH"] == ["cpu_processor", "gpu_forward"]
    assert any(e["event"] == "NATIVE_STRONG_GUARD_PASS" for e in store.events)
    if route == "video":
        assert native.base.decode_rgb16 is native.module.decode_t1
        assert native.base._install_vision_probe is native.module._install_strong_probe
    handle.close()
    assert getattr(session, method) == original and not session.vision._forward_pre_hooks


def test_native_guard_exception_never_fabricates_raw_or_gpu_success(native, monkeypatch):
    native.activate()
    _mock_gpu_capture(monkeypatch)
    session, store = _session(native)
    session.fail_guard = True
    adapter.instrument_session(session, "pair", native.module, native.path)
    raw = []
    with pytest.raises(ValueError, match="native scientific guard"):
        session.generate_pair({"sample_id": "original"}, raw.append)
    assert raw == [] and not session.vision._forward_pre_hooks
    assert not any(e.get("stage") == "gpu_forward" for e in store.events)


def test_missing_actual_gpu_capture_fails_and_keeps_existing_native_raw(native):
    native.activate()
    session, store = _session(native)
    session.skip_forward = True
    adapter.instrument_session(session, "pair", native.module, native.path)
    raw = []
    with pytest.raises(RuntimeError, match="GPU vision tensors"):
        session.generate_pair({"sample_id": "original"}, raw.append)
    assert len(raw) == 1
    assert not any(e.get("stage") == "gpu_forward" for e in store.events)


def test_qsa_pending_rejects_success_without_inventing_another_answer(native, monkeypatch):
    native.activate("qwen")
    _mock_gpu_capture(monkeypatch)
    session, store = _session(native, "qwen")
    native.core.QSA_STATE = "pending_target_import"
    adapter.instrument_session(session, "single", native.module, native.path)
    raw = []
    with pytest.raises(RuntimeError, match="QSA hook is not applied"):
        session.generate_image({"sample_id": "original"}, raw.append)
    assert len(raw) == 1 and session.native_guard_count == 1
    assert not any(e.get("stage") == "gpu_forward" for e in store.events)


def test_gpu_entry_rejects_cpu_tensor_instead_of_relabeling_it():
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        adapter._gpu_tensors(torch.nn.Identity(), (torch.ones(2, dtype=torch.bfloat16),), {}, "cosmos")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64])
def test_processor_hash_preserves_actual_dtype_shape_and_bytes(monkeypatch, dtype):
    monkeypatch.setattr(adapter, "_core", lambda: real_core)
    tensor = torch.arange(30).reshape(5, 6).to(dtype)[:, ::2]
    receipt = adapter._cpu_tensors({"input_ids": torch.tensor([[1]], dtype=torch.int64), "pixel_values": tensor})[-1]
    expected = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    assert receipt == {"name": "pixel_values", "dtype": str(dtype), "shape": [5, 3],
                       "sha256": hashlib.sha256(expected).hexdigest(), "bytes": len(expected)}


def test_unqualified_source_live_code_and_changed_coverage_are_rejected(native):
    native.activate()
    session, _ = _session(native)
    session.prepare_pair = lambda *a, **k: None
    with pytest.raises(RuntimeError, match="unqualified"):
        adapter.instrument_session(session, "pair", native.module, native.path)
    session, _ = _session(native)
    native.path.write_text(native.path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="Coverage changed"):
        adapter.instrument_session(session, "pair", native.module, native.path)


def test_non_deep_call_and_relaxed_torch_policy_are_rejected(native, monkeypatch):
    native.activate()
    session, _ = _session(native)
    adapter.instrument_session(session, "pair", native.module, native.path)
    with pytest.raises(RuntimeError, match="deep=True"):
        session.generate_pair({"sample_id": "original"}, lambda raw: None, deep=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="Torch policy"):
        session.generate_pair({"sample_id": "original"}, lambda raw: None)


@pytest.mark.parametrize("guard_ok", [True, False])
def test_gemma_explicit_loop_requires_original_validator(native, monkeypatch, guard_ok):
    native.activate("gemma")
    _mock_gpu_capture(monkeypatch)
    session, store = _session(native, "gemma", gemma_explicit=True)
    handle = adapter.instrument_session(session, "single", native.module, native.path)
    row = {"sample_id": "original"}
    def operation():
        _, receipt = session.prepare_image(row, {"original_image": True}, deep=True)
        session.vision(session.batch["pixel_values"].bfloat16())
        return {"raw": {"input_receipt": receipt}, "post": {"guard_ok": guard_ok,
                "actual_GPU_visual_forward": [{"shape": receipt["actual_processor_tensor_shape"],
                    "dtype": "torch.bfloat16", "device": "cuda:0", "tensor_sha256": receipt["expected_BF16_GPU_sha256"],
                    "positions_sha256": receipt["image_position_ids_sha256"]}]}}
    monkeypatch.setattr(adapter, "_gpu_tensors", lambda *args: [
        real_core.tensor_fingerprint(session.batch["pixel_values"].bfloat16(), "pixels"),
        real_core.tensor_fingerprint(session.batch["image_position_ids"], "pixel_position_ids")])
    if guard_ok:
        result = handle.capture(row, operation)
        assert native.module.VALIDATED == [result["raw"]]
        assert store.events[-1]["stage"] == "gpu_forward"
    else:
        with pytest.raises(ValueError, match="native Gemma guard"):
            handle.capture(row, operation)
        assert not any(e.get("stage") == "gpu_forward" for e in store.events)


def test_worker_probe_coexists_with_native_strong_probe_and_is_removed(native, monkeypatch):
    native.activate()
    _mock_gpu_capture(monkeypatch)
    session, store = _session(native)
    sentinel = object()
    worker = SimpleNamespace(get_model=lambda: session.model, _classic_precision_receipts=sentinel)
    session.engine = SimpleNamespace(collective_rpc=lambda function, args=(): [function(worker, *args)])
    adapter.instrument_session(session, "pair", native.module, native.path)
    session.generate_pair({"sample_id": "original"}, lambda raw: None)
    assert worker._classic_precision_receipts is sentinel
    assert not hasattr(worker, adapter._PROBE_ATTRIBUTE) and not session.vision._forward_pre_hooks
    assert store.events[-1]["stage"] == "gpu_forward"


def test_live_callable_substitution_in_same_source_file_is_rejected(native, monkeypatch):
    native.activate()
    session, _ = _session(native)
    monkeypatch.setattr(native.module.Session.prepare_pair, "__code__", native.module.Session.prepare_image.__code__)
    with pytest.raises(RuntimeError, match="API identity"):
        adapter.instrument_session(session, "pair", native.module, native.path)


def test_explicit_operation_cannot_attribute_another_scientific_row(native):
    native.activate("gemma")
    session, store = _session(native, "gemma", gemma_explicit=True)
    handle = adapter.instrument_session(session, "single", native.module, native.path)
    with pytest.raises(RuntimeError, match="scientific row"):
        handle.capture({"sample_id": "original", "prompt": "original"},
                       lambda: session.prepare_image({"sample_id": "other", "prompt": "changed"}, {}, True))
    assert store.events == [] and not session.vision._forward_pre_hooks


def test_actual_frozen_core_can_be_separated_without_torch_or_model_import(tmp_path):
    root = Path(__file__).resolve().parents[2]
    frozen = [root / "runtime_compat/classic_precision_determinism.py",
              root / "runtime_compat/classic_precision_qsa_determinism.py",
              root / "classic_precision/reproducibility.py", root / "classic_precision/inference.py"]
    before = {str(p): adapter._sha(p) for p in frozen}
    coverage = {"schema": "classic_precision_input_path_coverage/v1", "routes": [],
        "frozen_strict_profile": {"implementation": {
            "determinism_helper_sha256": before[str(frozen[0])],
            "base_runner_sha256": before[str(frozen[3])]}}}
    path = tmp_path / "actual-coverage.json"
    path.write_text(json.dumps(coverage))
    source = '''import sys
from classic_precision_adapter_determinism import activate
activate("cosmos", sys.argv[1])
assert "torch" not in sys.modules and "vllm" not in sys.modules
from classic_precision import inference
assert not hasattr(inference.ModelSession.prepare,"__wrapped__")
assert not hasattr(inference.ModelSession.generate,"__wrapped__")
assert hasattr(inference.ModelSession._initialize_processor,"__wrapped__")
'''
    env = {**os.environ, **real_core.launch_environment(),
           "PYTHONPATH": str(root / "runtime_compat") + os.pathsep + str(root)}
    result = subprocess.run([sys.executable, "-B", "-c", source, str(path)], env=env,
                            cwd=root, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert before == {str(p): adapter._sha(p) for p in frozen}


def test_qsa_source_is_checked_before_any_core_activation(native):
    path = Path(native.core.__file__).with_name("classic_precision_qsa_determinism.py")
    path.write_text('# changed source must not execute\n')
    with pytest.raises(RuntimeError, match="QSA helper source"):
        native.activate("qwen")
    assert native.core.FAMILY is None


def test_instrumentation_cannot_activate_after_torch_import(native):
    with pytest.raises(RuntimeError, match="before importing Torch"):
        adapter.activate("cosmos", native.path, native.base)


@pytest.mark.parametrize("fault", ["plain", "session", "base", "changed_after_wrap"])
def test_video_requires_original_connected_guard_store(native, fault):
    native.activate()
    session, _ = _session(native)
    if fault != "plain":
        guard = _guard_video(native, session)
        if fault == "session":
            guard.session = object()
        if fault == "base":
            guard.base = object()
    if fault == "changed_after_wrap":
        adapter.instrument_session(session, "video", native.module, native.path)
        session.store = Store()
        with pytest.raises(RuntimeError, match="GuardStore"):
            session.generate({"sample_id": "x"}, lambda raw: None)
    else:
        with pytest.raises(RuntimeError, match="GuardStore"):
            adapter.instrument_session(session, "video", native.module, native.path)


@pytest.mark.parametrize("fault", ["deep", "receipt", "post"])
def test_explicit_gemma_cannot_substitute_receipt_or_gpu_post(native, monkeypatch, fault):
    native.activate("gemma")
    session, store = _session(native, "gemma", gemma_explicit=True)
    monkeypatch.setattr(adapter, "_gpu_tensors", lambda *args: [
        real_core.tensor_fingerprint(session.batch["pixel_values"].bfloat16(), "pixels"),
        real_core.tensor_fingerprint(session.batch["image_position_ids"], "pixel_position_ids")])
    handle = adapter.instrument_session(session, "single", native.module, native.path)
    row = {"sample_id": "original"}
    def operation():
        _, receipt = session.prepare_image(row, {}, deep=True)
        session.vision(session.batch["pixel_values"].bfloat16())
        post = {"guard_ok": True, "actual_GPU_visual_forward": [{"shape": receipt["actual_processor_tensor_shape"],
            "dtype": "torch.bfloat16", "device": "cuda:0", "tensor_sha256": receipt["expected_BF16_GPU_sha256"],
            "positions_sha256": receipt["image_position_ids_sha256"]}]}
        if fault == "deep":
            receipt["deep_check"] = False
        elif fault == "receipt":
            receipt = {**receipt, "substituted": True}
        else:
            post["actual_GPU_visual_forward"][0]["tensor_sha256"] = "0" * 64
        return {"raw": {"input_receipt": receipt}, "post": post}
    with pytest.raises(RuntimeError, match="receipt differs|independently captured"):
        handle.capture(row, operation)
    assert not any(e.get("stage") == "gpu_forward" for e in store.events)


@pytest.mark.parametrize("family", ["cosmos", "qwen"])
def test_family_fill_policy_cannot_be_relaxed_or_enabled_instead(native, monkeypatch, family):
    native.activate(family)
    adapter._assert_policy(family)
    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", not native.core.FILL_UNINITIALIZED_MEMORY[family])
    with pytest.raises(RuntimeError, match="fill_uninitialized_memory"):
        adapter._assert_policy(family)
