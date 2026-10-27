"""CPU contract tests: no model weights, processor, CUDA, or network calls."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import nvfp4_cosmos as n


def panel_fixture():
    items = []
    for task in ("T1", "T2", "T3"):
        for cell in range(3):
            for side in ("LEFT", "RIGHT"):
                for i in range(24):
                    sid = f"{task}-c{cell}-{side}-{i:02}"
                    params = {"sf_cycles_per_aperture": 1, "modulation_depth": .1 + cell / 10} if task == "T1" else {
                        "onset_frame": i, "offset_frame": i + 30, "delta_Y": .1 + cell / 10,
                        "background_Y": .1, "angle_deg": cell + 1, "tilt_sign": 1}
                    row = {"sample_id": sid, "task": task, "prompt": "LEFT or RIGHT", "truth": side, "source_parameters": params}
                    ready = {"sample_id": sid, "video_relative": sid + ".mkv", "video_sha256": "a" * 64, "stream_sha256": "b" * 64}
                    items.append({"sample_id": sid, "task": task, "source_family": task.lower(), "row": row, "ready": ready,
                        "row_sha256": n.digest(row), "ready_sha256": n.digest(ready), "source_stream_sha256": "b" * 64})
    return {"items": items, "catalog_file_sha256": n.CATALOG_SHA}, {"items": {x["sample_id"]: x for x in items}}


def test_complete_cells_and_balanced_independent_workers():
    panel, catalog = panel_fixture()
    n.validate_panel(panel, catalog)
    workers = n.assignments(panel["items"])
    assert [len(w) for w in workers] == [240, 192]
    assert set(workers[0]).isdisjoint(workers[1])
    assert set(workers[0] + workers[1]) == set(catalog["items"])
    for worker in workers:
        assert [catalog["items"][i]["task"] for i in worker[:2]] == ["T2", "T3"]
        for task in ("T1", "T2", "T3"):
            counts = [sum(catalog["items"][i]["task"] == task and n.truth(catalog["items"][i]) == side for i in worker) for side in ("LEFT", "RIGHT")]
            assert counts[0] == counts[1] and counts[0] in (24, 48)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "row", "catalog", "incomplete_cell"])
def test_rejects_unbound_or_incomplete_scientific_panel(problem):
    panel, catalog = panel_fixture()
    if problem == "duplicate": panel["items"][-1] = deepcopy(panel["items"][0])
    elif problem == "missing": panel["items"].pop()
    elif problem == "row": panel["items"][0]["row_sha256"] = "changed"
    elif problem == "catalog": panel["catalog_file_sha256"] = "old"
    else:
        extra = deepcopy(panel["items"][0]); extra["sample_id"] = "extra"
        catalog["items"]["extra"] = extra
    with pytest.raises(ValueError): n.validate_panel(panel, catalog)


def test_t1_normalization_preserves_original_fields_and_values():
    item = {"row": {"sample_id": "x", "correct_answer": "LEFT", "params": {"m": .1}},
            "ready": {"sample_id": "x", "video_path": "media/x.mkv"}, "source_stream_sha256": "a" * 64}
    before = deepcopy(item)
    row, ready = n.normalized_input(item)
    assert item == before and row["truth"] == "LEFT" and row["source_parameters"] == item["row"]["params"]
    assert ready["video_relative"] == "media/x.mkv" and ready["stream_sha256"] == "a" * 64


def test_exclusive_writes_never_replace_existing_evidence(tmp_path):
    target = tmp_path / "sealed.json"
    n.write_new(target, {"original": True})
    with pytest.raises(FileExistsError): n.write_new(target, {"original": False})
    assert n.read(target) == {"original": True}


def records():
    def row(i):
        return {"sample_id": str(i), "task": "T1", "truth": "LEFT", "catalog_row_sha256": "r", "catalog_ready_sha256": "q",
                "source_family": "t1", "worker": 0, "effective_generation": {"temperature": 0},
                "input_receipt": {k: "same" for k in ("source_video_sha256", "source_stream_sha256", "chat_template_sha256",
                    "input_token_ids_sha256", "actual_processor_tensor_shape", "actual_processor_dtype")},
                "output_token_ids": [3, 4], "raw_answer": "LEFT", "finish_reason": "stop", "seconds": 10.,
                "full_request_seconds": 11., "deep": i == 0}
    return [row(i) for i in range(3)]


def test_output_differences_are_measured_not_rejected():
    a = records(); b = deepcopy(a)
    b[1].update(output_token_ids=[7, 4], raw_answer="RIGHT", seconds=5.)
    result = n.compare_records(a, b)
    assert result["all"]["N"] == 3 and result["all"]["tokens_equal"] == 2
    assert result["all"]["answers_equal"] == 2 and result["all"]["nvfp4_correct"] == 2
    assert result["steady"]["N"] == 2 and result["steady"]["paired_latency_speedup"] == 20 / 15
    assert result["rows"][1]["bf16_tokens"] == [3, 4] and result["rows"][1]["nvfp4_tokens"] == [7, 4]


@pytest.mark.parametrize("field", ["worker", "effective_generation", "source_video_sha256", "input_token_ids_sha256"])
def test_rejects_nonprecision_comparison_delta(field):
    a = records(); b = deepcopy(a)
    (b[1] if field in ("worker", "effective_generation") else b[1]["input_receipt"])[field] = "changed"
    with pytest.raises(ValueError): n.compare_records(a, b)


def test_rejects_missing_or_duplicate_paired_outputs():
    a = records()
    with pytest.raises(ValueError): n.compare_records(a, a[:-1])
    with pytest.raises(ValueError): n.compare_records(a + [a[0]], a)


def test_cpu_import_and_help_never_import_heavy_frameworks():
    script = "from classic_precision import nvfp4_cosmos; import sys; assert not {'torch','transformers','kernels','av'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", script], check=True)
    p = subprocess.run([sys.executable, "-m", "classic_precision.nvfp4_cosmos", "--help"], capture_output=True, text=True, check=True)
    assert "seal" in p.stdout and "compare" in p.stdout


def test_native_loader_preserves_exact_mapping_and_sdpa_source_contract():
    import ast
    tree = ast.parse(Path(n.__file__).read_text())
    calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "__func__"]
    assert len(calls) == 1
    args = {x.arg: ast.unparse(x.value) for x in calls[0].keywords if x.arg}
    assert args["key_mapping"] == "exact_mapping"
    assert args["device_map"] == "{'': 'cuda:0'}" and args["attn_implementation"] == "'sdpa'"
    assert args["local_files_only"] == "True" and args["dtype"] == "torch.bfloat16"


def test_failure_after_durable_answer_never_retries_and_closes_before_lease(monkeypatch, tmp_path):
    panel, _ = panel_fixture(); ids = n.assignments(panel["items"])[0]
    runtime = {"asset_roots": {"t2": str(tmp_path)}}
    plan = {"schema": n.SCHEMA, "max_scientific_calls": 864, "retries": 0, "runtime": runtime,
            "runtime_identity": {}, "modes": ["bf16", "nvfp4"], "assignments": [ids, []], "panel": panel}
    plan_file = tmp_path / "plan.json"; n.write_new(plan_file, plan)
    admission = tmp_path / "admission.json"
    n.write_new(admission, {"plan_sha256": n.file_sha256(plan_file), "production_held": True,
        "owned_engines_exited": True, "physical_locks_released": True, "workers": {"0": {"cuda_visible_devices": "0"}}})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(n, "runtime_identity", lambda runtime: {})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    life = []
    @contextmanager
    def lease(runtime, torch):
        life.append("acquire")
        try: yield {}
        finally: life.append("release")
    monkeypatch.setattr(n, "physical_lease", lease)
    class Fake:
        def __init__(self, *a): self.context = {}; life.append("load")
        def generate(self, row, ready, root, commit):
            life.append("generate"); commit({"raw_answer": "LEFT", "output_token_ids": [1, 2]})
            raise ValueError("post-output receipt failed")
        def close(self): life.append("close")
    monkeypatch.setattr(n, "CosmosSession", Fake)
    output = tmp_path / "output"
    args = SimpleNamespace(plan=plan_file, admission=admission, mode="nvfp4", worker=0, output=output)
    with pytest.raises(ValueError, match="post-output"): n.run(args)
    assert life == ["acquire", "load", "generate", "close", "release"]
    assert len((output / "raw.jsonl").read_text().splitlines()) == 1
    assert not (output / "complete.json").exists()
    with pytest.raises(FileExistsError): n.run(args)
    assert life.count("generate") == 1


def test_length_answer_is_retained_but_not_marked_correct():
    a = records(); b = deepcopy(a)
    b[1]["finish_reason"] = "length"
    result = n.compare_records(a, b)
    assert result["all"]["nvfp4_correct"] == 2
    assert result["all"]["nvfp4_invalid_or_length"] == 1
    assert result["rows"][1]["nvfp4_answer"] == "LEFT"


def test_native_policy_is_explicit_and_not_deterministic_override():
    assert n.NATIVE_TORCH_POLICY == {"float32_matmul_precision": "highest", "matmul_allow_tf32": False,
        "cudnn_allow_tf32": True, "cudnn_benchmark": False, "cudnn_deterministic": False,
        "deterministic_algorithms": False, "deterministic_warn_only": False}
