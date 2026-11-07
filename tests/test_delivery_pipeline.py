"""CPU regressions for exact inputs, durable execution, and archive rescoring."""

from __future__ import annotations

from contextlib import closing
import json

import pytest

from vlm_psychophysics import data, results, runner


def trial(sample_id, *, truth="LEFT", index=0):
    return {
        "sample_id": sample_id,
        "task": "T2",
        "domain": "classic",
        "path": f"classic/t2/videos/{sample_id}.mkv",
        "video_sha256": data.digest(sample_id),
        "fps": 60,
        "frames": 180,
        "width": 768,
        "height": 512,
        "truth": truth,
        "prompt": "Original prompt.\n",
        "parameters": {
            "delta_Y": 0.1,
            "background_Y": 0.2,
            "onset_frame": 18,
            "offset_frame": 18 + index + 1,
        },
    }


def still(parent):
    sample_id = parent["sample_id"]
    pixel_sha = data.digest("shared pixels")
    prompt = "Original still prompt.\n"
    return {
        "sample_id": sample_id,
        "original_video_ID": sample_id,
        "task": parent["task"],
        "domain": parent["domain"],
        "truth": parent["truth"],
        "video_path": parent["path"],
        "video_sha256": parent["video_sha256"],
        "prompt": prompt,
        "choices": data.CHOICES,
        "static_case_id": "one-static-case",
        "model_input_sha256": data.digest([[pixel_sha], prompt, data.CHOICES]),
        "images": [
            {
                "path": f"classic/t2/static/images/{sample_id}.png",
                "pixel_sha256": pixel_sha,
                "sha256": data.digest(f"PNG {sample_id}"),
                "frame_index": 33,
            }
        ],
        "frame_indices": [33],
    }


def test_static_deduplicates_exact_images_and_preserves_aliases(monkeypatch):
    monkeypatch.setitem(data.jinwei_release.SCOPES, "classic/t2", (2, 1, "", ""))
    parents = {"a": trial("a"), "b": trial("b", index=2)}
    stills = {key: still(row) for key, row in parents.items()}
    cases = data.static_cases("classic/t2", parents, stills)
    assert len(cases) == 1
    case = cases[0]
    assert case["prompt"] == "Original still prompt.\n"
    assert [alias["sample_id"] for alias in case["aliases"]] == ["a", "b"]
    assert "duration_frames" not in results.case_condition(case)
    selected = list(data.selected_static_cases("classic/t2", parents, stills, {"b"}))
    assert selected[0]["case_id"] == case["case_id"]
    assert [alias["sample_id"] for alias in selected[0]["aliases"]] == ["b"]

    parents["b"]["truth"] = stills["b"]["truth"] = "RIGHT"
    with pytest.raises(ValueError, match="Conflicting truth"):
        data.static_cases("classic/t2", parents, stills)


def test_resume_does_not_repeat_an_intent_and_archive_rescores(tmp_path):
    config_path = runner.PROJECT / "configs/current-x.json"
    config = json.loads(config_path.read_bytes())
    cases = [
        data.video_case("classic/t2", trial(f"case-{index}")) for index in range(5)
    ]
    count, _, sha, _ = data.jinwei_release.SCOPES["classic/t2"]
    manifest = {
        "repo_id": data.REPO_ID,
        "revision": data.REVISION,
        "index_sha256": data.jinwei_release.INDEX_SHA,
        "manifests": [
            {"path": "classic/t2/metadata.jsonl", "sha256": sha, "rows": count}
        ],
        "cases": cases,
    }
    runner.save_plan(tmp_path, config_path, config, ("qwen",), manifest, {})
    model = "qwen"
    uuid = "GPU-11111111-1111-1111-1111-111111111111"
    jobs = [results.job_id(case, model) for case in cases]
    recipe = results.recipe_for(config, cases[0], model)
    results.write_json(
        tmp_path / "evidence/schedule.json", {"model_gpus": {model: [uuid]}}
    )
    session = tmp_path / "evidence/sessions" / uuid / recipe / "123-456"
    tokens = [3, 6, 1, 20]
    reference_key = data.digest("reference")
    results.write_bytes(
        session / "input-references.jsonl",
        results.jsonl_bytes(
            [
                {
                    "reference_key": reference_key,
                    "reference": {"prompt_token_ids": tokens},
                }
            ]
        ),
    )
    profile = {"recipe": recipe}
    profile_sha = data.digest(profile)
    results.write_json(
        session / "resolved-model.json",
        {
            "gpu_uuid": uuid,
            "profile": profile,
            "profile_sha256": profile_sha,
        },
    )
    results.write_json(session / "runtime.json", [{"actual_kv_capacity": 10000}])
    results.write_bytes(
        session / "input-observations.jsonl",
        results.jsonl_bytes(
            [
                {
                    "sample_id": cases[0]["sample_id"],
                    "reference_key": reference_key,
                    "GPU": [{"verified": True}],
                    "application": [{"verified": True}],
                }
            ]
        ),
    )
    returned = [
        {
            "case_id": cases[index]["case_id"],
            "sample_id": cases[index]["sample_id"],
            "raw_answer": raw,
            "finish_reason": "stop",
            "profile_sha256": profile_sha,
            "prompt_token_count": len(tokens),
            "prompt_ids_sha256": data.digest(tokens),
            "reference_key": reference_key,
            "output_token_ids": [20],
        }
        for index, raw in ((0, " LEFT\n"), (1, "LEFT."), (3, "RIGHT"))
    ]
    results.write_bytes(session / "returned.jsonl", results.jsonl_bytes(returned))
    with closing(runner.init_ledger(tmp_path)) as ledger:
        for job in jobs[:4]:
            assert runner.record_intent(ledger, job, uuid)
            assert not runner.record_intent(ledger, job, uuid)
        for index, raw in enumerate((" LEFT\n", "LEFT.")):
            answer = runner.answer_receipt(
                {
                    "raw_answer": raw,
                    "finish_reason": "stop",
                    "profile_sha256": profile_sha,
                    "technical_pass": True,
                    "prompt_token_ids": tokens,
                    "output_token_ids": [20],
                    "input_receipt": {
                        "prompt_sha256": cases[index]["prompt_sha256"],
                        "prompt_ids_sha256": data.digest(tokens),
                        "reference_key": reference_key,
                        "deep_observation": index == 0,
                        "admission": {"verified": True} if index == 0 else None,
                    },
                }
            )
            runner.record_answer(ledger, jobs[index], answer)
    summary = runner.report(tmp_path)
    assert summary["accuracy_denominator"] == 5
    assert summary["correct"] == summary["invalid_answers"] == summary["missing"] == 1
    assert summary["uncertain"] == summary["technical_failures"] == 1
    archive = tmp_path / "return.zip"
    assert results.verify_archive(archive)["accuracy"] == 0.2
    with (tmp_path / "results.jsonl").open() as stream:
        rows = data.read_jsonl(stream)
    assert rows[0]["raw_answer"] == " LEFT\n" and rows[0]["prediction"] == "LEFT"
    assert rows[1]["status"] == "INVALID_ANSWER"
    assert rows[3]["raw_answer"] == "RIGHT" and rows[3]["status"] == "TECHNICAL_FAILURE"
    assert data.runtime_case(cases[0], tmp_path)["prompt"] == "Original prompt.\n"
    assert "truth" not in data.runtime_case(cases[0], tmp_path)
    returned[0]["raw_answer"] = "RIGHT"
    results.write_bytes(session / "returned.jsonl", results.jsonl_bytes(returned))
    with pytest.raises(ValueError, match="Durable engine output differs"):
        results.verify_archive(results.make_archive(tmp_path))


def test_schedule_skips_occupied_and_incompatible_cards(monkeypatch):
    config = json.loads((runner.PROJECT / "configs/current-x.json").read_bytes())
    first, second, third = [
        f"GPU-{i:08d}-1111-1111-1111-111111111111" for i in range(3)
    ]
    devices = {
        first: {
            "index": 0,
            "free_mib": 91000,
            "used_mib": 800,
            "utilization": 1,
            "compute_pids": [99],
            "compute_capability": "12.0",
        },
        second: {
            "index": 1,
            "free_mib": 91000,
            "used_mib": 20,
            "utilization": 0,
            "compute_pids": [],
            "compute_capability": "12.0",
        },
        third: {
            "index": 2,
            "free_mib": 91000,
            "used_mib": 20,
            "utilization": 0,
            "compute_pids": [],
            "compute_capability": "8.0",
        },
    }
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runner, "available_ram", lambda: 160 * runner.GIB)
    monkeypatch.setattr(runner, "gpu_list", lambda: devices)
    schedule = runner.new_schedule(config, ("qwen",), "auto")
    assert schedule["model_gpus"] == {"qwen": [second]}
