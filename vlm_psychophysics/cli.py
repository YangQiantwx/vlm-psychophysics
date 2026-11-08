"""Command line entry point for the pinned collaborator evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from . import data, results, runner


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Run or verify the pinned VLM psychophysics evaluation.",
        epilog="Example: python run.py --task t1-classic --model qwen --input video --sample-id SAMPLE_ID",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "plan", "doctor", "download", "report", "verify"),
    )
    parser.add_argument(
        "archive", nargs="?", type=Path, help="Return ZIP path for verify."
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=("all", *data.TASK_OPTIONS),
        help="Repeat to select scopes. Default: classic and natural T1–T4.",
    )
    parser.add_argument("--model", choices=("all", *data.MODELS), default="all")
    parser.add_argument(
        "--input",
        choices=("video", "static", "both"),
        help="Default: both, or video for the original paired T5 and --video.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--video", help="Published video name or a local video path.")
    selector.add_argument("--sample-id", help="Published video or static case ID.")
    parser.add_argument(
        "--config", type=Path, default=runner.PROJECT / "configs/current-x.json"
    )
    parser.add_argument(
        "--data-root", type=Path, help="SSD-backed dataset cache directory."
    )
    parser.add_argument(
        "--model-root", type=Path, help="SSD-backed model cache directory."
    )
    parser.add_argument("--out", type=Path, help="SSD-backed output directory.")
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Auto-select idle GPUs, or give a comma-separated UUID list.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally available media and weights.",
    )
    parser.add_argument(
        "--media-only", action="store_true", help="For download, fetch only media."
    )
    parser.add_argument(
        "--paired-results", type=Path, help="Rescore a historical original-reference T5 answer bundle."
    )
    args = parser.parse_args(argv)
    if args.archive is not None and args.command != "verify":
        parser.error("An archive path is only valid with verify.")
    if args.media_only and args.command != "download":
        parser.error("--media-only is only valid with download.")
    if args.paired_results and args.command != "report":
        parser.error("--paired-results is only valid with report.")
    return args


def selection_for(args):
    scopes = data.DEFAULT_SCOPES
    if args.task:
        requested = {data.TASK_OPTIONS[value] for value in args.task if value != "all"}
        if "all" in args.task:
            requested.update(data.DEFAULT_SCOPES)
        scopes = tuple(scope for scope in data.SCOPES if scope in requested)
    mode = args.input or ("video" if args.video else "both")
    modalities = data.MODALITIES if mode == "both" else (mode,)
    models = data.MODELS if args.model == "all" else (args.model,)
    return {
        "scopes": scopes,
        "modalities": modalities,
        "models": models,
        "sample_id": args.sample_id,
        "video_name": args.video,
    }


def output_directory(args, selection):
    if args.out:
        return args.out.expanduser().resolve()
    label = (
        selection["scopes"][0].replace("/", "-")
        if len(selection["scopes"]) == 1
        else "benchmark"
    )
    return (Path.cwd() / "outputs" / f"{label}-{data.digest(selection)[:12]}").resolve()


def load_config(path, models, modalities, scopes):
    config = json.loads(path.read_bytes())
    if (
        config["schema"] != "vlm-psychophysics-run-config-v1"
        or config["dataset"]["repo_id"] != data.REPO_ID
        or config["dataset"]["revision"] != data.REVISION
    ):
        raise ValueError(
            "Config must use the pinned collaborator runtime and dataset revision."
        )
    for model in models:
        entry = config["models"][model]
        if min(entry["minimum_free_vram_mib"], entry["minimum_host_ram_gib"]) <= 0:
            raise ValueError(f"Missing a positive resource threshold for {model}")
        for modality in modalities:
            for scope in scopes:
                profile = entry["profiles"][data.profile_key(modality, scope)]
                if profile not in config["recipes"]:
                    raise ValueError(f"Missing recipe {profile} for {model}")
    return config


def run_command(args):
    selection = selection_for(args)
    out = output_directory(args, selection)
    if args.command == "verify":
        return results.verify_archive(args.archive or out / "return.zip")
    if args.command == "report":
        if args.paired_results:
            from . import paired_results

            root = runner.data_directory(args.data_root)
            runner.check_storage(out, writable=True)
            return paired_results.report(args.paired_results, root, out, offline=args.offline)
        if not (out / "evidence/run.json").is_file():
            raise ValueError(f"No plan found in {out}. Use --out for an existing run.")
        runner.check_storage(out, writable=True)
        with runner.exclusive_lock(out / "evidence/run.lock"):
            return runner.report(out)

    scopes, modalities, models = (
        selection[key] for key in ("scopes", "modalities", "models")
    )
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, models, modalities, scopes)
    if args.command == "run":
        from .assets import assert_runtime_environment

        assert_runtime_environment(config, models)
    data_root = runner.data_directory(args.data_root, create=args.command != "doctor")
    if args.command == "doctor":
        from .assets import doctor_environment

        health = doctor_environment(config, models, args.model_root)
        observed = runner.gpu_list()
        allowed = runner.allowed_gpus(
            args.gpus, observed, runner.required_capability(config)
        )
        candidates = {
            model: [
                gpu
                for gpu in allowed
                if runner.idle_for_model(observed[gpu], config["models"][model])
            ]
            for model in models
        }
        return {
            **health,
            "status": (
                "READY"
                if health["ok"] and all(candidates.values())
                else "CHECK_REQUIRED"
            ),
            "eligible_idle_gpus": candidates,
            "data_root": str(data_root),
            "data_storage": runner.check_storage(data_root, writable=True),
            "out": str(out),
            "output_storage": runner.check_storage(out, writable=True),
        }

    runner.check_storage(out, minimum_free=runner.GIB, writable=True)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    with runner.exclusive_lock(out / "evidence/run.lock") as lock_fd:
        manifest, overrides = data.build_selection(
            data_root,
            scopes,
            modalities,
            sample_id=args.sample_id,
            video_name=args.video,
            offline=args.offline,
        )
        run = runner.save_plan(
            out,
            config_path,
            config,
            models,
            manifest,
            json.loads(data.json_bytes(selection)),
        )
        cases = manifest["cases"]
        if args.command == "plan":
            counts = Counter(
                f'{case["domain"]}/{case["task"].lower()}/{case["modality"]}'
                for case in cases
            )
            return {
                "status": "PLANNED",
                "out": str(out),
                "data_root": str(data_root),
                "canonical_cases": run["canonical_cases"],
                "logical_rows": run["logical_rows"],
                "planned_calls": run["planned_calls"],
                "models": models,
                "inputs": dict(sorted(counts.items())),
            }
        if args.command == "download":
            receipt = data.prepare_media(
                data_root,
                cases,
                overrides,
                offline=args.offline,
                check_storage=runner.check_storage,
            )
            assets = {}
            if not args.media_only:
                from .assets import ensure_models

                ready = ensure_models(
                    config, models, args.model_root, download=not args.offline
                )
                for model in models:
                    assets[model] = ready["models"][model]["model_checkpoint"]
            return {
                "status": "DOWNLOADED",
                "out": str(out),
                "data_root": str(data_root),
                "checkpoints": assets,
                **receipt,
            }
        worker_exits = (
            runner.execute(
                out,
                config,
                cases,
                models,
                data_root,
                overrides,
                args.gpus,
                args.model_root,
                offline=args.offline,
                lock_fd=lock_fd,
            )
            or []
        )
        report = runner.report(out)
        failed = sum(worker["exit_code"] != 0 for worker in worker_exits)
        complete = not (
            failed
            or report["missing"]
            or report["uncertain"]
            or report["technical_failures"]
        )
        return {
            "status": "COMPLETE" if complete else "PARTIAL",
            "failed_workers": failed,
            **report,
        }


def main(argv=None):
    args = parse_arguments(argv)
    try:
        response = run_command(args)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return int(response.get("status") in ("PARTIAL", "CHECK_REQUIRED"))
