"""Release only finished recorder objects; preserve sealed inference profiles.

The observed PyTorch profiler owns bound methods through action_map, creating a
cycle which can retain its native Kineto event buffers after every request. This
overlay detaches those finished buffers after the original trace has been saved.
It neither changes global GC nor calls a model/sampler or CUDA allocator API.
"""
import argparse
from contextlib import contextmanager
import gc
import inspect
import json
from pathlib import Path
import sys
import time
import weakref

from . import gemma_x_radon_on_confirm as on
from . import gemma_x_recheck_safe_continue as continuation
from . import gemma_x_recheck_observer_safe as observer
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import _append, file_sha256

core = on.core
BASE = on.BASE
MODULE = "classic_precision.gemma_x_recheck_memory_safe"
TORCH_PROFILER_SHA = "48829e86a5a28480b5c6e36529206efee393aace7fbc315020cbb7ced2b6b43e"
QUEUES = {
    "radon0": "01c0cc3c09f316c1333742c02118dfdeb546bd6cc0cb12754425eef485c3bea9",
    "fourier0": "c3ceea7c7e197b3c5344ade2dbf7bf02ebf7fd313225d5728e749dcf6995bdd1",
    "fourier1": "dcc9bfda34cf5c219d2eef0f630d075b2594071a1484c653247e6bd53b00e645",
}
ORIGINAL_BEGIN = observer.begin
ORIGINAL_FINISH = observer.finish


def namespace_name(card):
    require(card in QUEUES, "Unknown card")
    return on.NAME if card == "radon0" else continuation.name(card)


def process_memory():
    path = Path("/proc/self/status")
    return {key: int(value.split()[0]) for line in path.read_text().splitlines()
            if ":" in line for key, value in [line.split(":", 1)]
            if key in ("VmRSS", "VmHWM", "VmSwap")} if path.exists() else {}


def begin(worker, sid):
    import torch
    state = worker._gemma_recheck_trace
    if "memory_lifecycle_source_checked" not in state:
        source = inspect.getsourcefile(torch.profiler.profile)
        require(file_sha256(source) == TORCH_PROFILER_SHA,
                "Profiler source differs from the inspected lifecycle")
        state["memory_lifecycle_source_checked"] = descriptor(source)
    state["memory_before_request"] = process_memory()
    state.setdefault("memory_initial_request", state["memory_before_request"])
    return ORIGINAL_BEGIN(worker, sid)


def release_finished_profiler(profiler):
    """Called only after the original finish returns or raises; never stop twice."""
    if profiler is None:
        return {"present": False}
    actions = getattr(profiler, "action_map", None)
    require(isinstance(actions, dict), "Unexpected profiler lifecycle object")
    action_count = len(actions)
    had_buffers = getattr(profiler, "profiler", None) is not None
    # The profiler is exclusively ours, has finished inference, and its export
    # has already been attempted. Do not retain native buffers until cyclic GC.
    actions.clear()
    profiler.profiler = None
    return {"present": True, "released_action_entries": action_count,
            "detached_native_event_owner": had_buffers}


def finish(worker):
    state = worker._gemma_recheck_trace
    profiler = state.get("profiler")
    weak = weakref.ref(profiler) if profiler is not None else None
    gc_enabled = gc.isenabled()
    before = process_memory()
    started = time.monotonic()
    try:
        result = ORIGINAL_FINISH(worker)
    finally:
        release = release_finished_profiler(profiler)
        # An exceptional original finish may not yet have popped its reference.
        if state.get("profiler") is profiler:
            state.pop("profiler", None)
        del profiler
        record = {"sample_id": state["sid"], "release": release,
                  "profiler_destroyed": weak is None or weak() is None,
                  "gc_enabled_before": gc_enabled,
                  "gc_enabled_after": gc.isenabled(),
                  "before_request_kib": state.get("memory_before_request", {}),
                  "before_finish_kib": before, "after_release_kib": process_memory(),
                  "finish_and_release_seconds": time.monotonic() - started,
                  "math_sampling_or_global_gc_changed": False}
        _append(state["root"] / "memory-lifecycle.jsonl", record)
        require(gc_enabled == gc.isenabled(), "Recorder changed global GC state")
    # Preserve the original output descriptor and every observed tensor byte.
    initial_rss = state.get("memory_initial_request", {}).get("VmRSS")
    current_rss = record["after_release_kib"].get("VmRSS")
    # Stop after durable raw/diagnostics, rather than allowing recorder growth
    # to consume the entire host. This is a recorder guard, not an engine kwarg.
    require(initial_rss is None or current_rss is None
            or current_rss - initial_rss <= 16 * 1024**2,
            "Recorder RSS grew over16GiB; original answer and trace saved, stop without retry")
    return {**result, "memory_lifecycle": record}


def overlay_path(root, card):
    return Path(root) / "runs/reproducibility" / namespace_name(card) / (card + "-memory-observer-admission.json")


def admission(root, card):
    path = overlay_path(root, card)
    spec = read(path)
    require(spec["status"] == "AUTHORIZED_RECORDER_MEMORY_REPAIR_ONLY"
            and spec["entrypoint"] == descriptor(__file__)
            and spec["science_changes"] == [] and spec["new_budget"] == 0
            and spec["card"] == card, "Invalid recording-only admission")
    qp = Path(root) / "plans" / namespace_name(card) / (card + "-recheck-queue.json")
    require(spec["sealed_queue"] == descriptor(qp)
            and file_sha256(qp) == QUEUES[card], "Sealed queue changed")
    q = read(qp)
    require(q["profiles"] == spec["sealed_profiles"], "Sealed profiles changed")
    for ref in q["profiles"]:
        verified(ref)
    intent = path.parent / (card + "-recheck-queue.launch-intent.json")
    if intent.exists():
        require(read(intent).get("diagnostic_memory_overlay") == descriptor(path),
                "Existing launch was not bound to this observer; never redispatch")
    return spec


def validate_forced_terminal(records, spec):
    closed = verified(spec["authorized_abnormal_closure"])
    signal = verified(spec["authorized_single_worker_termination"])
    receipt = verified(spec["terminal_return_receipt"])
    require(receipt["status"] == "PASS_COMPLETE_ABNORMAL_TERMINAL_ALL_FILE_SHA_RETURN"
            and receipt["all_file_SHA_verified"] and receipt["total_raw"] == 71,
            "Complete terminal archive is missing")
    require(signal["status"] == "AUTHORIZED_SIGKILL_SENT_ONCE_TO_EXACT_OWNED_WORKER"
            and signal["authorized_pid"] == 315266 and signal["startticks"] == "6400022"
            and signal["boot_id"] == "bb9cf0ec-41bd-4937-a6a6-6efdb08077ad"
            and signal["signal"] == 9 and not signal["signals_to_other_processes"],
            "Different termination or missing explicit exception")
    require(closed["status"] == "AUTHORIZED_OOM_STALLED_WORKER_TERMINATED_TRUE_ABNORMAL_CLOSURE"
            and records["outer_wait"] == closed["outer_wait"]
            and records["outer_wait"]["returncode"] == 1
            and records["outer_wait"]["real_wait_observed"]
            and records["outer_owner"]["status"] == "OWNER_PROCESS_EXITED"
            and not records["outer_owner"]["alive"] and len(records["workers"]) == 2,
            "OFF queue is not actually closed")
    for index, (actual, expected) in enumerate(zip(records["workers"], closed["workers"])):
        require(actual["profile"] == expected["profile"]
                and actual["claimed_ids"] == expected["claimed_ids"]
                and actual["claims"] == actual["raw"] == (70 if index == 0 else 0)
                and len(actual["assigned_ids"]) == expected["assigned_count"]
                and actual["wait"] == expected["wait"]
                and actual["same_inode_lease_free"] and actual["live_owner_count"] == 0,
                "OFF ownership, claims or original lease differ")
        if index == 0:
            w = actual["wait"]
            require(actual["started"] and w["returncode"] == -9
                    and w["real_wait_observed"] and w["owned_engines_exited"]
                    and w["original_physical_lease_released"]
                    and w["profile_sha256"] == actual["profile_digest"],
                    "Authorized abnormal worker wait missing")
        else:
            require(not actual["started"] and actual["wait"] is None,
                    "Unstarted second OFF worker unexpectedly launched")


def prepare_overlay(root, card):
    root = Path(root)
    target = overlay_path(root, card)
    require(not target.exists(), "Overlay already sealed; never overwrite")
    run = target.parent
    require(not any((run / (card + "-recheck-queue." + x + ".json")).exists()
                    for x in ("launch-intent", "launch", "owner", "wait")),
            "Diagnostic queue already attempted")
    qp = root / "plans" / namespace_name(card) / (card + "-recheck-queue.json")
    require(file_sha256(qp) == QUEUES[card], "Original sealed queue changed")
    q = read(qp)
    for ref in q["profiles"]:
        p = verified(ref)
        c = read(p["control_file"])
        require(c["paused"] and c["generation"] == 0
                and not list(Path(p["claim_root"]).glob("*.json"))
                and not (Path(p["output"]) / "raw.jsonl").exists(),
                "An already-used profile cannot receive this overlay")
    spec = {"status": "AUTHORIZED_RECORDER_MEMORY_REPAIR_ONLY", "card": card,
            "entrypoint": descriptor(__file__), "sealed_queue": descriptor(qp),
            "sealed_profiles": q["profiles"], "science_changes": [], "new_budget": 0,
            "profiler_source_sha256": TORCH_PROFILER_SHA,
            "repair": "Release finished profiler action-map cycle and native event owner after original export; preserve global GC, input and output",
            "all_original_records_retained": True}
    if card == "radon0":
        old = root / "runs/reproducibility" / BASE / "autotune-off3"
        spec.update(authorized_abnormal_closure=descriptor(old / "stalled-worker-closed-evidence.json"),
                    authorized_single_worker_termination=descriptor(old / "stalled-worker-termination-signal-receipt.json"),
                    terminal_return_receipt=descriptor(old / "oom-terminal-return-receipt.json"),
                    human_instruction="仅结束卡住worker，修复后继续（推荐）")
    write_new(target, spec)
    return {"status": spec["status"], "admission": descriptor(target),
            "profiles_resealed": False, "model_calls": 0}


@contextmanager
def overlay(root, card):
    spec = admission(root, card)
    target = overlay_path(root, card)
    intent = target.parent / (card + "-recheck-queue.launch-intent.json")
    old = (observer.begin, observer.finish, on.MODULE, continuation.MODULE,
           on.validate_terminal, core.write_new)
    original_write = core.write_new
    def bound_write(path, value):
        if Path(path) == intent:
            value = {**value, "diagnostic_memory_overlay": descriptor(target)}
        return original_write(path, value)
    observer.begin, observer.finish = begin, finish
    on.MODULE = continuation.MODULE = MODULE
    core.write_new = bound_write
    if card == "radon0":
        on.validate_terminal = lambda records: validate_forced_terminal(records, spec)
    try:
        yield
    finally:
        (observer.begin, observer.finish, on.MODULE, continuation.MODULE,
         on.validate_terminal, core.write_new) = old


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command")
    for key in ("root", "card", "plan", "profile"):
        parser.add_argument("--" + key)
    args, _ = parser.parse_known_args()
    if args.profile:
        p = read(args.profile); card, root = p["card"], p["workspace_root"]
    elif args.plan:
        q = read(args.plan); card = q["recheck"]["card"]
        root = verified(q["profiles"][0])["workspace_root"]
    else:
        card, root = args.card or "radon0", args.root
    require(root is not None, "Missing exact workspace")
    if args.command == "prepare-overlay":
        print(json.dumps(prepare_overlay(root, card)), flush=True)
        return
    require(args.command in ("launch", "queue-supervisor", "queue", "supervise", "run"),
            "No new preparation, seal or experiment is authorized")
    with overlay(root, card):
        if card == "radon0":
            if args.command == "launch" and "--card" in sys.argv:
                at = sys.argv.index("--card"); del sys.argv[at:at + 2]
            on.main()
        else:
            continuation.main()


if __name__ == "__main__":
    main()
