"""Continue Natural T4 static after the original zero-claim admission failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import natural_t4_x as original

MODULE = "classic_precision.natural_t4_static_recovery"
OLD_PLAN, OLD_RUN = original.PLAN, original.RUN
PLAN = OLD_PLAN / "static-recovery-1"
RUN = OLD_RUN / "static-recovery-1"
CODE, CARDS = original.CODE, original.CARDS
base, static, static_parent, life = original.base, original.static, original.static_parent, original.life
ref, read, require, verified, write_new = original.ref, original.read, original.require, original.verified, original.write_new

# Only this isolated process uses the fresh paths and the inherited queue controller.
original.PLAN, original.RUN, original.MODULE = PLAN, RUN, MODULE


def original_zero_claims():
    launch = read(OLD_RUN/"launch.json")
    require(not life.process_alive(launch["owner"]), "Original controller still owns the queue")
    seal = verified(launch["seal"])
    require(seal["code"] == ref(original.__file__), "Original sealed adapter changed")
    video = ref(OLD_RUN/"video/terminal.json")
    result = verified(video)
    require(result["status"] == "COMPLETE_PASS" and result["model_calls"] == 6144, "Original video phase is incomplete")
    waits = read(OLD_RUN/"static/outer-waits.json")
    require(set(waits) == set(CARDS), "Missing failed card waits")
    for card in CARDS:
        outer = waits[card]
        require(outer["returncode"] == 1 and outer["real_wait_observed"] and not life.process_alive(outer["child"]),
                "Original static queue has not exited")
        for profile in seal["sets"]["static"][card]:
            p = verified(profile)
            out, claims = Path(p["output"]), Path(p["claim_root"])
            require(not out.exists() and not claims.exists(), "Original static already has output or claims")
            if static.model_key(p) == "cosmos":
                wait = read(out.with_suffix(".wait.json"))
                require(wait["returncode"] == 1 and wait["real_wait_observed"] and wait["completed_ids"] == []
                        and wait["owned_engines_exited"] and wait["original_physical_lease_released"],
                        "Original Cosmos static boundary is uncertain")
    return launch["seal"], seal["sets"]["static"], video


def review_gate():
    review = read(PLAN/"review.json")
    require(review["status"] == "PASS_ASTRA_XHIGH_STATIC_RECOVERY" and review["code"] == ref(__file__)
            and review["original_code"] == ref(original.__file__), "Missing targeted recovery review")


def policy(p):
    require(p["modality"] == "static" and p["execution_policy"] == {"module": MODULE, "source_sha256": ref(__file__)["sha256"]},
            "Wrong recovery modality or source")
    review_gate()
    catalog = original.config(p)
    require(p["static_entry"] == ref(static.__file__) and p["static_dependencies"] == static.dependencies(),
            "Frozen static dependencies changed")
    static_parent.verify_cpu(p, catalog)
    static_parent.qualified_deployment(p)
    return catalog


original.policy = policy


def input_validator(p, item, raw, evidence):
    return original.input_validator(p, item, raw, evidence)


def deep_policy(p, item, ordinal):
    return original.deep_policy(p, item, ordinal)


def runtime(p, actual):
    return original.runtime(p, actual)


def run(path, gate=None):
    p = read(path)
    protocol = verified(p["static_protocol"])
    require(protocol["scope"] == original.NAME, "Wrong Natural T4 static protocol")
    check, start = static.check_item, static.start_gate

    def scoped_check(item, candidate):
        if candidate == protocol:
            return original.check_static(item, candidate)
        return check(item, candidate)

    static.check_item, static.start_gate = scoped_check, gate or original.start_gate
    try:
        return static.execute_bound_profile(path, policy_validator=policy, input_validator=input_validator,
                                            deep_policy=deep_policy, runtime_validator=runtime)
    finally:
        static.check_item, static.start_gate = check, start


def prepare():
    old_seal, profiles, video = original_zero_claims()
    sets = {"static": {c: [] for c in CARDS}}
    for card in CARDS:
        for descriptor in profiles[card]:
            p = verified(descriptor)
            out = RUN/"static"/p["worker"]
            p.update(output=str(out), claim_root=str(out)+"-claims",
                     pause_file=str(out.with_name(p["worker"]+"-pause.json")),
                     execution_policy={"module":MODULE,"source_sha256":ref(__file__)["sha256"]})
            catalog = original.config(p)
            static_parent.verify_cpu(p, catalog)
            target = PLAN/"static"/(p["worker"]+"-profile.json")
            with tempfile.TemporaryDirectory(prefix="natural-t4-static-reseal-") as directory:
                draft = Path(directory)/"draft.json"
                write_new(draft, p)
                base.seal(draft, target)
            repaired, frozen = read(target), verified(descriptor)
            changed = {k for k in set(repaired)|set(frozen) if repaired.get(k) != frozen.get(k)}
            require(changed == {"output", "claim_root", "pause_file", "execution_policy"},
                    f"Unexpected static profile changes: {sorted(changed)}")
            sets["static"][card].append(ref(target))
    write_new(PLAN/"sealed.json", {"status":"STATIC_RECOVERY_ONLY", "sets":sets,
                                   "code":ref(__file__), "original_seal":old_seal, "video_terminal":video})


class BeforeGPU(Exception):
    pass


def probe(path):
    p = read(path)
    def gate(loaded):
        require(loaded == p, "Probed profile changed")
        raise BeforeGPU()
    try:
        run(path, gate)
    except BeforeGPU:
        require(not Path(p["output"]).exists() and not Path(p["claim_root"]).exists(),
                "Probe crossed the GPU/claim boundary")
        write_new(Path(path).with_suffix(".probe.json"), {"status":"PASS_EXACT_WORKER_ADMISSION_BEFORE_GPU",
                                                        "profile":ref(path), "review":ref(PLAN/"review.json"),
                                                        "model_calls":0})
    else:
        raise RuntimeError("Probe did not reach the checkpoint/start boundary")


def probe_all():
    sealed = read(PLAN/"sealed.json")
    def check(job):
        profile = verified(job)
        subprocess.run(original.argv(profile, "probe", "--profile", job["path"]), cwd=CODE,
                       env=original.environment(profile), check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, (d for c in CARDS for d in sealed["sets"]["static"][c])))


def launch():
    review_gate()
    seal = ref(PLAN/"sealed.json")
    sealed = verified(seal)
    old_seal, _, video = original_zero_claims()
    require(sealed["status"] == "STATIC_RECOVERY_ONLY" and sealed["code"] == ref(__file__)
            and sealed["original_seal"] == old_seal and sealed["video_terminal"] == video, "Wrong static recovery seal")
    for card in CARDS:
        for profile in sealed["sets"]["static"][card]:
            probe = read(Path(profile["path"]).with_suffix(".probe.json"))
            require(probe == {"status":"PASS_EXACT_WORKER_ADMISSION_BEFORE_GPU", "profile":profile,
                              "review":ref(PLAN/"review.json"), "model_calls":0}, "Missing exact worker admission probe")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Controller must be CPU-only")
    require(not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]).strip(),
            "Another compute workload is present")
    RUN.mkdir(parents=True)
    write_new(RUN/"launch.json", {"owner":life.process_identity(), "seal":seal, "automatic_retry":False})
    phase = RUN/"static"
    phase.mkdir()
    children = {}
    for card in CARDS:
        p = verified(sealed["sets"]["static"][card][0])
        with (phase/(card+".log")).open("xb") as log:
            child = subprocess.Popen(original.argv(p,"queue","static",card), cwd=CODE,
                                     env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children[card] = (child, life.process_identity(child.pid))
    waits = {c:{"child":i,"returncode":child.wait(),"real_wait_observed":True} for c,(child,i) in children.items()}
    write_new(phase/"outer-waits.json", waits)
    require(all(w["returncode"] == 0 and not life.process_alive(w["child"]) for w in waits.values()),
            "Static recovery card queue failed")
    original.finish("static", sealed["sets"]["static"], waits)
    write_new(RUN/"terminal.json", {"status":"COMPLETE_PASS","model_calls":12288,"recovery_static_calls":6144,
                                   "video":video,"static":ref(phase/"terminal.json"),
                                   "original_static_failure":ref(OLD_RUN/"static/outer-waits.json"),
                                   "review":ref(PLAN/"review.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","probe_all","launch","probe","run","supervise","queue"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--profile")
    args = parser.parse_args()
    actions = {"prepare":prepare,"probe_all":probe_all,"launch":launch,"probe":probe,"run":run,
               "supervise":original.supervise,"queue":original.queue}
    if args.profile is not None and args.args:
        parser.error("Use either --profile or positional arguments.")
    arguments = [args.profile] if args.profile is not None else args.args
    actions[args.action](*arguments)


if __name__ == "__main__":
    main()
