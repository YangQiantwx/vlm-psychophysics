"""Opt-in queue deferral; scientific runners and completion counts stay intact."""
import functools
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path


BASE_SCHEDULER_SHA256 = "2f482b7fee513c407bfa867775d259df564e7301286550a0ffed71183fa65b0a"
ENVIRONMENT_VARIABLE = "CLASSIC_PRECISION_DEFER_MODELS"
SELECTION_SCOPE = ("cosmos", "nemo", "qwen")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install():
    """Defer Gemma explicitly; reuse the original resource admission for each model."""
    requested = os.environ.get(ENVIRONMENT_VARIABLE)
    if requested is None:
        return None
    if requested != "gemma":
        raise ValueError(ENVIRONMENT_VARIABLE + " must be exactly gemma")
    spec = importlib.util.find_spec("classic_precision.scheduler")
    if spec is None or spec.origin is None or _sha(spec.origin) != BASE_SCHEDULER_SHA256:
        raise RuntimeError("Classic precision base scheduler SHA mismatch")
    scheduler = importlib.import_module("classic_precision.scheduler")
    if _sha(scheduler.__file__) != BASE_SCHEDULER_SHA256:
        raise RuntimeError("Classic precision base scheduler SHA mismatch")
    original = scheduler.choose_model
    installed = getattr(original, "_classic_precision_queue_override", None)
    if installed is not None:
        return dict(installed)
    original_json = scheduler.atomic_json
    evidence = {"environment_variable": ENVIRONMENT_VARIABLE, "deferred_models": ["gemma"],
                "selection_scope": list(SELECTION_SCOPE),
                "helper_source_sha256": _sha(__file__),
                "base_scheduler_sha256": BASE_SCHEDULER_SHA256}

    @functools.wraps(original)
    def choose_model(pending, specs, snapshot):
        active = [name for name in pending if name != "gemma"]
        # Calling the original chooser preserves both GPU and host-RAM gates.
        for name in SELECTION_SCOPE:
            if name in active and original([name], specs, snapshot) == name:
                return name
        return original([name for name in active if name not in SELECTION_SCOPE], specs, snapshot)

    @functools.wraps(original_json)
    def atomic_json(path, value):
        name = Path(path).name
        shard = name.removeprefix("supervisor-shard-").removesuffix(".json")
        if name == "supervisor-shard-" + shard + ".json" and shard.isdigit():
            value = {**value, "deferred_models": ["gemma"], "selection_scope": list(SELECTION_SCOPE)}
        return original_json(path, value)

    choose_model._classic_precision_queue_override = evidence
    scheduler.choose_model = choose_model
    scheduler.atomic_json = atomic_json
    return dict(evidence)
