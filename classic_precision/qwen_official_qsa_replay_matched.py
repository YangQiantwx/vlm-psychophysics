"""Official replay with explicit historical CUTLASS backends and fail-closed queue.

The already sealed first runner remains unchanged. This version adds only public
KernelConfig controls, actual backend inventory, and pre-load pause/queue gates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import inspect
import json
from pathlib import Path

from . import qwen_official_qsa_replay as base
from .refinement_worker import Boundary, descriptor, read, require, write_new
from .storage import file_sha256

_native_evidence = base.native_qsa_evidence


def native_qsa_evidence(worker=None):
    evidence = _native_evidence(worker)
    if worker is None:
        return evidence
    counts, examples, backends = Counter(), defaultdict(list), Counter()
    def label(value):
        cls = value if inspect.isclass(value) else type(value)
        return cls.__module__ + '.' + cls.__name__
    for name, module in worker.get_model().named_modules():
        method = getattr(module, 'quant_method', None)
        if method is None:
            continue
        counts['quant_method:' + label(method)] += 1
        backend = getattr(method, 'nvfp4_backend', None)
        if backend is not None:
            backends[str(backend)] += 1
        for attr, value in vars(method).items():
            if 'kernel' not in attr and 'linear' not in attr:
                continue
            kind = label(value)
            if kind.startswith(('vllm.', 'flashinfer.')):
                counts[attr + ':' + kind] += 1
                if len(examples[kind]) < 2:
                    examples[kind].append(name)
    require(backends and all(x.endswith('.VLLM_CUTLASS') for x in backends),
            'Actual NVFP4 MoE backend differs from historical VLLM_CUTLASS')
    nv = {k:v for k,v in counts.items() if 'NvFp4LinearKernel' in k}
    fp8 = {k:v for k,v in counts.items() if 'FP8ScaledMMLinearKernel' in k}
    require(nv and all(k.endswith('.CutlassNvFp4LinearKernel') for k in nv),
            'Actual NVFP4 linear kernel differs from original native selector')
    require(fp8 and all(k.endswith('.CutlassFP8ScaledMMLinearKernel') for k in fp8),
            'Actual FP8 linear kernel differs from original native selector')
    config = worker.vllm_config.kernel_config
    require(config.moe_backend == config.linear_backend == 'cutlass',
            'Public CUTLASS KernelConfig did not take effect')
    evidence['actual_math_backends'] = {'kernel_config':{'moe_backend':config.moe_backend,
        'linear_backend':config.linear_backend}, 'nvfp4_moe':dict(backends),
        'kernel_and_method_counts':dict(counts), 'representative_module_paths':dict(examples),
        'model_runner_class':label(worker.model_runner),
        'historical_linear_evidence':'Unchanged original 0.31 native selector CPU replay; no historical full-model inventory claimed'}
    return evidence


base.native_qsa_evidence = native_qsa_evidence


@contextmanager
def matched_file_loader_settings(root):
    import vllm
    from vllm.config import KernelConfig
    native = vllm.LLM
    def create(**kwargs):
        kwargs.update(media_io_kwargs=base.original.MEDIA, allowed_local_media_path=str(root),
                      kernel_config=KernelConfig(moe_backend='cutlass', linear_backend='cutlass'))
        return native(**kwargs)
    vllm.LLM = create
    try:
        yield
    finally:
        vllm.LLM = native


def run(profile_path):
    p = read(profile_path)
    require(file_sha256(__file__) == p['backend_entry']['sha256'], 'Sealed matched entry changed')
    boundary = Boundary(p)
    if boundary.check():
        out = Path(p['output'])
        require(not out.exists(), 'No implicit resume')
        out.mkdir(parents=True)
        result = {'status':'PAUSED_BEFORE_LOAD', 'completed_ids':[], 'UNCERTAIN_ids':[],
                  'unattempted_ids':p['assigned_ids'], 'automatic_retry':False,
                  'formal_auto_resume':False, 'model_load_started':False}
        write_new(out/'result.json', result)
        return result
    previous = base.original.file_loader_settings
    base.original.file_loader_settings = matched_file_loader_settings
    try:
        return base.run(profile_path)
    finally:
        base.original.file_loader_settings = previous


def supervise(profile_path):
    from . import refinement_worker as worker
    native_popen = worker.subprocess.Popen
    def popen(argv, **kwargs):
        argv = [x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv]
        return native_popen(argv, **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile_path)
    finally:
        worker.subprocess.Popen = native_popen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run','supervise','queue','compare'])
    parser.add_argument('--profile'); parser.add_argument('--plan')
    a = parser.parse_args()
    if a.command == 'run':
        print(json.dumps(run(a.profile)))
    elif a.command == 'supervise':
        print(json.dumps(supervise(a.profile)))
    elif a.command == 'compare':
        base.compare(a.plan)
    else:
        plan = read(a.plan)
        require(plan['status'] == 'SEALED' and plan['maximum_model_calls'] == 96, 'Wrong finite plan')
        for path in plan.get('technical_predecessor_waits', []):
            wait = read(path)
            require(wait['real_wait_observed'] and wait['owned_engines_exited']
                    and wait['original_physical_lease_released']
                    and not wait['completed_ids'],
                    'Technical predecessor has not exited without scientific requests')
        for profile in plan['profiles']:
            result = supervise(profile)
            print(json.dumps(result), flush=True)
            require(result['returncode'] == 0 and result['terminal_status'] == 'COMPLETE'
                    and result['real_wait_observed'] and result['owned_engines_exited']
                    and result['original_physical_lease_released'],
                    'Predecessor did not completely close; do not start another engine')


if __name__ == '__main__':
    main()
