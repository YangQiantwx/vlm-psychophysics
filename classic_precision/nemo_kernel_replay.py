"""Replay Nemo with the actual legacy CUTLASS kernels explicitly selected.

Uses the public KernelConfig; preserves processor, forward code, weights,
all sampling and existing resources. Separate diagnostic answers only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

from . import nemo_reproduction_diagnosis as base
from .refinement_worker import read,require,write_new
from .storage import atomic_json,file_sha256

CONTRACT={'linear_backend':'cutlass','moe_backend':'cutlass'}


def inventory(worker):
    model=worker.get_model();classes=Counter();backends=set()
    def visit(obj,depth):
        if obj is None or depth>3:return
        kind=type(obj).__module__+'.'+type(obj).__qualname__
        if not kind.startswith('vllm.'):return
        classes[kind]+=1
        for key,value in vars(obj).items():
            if 'backend'in key:backends.add(key+'='+str(value))
            if any(s in key for s in ['kernel','linear','experts']):visit(value,depth+1)
    for _,module in model.named_modules():
        method=getattr(module,'quant_method',None)
        if method is not None:visit(method,0)
    cfg=worker.vllm_config.kernel_config
    return {'classes':dict(classes),'backends':sorted(backends),
        'actual_kernel_config':{'linear_backend':cfg.linear_backend,'moe_backend':cfg.moe_backend,
            'enable_flashinfer_autotune':cfg.enable_flashinfer_autotune}}


def run(path):
    p=read(path)
    require(p['kernel_contract']==CONTRACT,'Only actual legacy backend restoration allowed')
    require(p['replay_implementation']['sha256']==file_sha256(__file__),'Replay entry changed')
    import vllm
    from . import gemma_speed as g
    original=vllm.LLM
    def create(**kwargs):
        require('kernel_config'not in kwargs,'Do not silently override a qualified explicit kernel policy')
        engine=original(**kwargs,kernel_config=CONTRACT)
        owned=engine.collective_rpc(g._worker_identity)
        try:
            selected=engine.collective_rpc(inventory)
            atomic_json(Path(p['output'])/'restored-kernels.json',selected)
            for item in selected:
                require({k:item['actual_kernel_config'][k]for k in CONTRACT}==CONTRACT,'Actual public kernel configuration differs')
                names=' '.join(item['classes'])
                require('CutlassNvFp4LinearKernel'in names and 'CutlassFP8ScaledMMLinearKernel'in names,
                    'Actual mixed NVFP4/FP8 linears did not restore legacy CUTLASS')
                require('FlashInferCutlassNvFp4LinearKernel'not in names and 'FlashInferFP8ScaledMMLinearKernel'not in names,
                    'Unexpected FlashInfer linear retained')
        except BaseException:
            g.shutdown_owned_engine(engine,owned,Path(p['output']));raise
        return engine
    vllm.LLM=create
    try:return base.run(path)
    finally:vllm.LLM=original


def supervise(path):
    import classic_precision.refinement_worker as worker
    original=worker.subprocess.Popen
    def popen(argv,**kwargs):
        return original([s.replace('classic_precision.refinement_worker','classic_precision.nemo_kernel_replay')for s in argv],**kwargs)
    worker.subprocess.Popen=popen
    try:return worker.supervise(path)
    finally:worker.subprocess.Popen=original


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','supervise']);p.add_argument('--profile',required=True)
    a=p.parse_args();print(__import__('json').dumps(run(a.profile)if a.command=='run'else supervise(a.profile),sort_keys=True))


if __name__=='__main__':main()
