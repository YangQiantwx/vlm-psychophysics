"""Compose the ordered T1 observation with the explicitly required X-QSA selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

# Import order is part of this sealed entry: T1 probes retain QSA call receipts.
from . import official_x_runner_qsa as qsa
from . import official_x_t1_images as t1
from .refinement_worker import descriptor,read,require
from .storage import file_sha256


def seal(draft, output):
    p=read(draft)
    require(p['model']=='qwen_x_qsa' and p['engine_kwargs']['worker_extension_cls']==
        'classic_precision.official_x_qsa_selector.XQSAWorkerExtension','Missing explicit T1 X-QSA configuration')
    p['t1_qsa_entry']=descriptor(__file__);p['qsa_entry']=descriptor(qsa.__file__)
    p['selector_source']=descriptor(qsa.selector.__file__)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(p,f);path=f.name
    try:return t1.seal(path,output)
    finally:Path(path).unlink()


def supervise(profile):
    from . import refinement_worker as worker
    native=worker.subprocess.Popen
    def popen(argv,**kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv],**kwargs)
    worker.subprocess.Popen=popen
    try:return worker.supervise(profile)
    finally:worker.subprocess.Popen=native


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['seal','run','supervise','queue'])
    p.add_argument('--profile');p.add_argument('--draft');p.add_argument('--output');p.add_argument('--plan');a=p.parse_args()
    if a.command=='seal':result=seal(a.draft,a.output)
    elif a.command=='run':
        value=read(a.profile)
        for key,path in [('t1_qsa_entry',__file__),('t1_entry',t1.__file__),('qsa_entry',qsa.__file__),
                         ('selector_source',qsa.selector.__file__)]:
            require(file_sha256(path)==value[key]['sha256'],'The frozen T1 X-QSA source changed')
        result=t1.v2.run(a.profile)
    elif a.command=='supervise':result=supervise(a.profile)
    else:
        plan=read(a.plan);require(plan['status']=='SEALED' and plan['automatic_retry'] is False,'Wrong finite T1 X-QSA queue')
        for profile in plan['profiles']:
            result=supervise(profile);print(json.dumps(result),flush=True)
            require(result['returncode']==0 and result['terminal_status']=='COMPLETE' and result['real_wait_observed']
                and result['owned_engines_exited'] and result['original_physical_lease_released'],'Prior T1 worker did not close COMPLETE')
        result={'status':'COMPLETE','profiles':plan['profiles']}
    print(json.dumps(result))


if __name__=='__main__':main()
