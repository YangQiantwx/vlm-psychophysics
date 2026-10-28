"""Versioned X-QSA ledger entry: official vLLM plus one explicit custom selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from . import official_x_runner_v2 as v2
from . import official_x_qsa_selector as selector
from .refinement_worker import descriptor,read,require,write_new
from .storage import digest,file_sha256

base = v2.base
_worker_identity = base.worker_identity
_install_probe = base.install_probe
_read_probe = base.read_probe
_validate_deep = base.validate_deep
_append = base._append


def worker_identity(worker, model):
    result = _worker_identity(worker, model)
    result['custom_qsa_selector'] = selector.evidence()
    return result


def install_probe(worker, spec, model):
    result = _install_probe(worker,spec,model)
    worker._official_x_probe['selector_calls_before'] = selector.evidence()['actual_calls']
    return result


def read_probe(worker, model):
    result = _read_probe(worker,model)
    result['actual_custom_selector_calls_during_request'] = (
        selector.evidence()['actual_calls']-result['selector_calls_before'])
    return result


def validate_deep(p,item,deep):
    for worker in deep:
        receipt = worker['runtime']['custom_qsa_selector']
        require(receipt['status'] == 'APPLIED_X_QSA' and receipt['actual_function_is_canonical_wrapper']
            and receipt['native_function_verified_before_replacement']
            and receipt['selector_source_sha256'] == p['selector_source']['sha256']
            and receipt['mathematics_source_sha256'] == selector.HELPER_SOURCE_SHA
            and receipt['helper_status_after_import'] is None
            and not receipt['helper_import_added_meta_path_hook']
            and worker['actual_custom_selector_calls_during_request'] > 0,
            'The expected canonical custom selector was not actually used')
    _validate_deep(p,item,deep)
    raw = base.records(Path(p['output'])/'raw.jsonl')[-1]
    _append(Path(p['output'])/'custom-selector-admissions.jsonl', {'sample_id':raw['sample_id'],
        'raw_digest':digest(raw),'selector_receipts':[x['runtime']['custom_qsa_selector'] for x in deep],
        'actual_request_calls':[x['actual_custom_selector_calls_during_request'] for x in deep],
        'variant':'X_QSA','pure_official_config':False})


def append(path,value):
    if isinstance(value,dict) and value.get('variant') == 'OFFICIAL_X' and 'output_token_ids' in value:
        value.update(variant='X_QSA',custom_qsa_selector_required=True,pure_official_config=False)
    return _append(path,value)


base.worker_identity = worker_identity
base.install_probe = install_probe
base.read_probe = read_probe
base.validate_deep = validate_deep
base._append = append


def seal(draft, output):
    p = read(draft)
    require(p['model'] == 'qwen_x_qsa', 'X-QSA requires an explicitly distinct model/variant label')
    require(p['engine_kwargs']['worker_extension_cls'] == 'classic_precision.official_x_qsa_selector.XQSAWorkerExtension',
            'Missing explicit QSA worker extension')
    p['qsa_entry'] = descriptor(__file__);p['selector_source'] = descriptor(selector.__file__)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(p,f);path=f.name
    try:
        return v2.seal(path,output)
    finally:
        Path(path).unlink()


def supervise(profile):
    from . import refinement_worker as worker
    native = worker.subprocess.Popen
    def popen(argv,**kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv],**kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['seal','run','supervise','queue'])
    parser.add_argument('--profile');parser.add_argument('--draft');parser.add_argument('--output');parser.add_argument('--plan')
    a = parser.parse_args()
    if a.command == 'seal':result=seal(a.draft,a.output)
    elif a.command == 'run':
        p=read(a.profile);require(file_sha256(__file__)==p['qsa_entry']['sha256'] and
            file_sha256(selector.__file__)==p['selector_source']['sha256'],'The frozen X-QSA entry changed')
        result=v2.run(a.profile)
    elif a.command == 'supervise':result=supervise(a.profile)
    else:
        plan=read(a.plan);require(plan['status']=='SEALED' and plan['automatic_retry'] is False,'Wrong finite X-QSA queue')
        for profile in plan['profiles']:
            result=supervise(profile);print(json.dumps(result),flush=True)
            require(result['returncode']==0 and result['terminal_status']=='COMPLETE' and result['real_wait_observed']
                and result['owned_engines_exited'] and result['original_physical_lease_released'],
                'Previous X-QSA worker did not close COMPLETE')
        result={'status':'COMPLETE','profiles':plan['profiles']}
    print(json.dumps(result))


if __name__ == '__main__':main()
