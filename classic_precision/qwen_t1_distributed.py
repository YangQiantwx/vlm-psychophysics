"""A finite T1 shard with explicit device/ID ownership and cross-host controls.

The accepted production run body is reused byte-for-byte except for the declared
measurement-role field and a comparison gate before fresh production IDs. The
model, native adapter, numerical policy and generation code stay unchanged.
"""
from __future__ import annotations
from copy import deepcopy
import inspect
import json
from pathlib import Path
import socket
import time

from . import qwen_t1_production as production
from .storage import atomic_json, digest, file_sha256

SCHEMA = 'classic_precision_qwen_t1_distributed/v1'
RUNNER_SHA = 'edd47288ea038db36144a806c6e743bf31770ffdfa6e667b49036e82ebc00a9a'
EVIDENCE_MODE = 'cross-host-controls-then-qualified-native-production'
require = production.require


class CrossHostDifference(RuntimeError):
    pass


def normalized(value):
    if isinstance(value, dict):
        return {k:normalized(v) for k,v in value.items()}
    if isinstance(value, list): return [normalized(v) for v in value]
    # Code paths differ by installation root; source hashes remain mandatory.
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def fingerprint(raw, events):
    sid=raw['sample_id'];attempt=raw['attempt_id']
    hashes={}
    for stage in ('cpu_processor','gpu_forward'):
        matches=[e for e in events if e.get('event')=='REPRO_INPUT_HASH' and e.get('sample_id')==sid and e.get('stage')==stage]
        require(len(matches)==1,'Missing unique actual CPU/GPU hash '+sid)
        hashes[stage]=sorted(matches[0]['tensors'],key=lambda t:t['name'])
    posts=[e for e in events if e.get('event')=='POST_OUTPUT_INPUT_RECEIPT' and e.get('sample_id')==sid and e.get('attempt_id')==attempt]
    require(len(posts)==1 and posts[0]['t1_deep_qualification'] is True,'Cross-host control must be actually deep')
    p=posts[0]
    controls={'source_parameters':raw['source_parameters'],'ready_record':raw['ready_record'],
        'input_receipt':raw['input_receipt'],'effective_generation':raw['effective_generation'],
        'actual_input_tokens':raw['actual_input_tokens'],'hashes':hashes,
        'native_token_receipt':p['t1_token_receipt'],'expected_GPU_hashes':p['t1_expected_hashes'],
        'actual_GPU_hashes':p['t1_actual_hashes'],'actual_engine_processor_receipt':p['actual_engine_processor_receipt']}
    return {'sample_id':sid,'controls':normalized(controls),
        'output':{k:raw.get(k) for k in ('raw_answer','output_token_ids','finish_reason','stop_reason')},
        'truth':raw['truth'],'raw_record_sha256':digest(raw)}


def validate(plan):
    require(plan.get('schema')==SCHEMA and plan['passes']==1 and plan['automatic_retry'] is False
        and plan['global_expected']==3840 and plan['source_sha256']==file_sha256(__file__)
        and file_sha256(production.__file__)==RUNNER_SHA,'Distributed source/scope differs')
    require(plan['host']==socket.gethostname() and plan['gpu'] in ('0','1'),'Wrong host/device')
    require(plan['sample_ids']==plan['cross_host_control_ids']+plan['production_ids']
        and plan['expected']==len(plan['sample_ids'])==len(set(plan['sample_ids']))
        and plan['production_expected']==len(plan['production_ids'])>0,'Finite unique shard differs')
    require(plan['replay_implementation']==production.replay.implementation(),'Frozen helper closure changed')
    production.verified(plan['reference_plan']);production.verified(plan['allocation'])
    allocation=json.loads(Path(plan['allocation']['path']).read_text())
    require(allocation['production_by_host_gpu'][plan['allocation_key']]==plan['production_ids'],'Actual ownership allocation differs')
    require(set(plan['cross_host_control_ids']).isdisjoint(plan['production_ids']),'Control/production overlap')
    if plan['cross_host_control_ids']:
        production.verified(plan['cross_host_reference'])
        ref=json.loads(Path(plan['cross_host_reference']['path']).read_text())
        require([r['sample_id'] for r in ref['records']]==plan['cross_host_control_ids'] and len(ref['records'])==48,'Fixed48 cross-host reference differs')
    return plan


def admission(plan):
    anchor=plan['qualified_parent']
    state=json.loads(production.verified(anchor['state']).read_text())
    report=json.loads(production.verified(anchor['report']).read_text())
    require(state['status']=='COMPLETE' and state['comparison_status']=='MATCH'
        and report['evidence_status']=='COMPLETE' and report['comparison_status']=='MATCH'
        and state['plan_sha256']==report['plan_sha256']==anchor['plan_sha256'],'Accepted numerical qualification differs')
    require(all(j['returncode']==0 and j['comparison_status']=='MATCH' for j in state['jobs']),'Prior qualification incomplete')


def context(plan, assets=True):
    replay=production.replay
    ref=replay.native.read(production.verified(plan['reference_plan']))
    coverage=replay.validate_plan(ref,physical=assets)
    bound=replay.native.bind_route(ref,coverage,'video','qwen',assets=assets)
    module=replay.native._native(bound,ref,'qwen');queue,ready=module.input_catalog()
    require(file_sha256(bound['root']/'evaluation_manifest.json')==production.MANIFEST_SHA and len(queue['rows'])==3840,'Full source T1 differs')
    by_id={r['sample_id']:r for r in queue['rows']}
    require(set(plan['sample_ids'])<=set(by_id),'Unknown source ID')
    ordered=[by_id[s] for s in plan['sample_ids']]
    first=[by_id[s] for s in plan['cross_host_control_ids']] if plan['cross_host_control_ids'] else ordered[:2]
    bound['rows']=ordered;bound['entry']=deepcopy(bound['entry'])
    bound['entry']['representative_samples']=[{'sample_id':r['sample_id'],'source_assets':[{
        'ready_file':production.descriptor(bound['root']/'ready'/(r['sample_id']+'.json')),
        'expected_file_sha256':ready[r['sample_id']]['video_sha256']}]} for r in ordered]
    view=deepcopy(ref);view.update(schema=SCHEMA,production_plan_sha256=digest(plan),sample_ids={**ref['sample_ids'],'video':plan['sample_ids']})
    return ref,view,bound,module,ready,first


def cross_host_gate(plan, root):
    if not plan['cross_host_control_ids']: return
    reference=json.loads(production.verified(plan['cross_host_reference']).read_text())
    store=root/'repeat-0/qwen'
    raw=[json.loads(l) for l in (store/'outputs.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (store/'events.jsonl').read_text().splitlines()]
    by_id={r['sample_id']:r for r in raw};comparisons=[]
    for old in reference['records']:
        new=fingerprint(by_id[old['sample_id']],events)
        changes=[k for k in old['controls'] if old['controls'][k]!=new['controls'].get(k)]
        outputs=[k for k in old['output'] if old['output'][k]!=new['output'].get(k)]
        comparisons.append({'sample_id':old['sample_id'],'control_differences':changes,'output_differences':outputs,
            'old_correct':old['output']['raw_answer']==old['truth'],'new_correct':new['output']['raw_answer']==new['truth'],
            'old_raw_sha256':old['raw_record_sha256'],'new_raw_sha256':new['raw_record_sha256']})
    invalid=sum(bool(r['control_differences']) for r in comparisons);changed=sum(bool(r['output_differences']) for r in comparisons)
    status='INVALID_EVIDENCE' if invalid else ('DIFFERENT' if changed else 'MATCH')
    runtime=next(e for e in events if e.get('event')=='REPLAY_RUNTIME')
    report={'comparison_status':status,'evidence_status':'COMPLETE','inputs':48,'controls_different':invalid,'outputs_changed':changed,
        'old_correct':sum(r['old_correct'] for r in comparisons),'new_correct':sum(r['new_correct'] for r in comparisons),
        'reference_host':reference['host'],'new_host':plan['host'],'reference_runtime':reference['runtime'],
        'new_runtime':runtime['snapshot']['comparison_identity'],'comparisons':comparisons,
        'scope':'Exactly48 sealed T1 videos; cross-host observation, not a universal hardware guarantee.'}
    atomic_json(root/'cross-host-acceptance.json',report)
    require(not invalid,'Cross-host actual controls differ; no production launched')
    if changed: raise CrossHostDifference('Cross-host same-input output difference; no production launched')
    if plan.get('production_commit_path'):
        deadline=time.monotonic()+1800
        commit=Path(plan['production_commit_path'])
        while not commit.exists():
            require(time.monotonic()<deadline,'30-minute exact ownership handoff wait exhausted')
            time.sleep(5)
        value=json.loads(commit.read_text())
        require(value['allocation_sha256']==plan['allocation']['sha256']
            and value['production_ids_sha256']==digest(plan['production_ids'])
            and value['predecessor_waiter_retired'] is True,'Ownership handoff is not committed')


def run(path):
    require(file_sha256(production.__file__)==RUNNER_SHA,'Accepted production runner changed')
    source=inspect.getsource(production.run)
    old='                        deep_qualification = '
    require(source.count(old)==1,'Accepted qualification seam changed')
    source=source.replace(old,'                        cross_host_gate(plan, root)\n'+old)
    old='"replay_parent_sample_id": sid,'
    require(source.count(old)==1,'Accepted durable-output seam changed')
    source=source.replace(old,'"replay_parent_sample_id": sid, "measurement_role": "cross_host_control" if sid in plan["cross_host_control_ids"] else "production",')
    source=source.replace('return finish("INVALID_EVIDENCE", error_type=', 'return finish("DIFFERENT" if isinstance(error, CrossHostDifference) else "INVALID_EVIDENCE", error_type=')
    namespace=dict(production.run.__globals__)
    namespace.update(__file__=__file__,SCHEMA=SCHEMA,EVIDENCE_MODE=EVIDENCE_MODE,validate=validate,
        admission=admission,context=context,cross_host_gate=cross_host_gate,CrossHostDifference=CrossHostDifference)
    compiled={};exec(compile(source,__file__+'::accepted_production_body','exec'),namespace,compiled)
    return compiled['run'](path)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run']);parser.add_argument('--plan',required=True)
    args=parser.parse_args();result=run(args.plan);print(json.dumps(result,sort_keys=True));return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
