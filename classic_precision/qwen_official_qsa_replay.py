"""Finite official-vLLM replay of one preselected original C cell, two cold loads.

The original reader experiment is reused without editing it. No custom QSA
selector, import hook, Torch policy, retry, or implicit resume is installed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from contextlib import ExitStack
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import marshal
import os
from pathlib import Path
import signal
import socket
import statistics
import sys
import time
from types import CodeType

from .storage import atomic_json, _append, digest, file_sha256, owner_identity
from .refinement_worker import (Boundary, descriptor, physical_lease, read,
                                require, verified, write_new)
from . import reader_route_smoke as original

NAME = 'qwen-official-qsa-replay-20261028'
QSA = 'vllm.models.qwen4_exp.nvidia.ops.qsa_indexer'
COMMIT = 'f4fd1ab68e2ba3a76bf6b903662eb6a8ee8202ab'
VERSION = '0.33.0rc1.dev175+gf4fd1ab68'


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(
        __import__('torch').uint8).numpy().tobytes()).hexdigest()


def native_qsa_evidence(worker=None):
    """Check the live function against compiled official source, not an env flag."""
    import torch
    import vllm
    qsa = importlib.import_module(QSA)
    function = qsa._topk
    source = Path(qsa.__file__).read_bytes()
    compiled = compile(source, qsa.__file__, 'exec', dont_inherit=True)
    expected = [x for x in compiled.co_consts
                if isinstance(x, CodeType) and x.co_name == '_topk']
    modules = [n for n in sys.modules if n.rsplit('.', 1)[-1]
               in ('classic_precision_qsa_determinism', 'classic_precision_qsa_only')]
    hooks = [type(x).__module__ + '.' + type(x).__name__ for x in sys.meta_path
             if 'classic_precision_qsa' in type(x).__module__]
    exact = (inspect.isfunction(function) and len(expected) == 1
             and marshal.dumps(function.__code__) == marshal.dumps(expected[0])
             and function.__globals__ is vars(qsa)
             and function.__module__ == QSA
             and not hasattr(function, '__wrapped__')
             and not hasattr(function, '_classic_precision_qsa_determinism'))
    binary = importlib.import_module('vllm._C_stable_libtorch')
    from vllm.platforms import current_platform
    actual = {'function_source_exact': exact, 'custom_selector_modules': modules,
              'custom_selector_import_hooks': hooks, 'qsa_file': qsa.__file__,
              'qsa_sha256': hashlib.sha256(source).hexdigest(),
              'topk_source': inspect.getsource(function),
              'topk_code_sha256': hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
              'persistent_schema': str(torch.ops._C.persistent_topk.default._schema),
              'persistent_dispatch': torch._C._dispatch_dump_table('_C::persistent_topk'),
              'binary_file': binary.__file__, 'binary_sha256': file_sha256(binary.__file__),
              'binary_mapped': str(Path(binary.__file__).resolve()) in Path('/proc/self/maps').read_text(),
              'vllm_version': vllm.__version__, 'vllm_file': vllm.__file__,
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'sm120': current_platform.is_device_capability_family(120),
              'device_capability': list(torch.cuda.get_device_capability())}
    require(exact and not modules and not hooks and actual['binary_mapped'],
            'Actual custom QSA replacement/hook or non-source function detected')
    require(actual['sm120'] and actual['vllm_version'] == VERSION,
            'Official version/physical architecture changed')
    actual['actual_topk_branch'] = 'persistent_topk; native _topk excludes SM120 from cooperative path'
    return actual


def install_deep_probe(worker):
    """Read the real visual tensor and observe actual native TopK operator calls."""
    import torch
    worker._official_deep = []
    model = worker.get_model()
    visual = getattr(model, 'vision_model', None)
    if visual is None:
        visual = model.visual
    def hook(module, args, kwargs):
        x = args[0]
        grid = kwargs.get('grid_thw')
        worker._official_deep.append({'shape': list(x.shape), 'dtype': str(x.dtype),
            'device': str(x.device), 'grid_thw': grid.tolist() if grid is not None else None,
            'sha256': tensor_sha(x)})
    worker._official_vision_hook = visual.register_forward_pre_hook(hook, with_kwargs=True)
    worker._official_profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
    worker._official_profiler.__enter__()
    return native_qsa_evidence(worker)


def read_deep_probe(worker):
    worker._official_vision_hook.remove()
    profiler = worker._official_profiler
    profiler.__exit__(None, None, None)
    ops = [{'key': x.key, 'count': x.count} for x in profiler.key_averages()
           if 'persistent_topk' in x.key or 'cooperative_topk' in x.key]
    del worker._official_profiler
    require(any('persistent_topk' in x['key'] and x['count'] > 0 for x in ops),
            'No actual native persistent_topk call observed')
    return {'actual_GPU_input': worker._official_deep, 'actual_topk_operator_calls': ops,
            'native_qsa': native_qsa_evidence(worker)}


class DeepNativeObservation(original.NativeObservation):
    deep = False
    def describe(self, data):
        result = super().describe(data)
        if self.deep:
            for key, value in data.items():
                if 'pixel' in key and hasattr(value, 'shape'):
                    result[key]['sha256'] = tensor_sha(value)
        return result

    def __enter__(self):
        super().__enter__()
        from vllm.multimodal.media.video import VideoMediaIO
        base = VideoMediaIO.load_file
        def load(loader, path):
            result = base(loader, path)
            if self.active and self.deep:
                frames, metadata = result.media
                self.reads[-1]['decoded_sha256'] = hashlib.sha256(frames.tobytes()).hexdigest()
                self.reads[-1]['per_frame_sha256'] = [hashlib.sha256(x.tobytes()).hexdigest() for x in frames]
            return result
        VideoMediaIO.load_file = load
        return self


def validate_binding(profile, evidence):
    expected = verified(profile['official_runtime'])
    require(evidence['qsa_sha256'] == expected['qsa_sha256']
            and evidence['binary_sha256'] == expected['binary_sha256']
            and evidence['vllm_file'].startswith(expected['environment'] + '/'),
            'Official live source/binary/environment changed')


def seal(root, runtime_path):
    root = Path(root)
    plan = root/'plans'/NAME
    require(not plan.exists(), 'No reseal/resume allowed')
    old = root/'plans/reader-route-smoke-20261026/qwen-fourier1-C-repair1-profile.json'
    oldp = read(old)
    candidates = []
    chosen = None
    for group, worker in [('reader-route-smoke-20261026', 'qwen-fourier1-C-repair1'),
                          ('reader-neighbors-extension-20261027', 'qwen-fourier1-C-neighbors2')]:
        out = root/'runs/reproducibility'/group/worker
        rows = records(out/'raw.jsonl')
        times = {x['sample_id']: x for x in records(out/'request-timings.jsonl')}
        cells = {}
        for r in rows:
            params = r['source']['source_parameters']
            key = (r['task'], params.get('background_Y'), params.get('delta_Y', params.get('angle_deg')))
            cells.setdefault(key, []).append(r)
        for key, rs in cells.items():
            ordinary = [times[r['sample_id']]['end_to_end_durable_seconds']
                        for r in rs if not r['observed_or_cold']]
            c = {'task': key[0], 'background_Y': key[1], 'point': key[2], 'n': len(rs),
                 'correct': sum(r['correct'] for r in rs), 'ordinary_n': len(ordinary),
                 'ordinary_mean_seconds': sum(ordinary)/len(ordinary),
                 'historical_raw': descriptor(out/'raw.jsonl'),
                 'historical_timing': descriptor(out/'request-timings.jsonl')}
            candidates.append(c)
            if c['correct'] < c['n'] and (chosen is None or c['ordinary_mean_seconds'] < chosen[0]['ordinary_mean_seconds']):
                chosen = (c, rs)
    cell, rows = chosen
    require((cell['task'], cell['background_Y'], cell['point'], cell['n']) == ('T2', .4, .06, 48),
            'Selected cell changed; do not silently select by new outcome')
    runtime = read(runtime_path)
    require(runtime['status'] == 'PASS' and runtime['dependency_mismatches'] == [],
            'Complete official runtime not qualified')
    plan.mkdir(parents=True)
    ids = [r['sample_id'] for r in rows]
    baseline = {r['sample_id']: r for r in rows}
    write_new(plan/'historical-cell.json', {'cell': cell, 'rows': baseline})
    profiles = []
    for cold in (1, 2):
        name = 'qwen-fourier1-official-cold' + str(cold)
        p = {k: oldp[k] for k in ('model', 'arm', 'checkpoint', 'checkpoint_assets',
             'source_catalog', 'hostname', 'gpu_uuid', 'physical_lock',
             'entry_implementation', 'resources', 'cleared_environment')}
        p.update(schema='qwen-complete-official-replay-v1', worker=name, worker_token=digest({'name':name,'ids':ids}),
                 assigned_ids=ids, maximum_model_calls=48, automatic_retry=False, formal_auto_resume=False,
                 implementation=descriptor(__file__), historical_profile=descriptor(old),
                 historical_raw=cell['historical_raw'], historical_cell=descriptor(plan/'historical-cell.json'),
                 official_runtime=descriptor(runtime_path), cold=cold,
                 output=str(root/'runs/reproducibility'/NAME/name),
                 claim_root=str(root/'runs/reproducibility'/NAME/'claims'/name),
                 pause_file=str(plan/(name+'-PAUSE.json')),
                 route_label='complete official vLLM '+VERSION+'; native unhooked QSA; official OpenCV uint8/all180/no resize')
        p['environment'] = dict(oldp['environment'])
        p['environment']['CLASSIC_PRECISION_QSA_DETERMINISM'] = '0'
        p['environment']['PATH'] = runtime['environment']+'/bin:'+os.environ['PATH']
        p['sources'] = {sid: oldp['sources'][sid] for sid in ids}
        p['python'] = runtime['environment']+'/bin/python'
        path = plan/(name+'-profile.json')
        write_new(path, p)
        profiles.append(str(path))
    value = {'schema':'finite-official-qsa-replay-v1', 'status':'SEALED',
             'sealed_at_HST':datetime.now(timezone(timedelta(hours=-10))).isoformat(),
             'authorization':'User authorized latest official QSA fixed runtime, new sibling environment and 2 fresh cold full-cell replays',
             'selected':cell, 'selection_candidates':candidates, 'sample_ids':ids,
             'maximum_model_calls':96, 'cold_loads':2, 'repeats_per_engine':1,
             'first_input_deep_validation_counts_toward_quota':True,
             'profiles':profiles, 'official_runtime':descriptor(runtime_path),
             'implementation':descriptor(__file__), 'automatic_retry':False,
             'formal_auto_resume':False, 'historical_custom_arm_rerun':False,
             'claim_boundary':'New official version only; no universal tie/order determinism or cross-version identity claim'}
    write_new(plan/'plan.json', value)
    return value


def run(profile_path):
    p = read(profile_path)
    out = Path(p['output'])
    require(sys.executable == p['python'] and socket.gethostname() == p['hostname'] and not out.exists(),
            'Wrong host/python or attempted implicit resume')
    require(file_sha256(__file__) == p['implementation']['sha256'], 'Sealed runner changed')
    require(len(p['assigned_ids']) == len(set(p['assigned_ids'])) == p['maximum_model_calls'] == 48,
            'Finite quota changed')
    require(not any((Path(p['claim_root'])/(sid+'.json')).exists() for sid in p['assigned_ids']),
            'Claim already exists; no scientific retry')
    for k, v in p['environment'].items():
        require(os.environ.get(k) == v, 'Environment changed: '+k)
    for k in p['cleared_environment']:
        require(k not in os.environ, 'Unexpected environment: '+k)
    verified(p['historical_profile'])
    require(file_sha256(p['historical_raw']['path']) == p['historical_raw']['sha256'],
            'Historical raw answers changed')
    history = verified(p['historical_cell'])['rows']
    catalog = verified(p['source_catalog'])
    for name, sha in p['entry_implementation'].items():
        path = Path(__file__).with_name(name) if '/' not in name else Path(__file__).parent.parent/name
        require(file_sha256(path) == sha, 'Unchanged qualified loader changed: '+name)
    from . import inference, gemma_speed as lifecycle
    import torch
    out.mkdir(parents=True)
    sink = original.Events(out)
    atomic_json(out/'identity.json', {'profile':p, 'profile_sha256':digest(p),
                                     'owner':owner_identity(), 'python':sys.executable})
    boundary = Boundary(p)
    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, boundary.signal)
    completed, pending = [], None
    with physical_lease(p, torch) as lease:
        atomic_json(out/'lease.json', lease)
        with ExitStack() as stack:
            stack.enter_context(original.file_loader_settings(Path(profile_path).parents[2]))
            started = time.monotonic()
            require(inference.checkpoint_asset_identity(p['checkpoint']) == p['checkpoint_assets'], 'Checkpoint changed')
            session = inference.ModelSession('qwen', p['checkpoint'], sink, resources=p['resources'])
            engine = session.engine
            owned = engine.collective_rpc(lifecycle._worker_identity)
            atomic_json(out/'engine-owner.json', {'owned':owned})
            stack.callback(lifecycle.shutdown_owned_engine, engine, owned, out)
            def close_session():
                session.engine = None
                session.close()
            stack.callback(close_session)
            native = engine.collective_rpc(native_qsa_evidence)
            parent = native_qsa_evidence()
            for value in [parent, *native]:
                validate_binding(p, value)
            atomic_json(out/'official-native-qsa.json', {'parent':parent, 'workers':native})
            atomic_json(out/'load.json', {'seconds':time.monotonic()-started,
                'native_eos':engine.get_tokenizer().eos_token_id, 'route_label':p['route_label']})
            sampling = session.sampling
            generation = original.generation_receipt(sampling)
            atomic_json(out/'generation.json', generation)
            original.warm(p, out)
            observation = stack.enter_context(DeepNativeObservation('qwen'))
            try:
                for ordinal, sid in enumerate(p['assigned_ids']):
                    if boundary.check():
                        break
                    item, ref = catalog['items'][sid], p['sources'][sid]
                    require(original.state(ref['path']) == ref['stat'], 'Source changed after sealing')
                    require(file_sha256(ref['path']) == item['ready']['video_sha256'], 'Source bytes changed')
                    pending = sid
                    write_new(Path(p['claim_root'])/(sid+'.json'), {'sample_id':sid,
                        'profile_sha256':digest(p), 'worker_token':p['worker_token'], 'attempt':1})
                    observed = ordinal == 0
                    observation.clear(); observation.active=True; observation.deep=observed
                    if observed:
                        initial = engine.collective_rpc(install_deep_probe)
                        for value in initial:
                            validate_binding(p, value)
                    _append(out/'events.jsonl', {'event':'INTENT', 'sample_id':sid, 'ordinal':ordinal,
                                                'observed_or_cold':observed})
                    request = [{'role':'user', 'content':[
                        {'type':'video_url', 'video_url':{'url':Path(ref['path']).as_uri()}},
                        {'type':'text', 'text':item['row']['prompt']}]}]
                    start = time.monotonic()
                    answer = original.submit(engine, 'qwen', 'C', request, sampling)
                    request_seconds = time.monotonic()-start
                    choice = answer.outputs[0]
                    raw = {'sample_id':sid, 'task':'T2', 'model':'qwen', 'arm':'OFFICIAL', 'cold':p['cold'],
                        'ordinal':ordinal, 'observed_or_cold':observed, 'attempt':1,
                        'truth':item['row']['truth'], 'raw_answer':choice.text,
                        'correct':choice.text.strip() == item['row']['truth'],
                        'output_token_ids':list(choice.token_ids), 'finish_reason':choice.finish_reason,
                        'stop_reason':choice.stop_reason, 'actual_prompt_token_ids':list(answer.prompt_token_ids),
                        'question':item['row']['prompt'], 'profile_sha256':digest(p),
                        'source':history[sid]['source'], 'generation':generation,
                        'native_reader':observation.reads,
                        'native_processor':observation.processors if observed else [],
                        'timing':{'engine_request_inclusive_seconds':request_seconds,
                                  'native_reader_nested_seconds':observation.read_seconds,
                                  'native_processor_nested_seconds':observation.processor_seconds}}
                    _append(out/'raw.jsonl', raw)
                    require(choice.finish_reason == 'stop' and choice.text.strip() in ('LEFT','RIGHT'),
                            'Unexpected scientific completion; raw retained')
                    require(raw['actual_prompt_token_ids'] == history[sid]['actual_prompt_token_ids']
                            and raw['question'] == history[sid]['question'], 'Actual prompt/template changed')
                    require(observation.reads and observation.processors, 'Actual reader/processor not observed')
                    deep = engine.collective_rpc(read_deep_probe) if observed else None
                    if deep:
                        gpu = [r for w in deep for r in w['actual_GPU_input']]
                        inference._validate_forwards('qwen', gpu)
                        hashes = [x['after_native_dtype']['pixel_values_videos']['sha256']
                                  for x in observation.processors if 'pixel_values_videos' in x['after_native_dtype']]
                        require(len(hashes) == 1 and all(r['sha256'] == hashes[0] for r in gpu),
                                'Actual GPU tensor differs from native processor BF16 payload')
                    _append(out/'events.jsonl', {'event':'TECHNICAL_PASS', 'sample_id':sid,
                        'raw_digest':digest(raw), 'deep':deep, 'observed_or_cold':observed})
                    elapsed = time.monotonic()-start
                    timing = {**raw['timing'], 'sample_id':sid, 'task':'T2', 'cold':p['cold'],
                              'observed_or_cold':observed, 'end_to_end_durable_seconds':elapsed,
                              'nested_not_additive':True}
                    _append(out/'request-timings.jsonl', timing)
                    completed.append(sid); pending=None
                    atomic_json(out/'progress.json', {'status':'RUNNING', 'completed':len(completed),
                        'expected':48, 'last':sid, 'last_timing':timing})
                    del answer, raw
                    if (ordinal+1) % 8 == 0:
                        gc.collect()
            except BaseException as error:
                atomic_json(out/'failure.json', {'error_type':type(error).__name__, 'error':str(error),
                    'UNCERTAIN_id':pending, 'completed':len(completed), 'automatic_retry':False})
                raise
            finally:
                observation.active=False
    result = {'status':'COMPLETE' if completed == p['assigned_ids'] else 'PAUSED',
              'completed_ids':completed, 'UNCERTAIN_ids':[pending] if pending else [],
              'unattempted_ids':[s for s in p['assigned_ids'] if s not in completed and s != pending],
              'automatic_retry':False, 'formal_auto_resume':False}
    atomic_json(out/'result.json', result)
    return result


def supervise(profile_path):
    from . import refinement_worker as worker
    base = worker.subprocess.Popen
    def popen(argv, **kwargs):
        argv = [x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv]
        return base(argv, **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile_path)
    finally:
        worker.subprocess.Popen = base


def compare(plan_path):
    plan = read(plan_path)
    profiles = [read(x) for x in plan['profiles']]
    history = verified(profiles[0]['historical_cell'])['rows']
    ids = plan['sample_ids']
    arms, closures, timing, deep = [], [], [], []
    for profile in profiles:
        out = Path(profile['output'])
        wait = read(out.with_suffix('.wait.json'))
        require(wait['returncode'] == 0 and wait['real_wait_observed']
                and wait['owned_engines_exited'] and wait['original_physical_lease_released']
                and wait['terminal_status'] == 'COMPLETE' and wait['completed_ids'] == ids,
                'Cold process did not close completely')
        raw = records(out/'raw.jsonl')
        require([r['sample_id'] for r in raw] == ids, 'Full ordered cell coverage differs')
        claims = list(Path(profile['claim_root']).glob('*.json'))
        require(len(claims) == 48 and all(read(x)['attempt'] == 1 for x in claims), 'Claims/quota changed')
        for row in raw:
            require(row['actual_prompt_token_ids'] == history[row['sample_id']]['actual_prompt_token_ids'],
                    'Historical prompt expansion changed')
        arms.append({r['sample_id']:r for r in raw})
        closures.append({'cold':profile['cold'], 'wait':wait, 'exit':read(out/'exit.json'),
                         'owned':read(out/'engine-owner.json'), 'lease':read(out/'lease.json')})
        times = records(out/'request-timings.jsonl')
        ordinary = [x['end_to_end_durable_seconds'] for x in times if not x['observed_or_cold']]
        timing.append({'cold':profile['cold'], 'load_seconds':read(out/'load.json')['seconds'],
            'first_deep_request_seconds':times[0]['end_to_end_durable_seconds'],
            'ordinary_n':len(ordinary), 'ordinary_mean_seconds':statistics.mean(ordinary),
            'ordinary_median_seconds':statistics.median(ordinary),
            'all_request_seconds':sum(x['end_to_end_durable_seconds'] for x in times)})
        events = records(out/'events.jsonl')
        proof = [x for x in events if x['event'] == 'TECHNICAL_PASS' and x['observed_or_cold']]
        require(len(proof) == 1 and proof[0]['sample_id'] == ids[0], 'Missing first actual GPU input proof')
        deep.append({'cold':profile['cold'], 'proof':proof[0]['deep'],
                     'reader':raw[0]['native_reader'], 'processor':raw[0]['native_processor']})
    fields = ('raw_answer','output_token_ids','finish_reason','stop_reason')
    paired = []
    for sid in ids:
        old, a, b = history[sid], arms[0][sid], arms[1][sid]
        paired.append({'sample_id':sid, 'truth':old['truth'], 'historical_answer':old['raw_answer'],
            'cold1_answer':a['raw_answer'], 'cold2_answer':b['raw_answer'],
            'historical_tokens':json.dumps(old['output_token_ids']),
            'cold1_tokens':json.dumps(a['output_token_ids']), 'cold2_tokens':json.dumps(b['output_token_ids']),
            'historical_finish':old['finish_reason'], 'cold1_finish':a['finish_reason'], 'cold2_finish':b['finish_reason'],
            'historical_stop':json.dumps(old['stop_reason']), 'cold1_stop':json.dumps(a['stop_reason']),
            'cold2_stop':json.dumps(b['stop_reason']),
            'cold_full_equal':all(a[k] == b[k] for k in fields),
            'cold1_historical_full_equal':all(a[k] == old[k] for k in fields),
            'cold2_historical_full_equal':all(b[k] == old[k] for k in fields)})
    summary = {'status':'COMPLETE', 'route':'COMPLETE_OFFICIAL_VERSION', 'commit':COMMIT, 'version':VERSION,
        'selected':plan['selected'], 'model_calls':sum(len(x) for x in arms), 'maximum_model_calls':96,
        'same_cold_comparison_n':48,
        'cold_answer_differences':sum(arms[0][s]['raw_answer'] != arms[1][s]['raw_answer'] for s in ids),
        'cold_output_token_differences':sum(arms[0][s]['output_token_ids'] != arms[1][s]['output_token_ids'] for s in ids),
        'cold_finish_differences':sum(arms[0][s]['finish_reason'] != arms[1][s]['finish_reason'] for s in ids),
        'cold_stop_differences':sum(arms[0][s]['stop_reason'] != arms[1][s]['stop_reason'] for s in ids),
        'cold_full_equal':sum(x['cold_full_equal'] for x in paired),
        'historical_correct':sum(x['correct'] for x in history.values()), 'colds':[],
        'timing':timing, 'closures':closures, 'actual_first_inputs':deep,
        'automatic_retry':False, 'formal_auto_resume':False,
        'boundary':'One fixed 48-ID cell, two fresh loads on one SM120. Full runtime/dependencies changed. Overflow exactness is distinct from canonical tie/order determinism; no all-history or general determinism claim.',
        'plan':descriptor(plan_path), 'implementation':descriptor(__file__)}
    for i, arm in enumerate(arms):
        summary['colds'].append({'cold':i+1, 'correct':sum(r['correct'] for r in arm.values()),
            'answer_changes_vs_historical':sum(arm[s]['raw_answer'] != history[s]['raw_answer'] for s in ids),
            'tokens_changes_vs_historical':sum(arm[s]['output_token_ids'] != history[s]['output_token_ids'] for s in ids),
            'finish_changes_vs_historical':sum(arm[s]['finish_reason'] != history[s]['finish_reason'] for s in ids),
            'stop_changes_vs_historical':sum(arm[s]['stop_reason'] != history[s]['stop_reason'] for s in ids),
            'wrong_to_correct':sum(not history[s]['correct'] and arm[s]['correct'] for s in ids),
            'correct_to_wrong':sum(history[s]['correct'] and not arm[s]['correct'] for s in ids),
            'by_truth':{side:{'n':sum(history[s]['truth'] == side for s in ids),
                'correct':sum(arm[s]['correct'] for s in ids if history[s]['truth'] == side)} for side in ('LEFT','RIGHT')}})
    require(closures[0]['owned'] != closures[1]['owned'], 'Two cold engines reused the same identity')
    hashes = lambda value: [x['sha256'] for w in value['proof'] for x in w['actual_GPU_input']]
    summary['first_actual_GPU_hashes_equal_between_colds'] = hashes(deep[0]) == hashes(deep[1])
    summary['first_decoded_video_hashes_equal_between_colds'] = deep[0]['reader'][0]['decoded_sha256'] == deep[1]['reader'][0]['decoded_sha256']
    out = Path(profiles[0]['output']).parent
    with (out/'paired.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    write_new(out/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('actual_first_inputs','closures')}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['seal','run','supervise','queue','compare'])
    parser.add_argument('--root'); parser.add_argument('--runtime'); parser.add_argument('--profile'); parser.add_argument('--plan')
    a = parser.parse_args()
    if a.command == 'seal':
        value = seal(a.root, a.runtime)
        print(json.dumps({'status':value['status'], 'selected':value['selected'], 'maximum_model_calls':96}))
    elif a.command == 'run':
        print(json.dumps(run(a.profile)))
    elif a.command == 'supervise':
        print(json.dumps(supervise(a.profile)))
    elif a.command == 'compare':
        compare(a.plan)
    else:
        p = read(a.plan)
        require(p['status'] == 'SEALED' and p['maximum_model_calls'] == 96, 'Wrong finite plan')
        for profile in p['profiles']:
            print(json.dumps(supervise(profile)), flush=True)


if __name__ == '__main__':
    main()
