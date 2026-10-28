"""Finite native-video versus ordered-image-list observation, without math patches.

The frozen X runners are reused. This entry adds source-pixel and logical-image
checks; native duplicate-image encoder reuse remains enabled and is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from . import official_x_runner_v2 as v2
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256

base = v2.base
_native_observer = base.NativeObserver
_native_install = base.install_probe
_video_processor = base.validate_processor
_video_deep = base.validate_deep


def byte_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def describe_payload(value, deep=False):
    import torch
    if isinstance(value, torch.Tensor):
        result = base.tensor_description(value, deep)
        if deep and value.ndim == 2 and value.shape[1] == 1536 and value.shape[0] % 1536 == 0:
            result['per_image_patch_sha256'] = [byte_sha(x) for x in value.split(1536, dim=0)]
        return result
    if isinstance(value, (list, tuple)):
        return [describe_payload(x, deep) for x in value]
    return None


class T1Observer(_native_observer):
    def __init__(self, p):
        super().__init__(p)
        catalog = verified(p['source_catalog'])
        self.media_items = {}
        for sid in p['assigned_ids']:
            item = catalog['items'][sid]
            for media in item['content']:
                if media['type'] in ('image', 'video'):
                    self.media_items[str(Path(media['path']))] = (item, media)

    def describe(self, data):
        return {k:describe_payload(v, self.deep) for k,v in data.items()
                if 'pixel' in k or 'grid' in k or k in ('video_num_patches', 'video_frame_counts')}

    def __enter__(self):
        super().__enter__()
        import numpy as np
        from vllm.multimodal.media.image import ImageMediaIO
        from vllm.multimodal.media.video import VideoMediaIO
        self.original_image_read = ImageMediaIO.load_file
        obs = self

        def image_read(loader, path):
            started = time.monotonic()
            result = obs.original_image_read(loader, path)
            if obs.active:
                key = str(Path(path)); item, media = obs.media_items[key]
                pixels = np.asarray(result.media.convert('RGB'))
                checksum = hashlib.sha256(pixels.tobytes()).hexdigest()
                require(pixels.dtype == np.uint8 and list(pixels.shape) == [512,768,3],
                        'Native PNG pixels changed shape or dtype')
                require(checksum == media['pixel_sha256'] == item['frame_pixel_sha256'][media['frame_index']],
                        'Native PNG pixels differ from the corresponding original video frame')
                obs.reads.append({'kind':'image', 'path':key, 'frame_index':media['frame_index'],
                    'shape':list(pixels.shape), 'dtype':str(pixels.dtype), 'pixel_sha256':checksum})
                obs.read_seconds += time.monotonic()-started
            return result

        def video_read(loader, path):
            started = time.monotonic()
            result = obs.original_read(loader, path)
            if obs.active:
                item, media = obs.media_items[str(Path(path))]
                frames, meta = result.media
                per_frame = [hashlib.sha256(x.tobytes()).hexdigest() for x in frames]
                checksum = hashlib.sha256(frames.tobytes()).hexdigest()
                require(str(frames.dtype) == 'uint8' and list(frames.shape) == [180,512,768,3]
                    and meta['frames_indices'] == list(range(180)), 'Native video frame contract changed')
                require(checksum == item['decoded_video_sha256'] and per_frame == item['frame_pixel_sha256'],
                        'Native video decode differs from the frozen PNG source pixels')
                obs.reads.append({'kind':'video','path':str(Path(path)), 'shape':list(frames.shape),
                    'dtype':str(frames.dtype),'metadata':meta,'decoded_sha256':checksum,
                    'per_frame_sha256':per_frame})
                obs.read_seconds += time.monotonic()-started
            return result

        ImageMediaIO.load_file = image_read
        VideoMediaIO.load_file = video_read
        return self

    def __exit__(self, *args):
        from vllm.multimodal.media.image import ImageMediaIO
        ImageMediaIO.load_file = self.original_image_read
        return super().__exit__(*args)


def install_probe(worker, spec, model):
    result = _native_install(worker, spec, model)
    import torch
    _, outer = base.resolve_module(worker.get_model(), spec['outer_paths'])
    state = worker._official_x_probe

    def image_chunks(module, args, kwargs):
        tensor = base.first_tensor(args, kwargs)
        grid = kwargs.get('grid_thw')
        if not isinstance(grid, torch.Tensor):
            return
        values = grid.tolist()
        if values and isinstance(values[0], int):
            values = [values]
        if not values or any(x != [1,32,48] for x in values):
            return
        require(tensor is not None and list(tensor.shape) == [len(values)*1536,1536],
                'Actual native image GPU patches differ from their declared grids')
        state['outer'][-1]['per_image_patch_sha256'] = [byte_sha(x) for x in tensor.split(1536, dim=0)]

    state['handles'].append(outer.register_forward_pre_hook(image_chunks, with_kwargs=True))
    return result


def budget_for(p, item):
    record = verified(p['t1_cpu_budget'])
    arm = p['input_arm']
    return next(x for x in record['attempts'] if x['arm'] == arm and x['sample_id'] == item['sample_id'])


def validate_processor(p, item, processors):
    raw = base.records(Path(p['output'])/'raw.jsonl')[-1]
    expected = budget_for(p, item)
    token_ids = raw['actual_prompt_token_ids']
    token_sha = hashlib.sha256(json.dumps(token_ids, separators=(',',':')).encode()).hexdigest()
    require(token_sha == expected['prompt_token_ids_sha256'], 'Actual logical native prompt differs from CPU expansion')
    if p['input_arm'] == 'video':
        return _video_processor(p, item, processors)
    request = raw['request'][0]['content']
    require(len(request) == 181 and [x['type'] for x in request] == ['image_url']*180+['text'],
            'The image arm must be one ordered 180-image request and the original text')
    paths = [str(Path(x['path'])) for x in item['content'] if x['type'] == 'image']
    require([x['image_url']['url'] for x in request[:-1]] == [Path(x).as_uri() for x in paths],
            'Logical image URL order differs')
    reads = raw['native_reader']
    require(len(reads) == 180 and {x['path'] for x in reads} == set(paths), 'Native reader did not load all 180 PNGs')
    require(sorted(x['frame_index'] for x in reads) == list(range(180)), 'Logical frame identities differ')
    grids = [x[phase]['image_grid_thw'].get('values') for x in processors
             for phase in ('before_native_dtype','after_native_dtype') if 'image_grid_thw' in x[phase]]
    require([[1,32,48]]*180 in grids, 'Native CPU processor did not retain all 180 image grids')
    if raw['deep']:
        expected_patches = [x['sha256'] for x in expected['pixel_payloads']]
        observed_patches = [x['after_native_dtype']['pixel_values'].get('per_image_patch_sha256')
                            for x in processors if 'pixel_values' in x['after_native_dtype']]
        require(expected_patches in observed_patches, 'Native CPU image patch byte sequence differs from source order')


def validate_deep(p, item, deep):
    if p['input_arm'] == 'video':
        return _video_deep(p, item, deep)
    expected = budget_for(p, item)
    expected_hashes = [x['sha256'] for x in expected['pixel_payloads']]
    require(len(expected_hashes) == 180, 'CPU expected image patch sequence incomplete')
    actual = []
    for worker in deep:
        require(not worker.get('runtime_error') and not worker.get('profiler_error'), 'GPU observer failed')
        runtime = worker['runtime']; require(runtime['actual_kv_capacity'] >= p['minimum_kv_capacity'], 'Actual KV too small')
        if p['model'] == 'qwen_x_qsa':
            qsa = runtime['custom_qsa_selector']
            require(qsa['status'] == 'APPLIED_X_QSA' and qsa['actual_function_is_canonical_wrapper']
                and qsa['native_function_verified_before_replacement']
                and qsa['selector_source_sha256'] == p['selector_source']['sha256']
                and worker['actual_custom_selector_calls_during_request'] > 0,
                    'Actual canonical custom QSA not observed')
        else:
            qsa = runtime['native_qsa']
            require(qsa['function_source_exact'] and not qsa['custom_selector_modules']
                    and not qsa['custom_selector_import_hooks'] and qsa['binary_mapped'], 'Official QSA identity differs')
            require(any('persistent_topk' in x['key'] and x['count'] > 0 for x in worker['actual_topk_operator_calls']),
                    'Actual native TopK not observed')
        norm = worker['normalizer']
        require(norm['class'].endswith(('.FusedMMInputNorm','.IdentityInputNorm')) and norm.get('source'),
                'No identified official image normalization module')
        source = v2.official_file(norm['source']['path'])
        require(source['sha256'] == norm['source']['sha256'], 'Official normalizer changed')
        for row in worker['outer']:
            t = row['tensor']
            require(t['dtype'] == 'torch.uint8' and t['device'].startswith('cuda'), 'Actual image GPU ingress differs')
            actual.extend(row.get('per_image_patch_sha256', []))
    require(actual and set(actual) == set(expected_hashes), 'Actual GPU encoded image bytes do not cover all logical images')
    raw = base.records(Path(p['output'])/'raw.jsonl')[-1]
    proof = {'status':'PASS_NATIVE_ORDERED_IMAGE_LIST', 'logical_image_count':180,
        'logical_image_grid_thw':[[1,32,48]]*180, 'logical_patch_sha256_in_order':expected_hashes,
        'actual_GPU_encoded_image_count':len(actual), 'actual_GPU_encoded_patch_sha256':actual,
        'unique_logical_image_payload_count':len(set(expected_hashes)), 'native_duplicate_image_reuse_allowed':True,
        'logical_prompt_token_ids_match_CPU':True, 'actual_GPU_bytes_cover_all_logical_images':True,
        'internal_normalized_tensor_observed':any(x.get('normalized') for x in deep),
        'boundary':'Native caching may reuse repeated pixels. Logical 180-image placement is verified against the exact CPU-expanded prompt; physical encoder executions are reported separately.'}
    base._append(Path(p['output'])/'input-admissions.jsonl', {'sample_id':raw['sample_id'],
        'raw_digest':digest(raw),'observations_digest':digest(deep),'proof':proof})


base.NativeObserver = T1Observer
base.install_probe = install_probe
base.validate_processor = validate_processor
base.validate_deep = validate_deep


def seal(draft, output):
    p = read(draft); p['t1_entry'] = descriptor(__file__)
    budget = verified(p['t1_cpu_budget'])
    require(budget['status'] == 'PASS' and not budget['CUDA_initialized'] and budget['model_calls'] == 0,
            'The complete native CPU budget must pass before launch')
    require(p['input_arm'] in ('video','images') and len(p['assigned_ids']) == 8, 'Wrong finite T1 arm')
    for sid in p['assigned_ids']:
        require(any(x['arm'] == p['input_arm'] and x['sample_id'] == sid and x['fits_context'] for x in budget['attempts']),
                'Missing expanded CPU budget for a source')
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(p,f); path = f.name
    try:
        return v2.seal(path,output)
    finally:
        Path(path).unlink()


def supervise(profile):
    from . import refinement_worker as worker
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['seal','run','supervise','queue'])
    parser.add_argument('--profile'); parser.add_argument('--draft'); parser.add_argument('--output'); parser.add_argument('--plan')
    a = parser.parse_args()
    if a.command == 'seal':
        result = seal(a.draft,a.output)
    elif a.command == 'run':
        p = read(a.profile); require(file_sha256(__file__) == p['t1_entry']['sha256'], 'T1 observer source changed')
        result = v2.run(a.profile)
    elif a.command == 'supervise':
        result = supervise(a.profile)
    else:
        plan = read(a.plan); require(plan['status'] == 'SEALED' and plan['automatic_retry'] is False, 'Wrong finite T1 queue')
        for profile in plan['profiles']:
            result = supervise(profile); print(json.dumps(result),flush=True)
            require(result['returncode'] == 0 and result['terminal_status'] == 'COMPLETE'
                and result['real_wait_observed'] and result['owned_engines_exited']
                and result['original_physical_lease_released'], 'Prior T1 worker did not close COMPLETE')
        result = {'status':'COMPLETE','profiles':plan['profiles']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
