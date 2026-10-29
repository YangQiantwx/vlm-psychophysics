"""Finite Cosmos X ordered-image comparison, using the sealed Astra T1 IDs.

The original X scientific runner is unchanged. This module prepares lossless
PNG frames with its official decoder and adds image-specific observation only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from . import official_x_t1_images as t1
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256

base = t1.base
NAME = 'cosmos-t1-images-20261029'
CONTEXT = 98304
KV_BYTES = 25 * 2**30


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_deep(p, item, deep):
    """Dense Cosmos has no Qwen-Next QSA selector; check its actual image path."""
    expected = t1.budget_for(p, item)
    hashes = [x['sha256'] for x in expected['pixel_payloads']]
    require(len(hashes) == 180, 'Missing first-input CPU patch sequence')
    actual, norms = [], []
    for worker in deep:
        require(not worker.get('runtime_error') and not worker.get('profiler_error'), 'GPU observer failed')
        runtime = worker['runtime']
        require(runtime['actual_kv_capacity'] >= CONTEXT, 'Image context exceeds actual KV capacity')
        require(any('FlashInferCutlassNvFp4LinearKernel' in key and n > 0
                    for key, n in runtime['method_kernel_counts'].items()), 'Expected Cosmos NVFP4 kernel absent')
        norm = worker['normalizer']
        require(norm['class'].endswith(('.FusedMMInputNorm', '.IdentityInputNorm')) and norm.get('source'),
                'Missing official image normalizer')
        source = t1.v2.official_file(norm['source']['path'])
        require(source['sha256'] == norm['source']['sha256'], 'Official normalizer identity differs')
        norms.append({**norm, 'official_source_verification': source})
        for row in worker['outer']:
            tensor = row['tensor']
            require(tensor['dtype'] == 'torch.uint8' and tensor['device'].startswith('cuda'),
                    'Image GPU ingress dtype/device changed')
            actual.extend(row.get('per_image_patch_sha256', []))
    require(actual and set(actual) == set(hashes), 'GPU encoder did not cover all logical image pixel payloads')
    raw = base.records(Path(p['output']) / 'raw.jsonl')[-1]
    base._append(Path(p['output']) / 'input-admissions.jsonl', {
        'sample_id': raw['sample_id'], 'raw_digest': digest(raw), 'observations_digest': digest(deep),
        'proof': {'status': 'PASS_NATIVE_ORDERED_IMAGE_LIST', 'logical_image_count': 180,
                  'logical_image_grid_thw': [[1, 32, 48]] * 180,
                  'logical_patch_sha256_in_order': hashes,
                  'actual_GPU_encoded_image_count': len(actual),
                  'actual_GPU_encoded_patch_sha256': actual,
                  'unique_logical_image_payload_count': len(set(hashes)),
                  'logical_prompt_token_ids_match_CPU': True,
                  'actual_GPU_bytes_cover_all_logical_images': True,
                  'native_duplicate_image_reuse_allowed': True, 'official_normalizers': norms,
                  'boundary': 'Logical chronological 180-image request, CPU grids and prompt are checked separately from physical encoder reuse of identical images.'}})


base.validate_deep = validate_deep


def native_cpu_budget(parent, item):
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.media.image import ImageMediaIO

    tokenizer = parent['engine_kwargs']['tokenizer']
    hf = AutoProcessor.from_pretrained(tokenizer, trust_remote_code=False, local_files_only=True)
    config = ModelConfig(model=parent['checkpoint'], tokenizer=tokenizer, dtype='bfloat16',
                        trust_remote_code=False, max_model_len=CONTEXT,
                        limit_mm_per_prompt={'image': 180, 'video': 0, 'audio': 0},
                        mm_processor_cache_gb=0,
                        mm_processor_kwargs={'do_resize': False, 'do_sample_frames': False})
    native = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=hf.tokenizer)
    images = [ImageMediaIO().load_file(Path(x['path'])).media for x in item['content'] if x['type'] == 'image']
    require([sha_bytes(np.asarray(x.convert('RGB')).tobytes()) for x in images] == item['frame_pixel_sha256'],
            'Official image reader changed decoded video pixels')
    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': item['row']['prompt']}]
    text = hf.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
    output = native.apply(ProcessorInputs(prompt=hf.tokenizer.encode(text, add_special_tokens=False),
                          mm_data_items=native.info.parse_mm_data({'image': images}),
                          hf_processor_mm_kwargs={'do_resize': False, 'do_sample_frames': False}),
                          TimingContext(enabled=False))
    patches, grids = [], []
    def walk(value, key=''):
        if isinstance(value, torch.Tensor):
            if 'pixel' in key:
                patches.append({'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
                                'sha256': t1.byte_sha(value)})
            if 'grid_thw' in key:
                grids.append({'key': key, 'values': value.tolist()})
        elif isinstance(value, dict) or hasattr(value, 'items'):
            for k, v in value.items(): walk(v, k)
        elif isinstance(value, (list, tuple)):
            for v in value: walk(v, key)
        elif hasattr(value, 'data'): walk(value.data, key)
    walk(output['mm_kwargs'])
    ids = output['prompt_token_ids']
    require(len(patches) == 180 and all(x['shape'] == [1536, 1536] and x['dtype'] == 'torch.uint8' for x in patches),
            'Native CPU image patch count/shape/dtype differs')
    require(len(grids) == 180 and all(x['values'] == [1, 32, 48] for x in grids), 'Image processor resized or lost an image')
    require(len(ids) + 16384 <= CONTEXT and not torch.cuda.is_initialized(), 'CPU context qualification failed')
    return {'arm': 'images', 'sample_id': item['sample_id'], 'prompt_tokens': len(ids),
            'prompt_token_ids_sha256': sha_bytes(json.dumps(ids, separators=(',', ':')).encode()),
            'max_tokens': 16384, 'context': CONTEXT, 'fits_context': True,
            'pixel_payloads': patches, 'grids': grids, 'processor_tested_this_id': True,
            'template_text_sha256': sha_bytes(text.encode()), 'source_all180_pixel_hashes_match': True}


def prepare(root, card):
    import numpy as np
    import torch
    from PIL import Image
    from vllm.multimodal.media.video import VideoMediaIO
    from vllm.multimodal.media.image import ImageMediaIO

    torch.set_num_threads(6)
    root = Path(root); plans = root / 'plans' / NAME; run = root / 'runs/reproducibility' / NAME
    plan = read(plans / 'plan.json'); selection = verified(plan['selection'])
    require(plan['maximum_new_model_calls'] == 320 and plan['automatic_retry'] is False, 'Unexpected study quota')
    parent_path = root / 'plans/cosmos-x-full-20261028' / (card + '-profile.json')
    parent = read(parent_path); rows = [x for x in selection['samples'] if x['card'] == card]
    require(len(rows) == plan['allocation'][card] and all(x['sample_id'] in parent['assigned_ids'] for x in rows),
            'New image assignment must reuse each source previous Cosmos video GPU')
    require(not (plans / (card + '-profile.json')).exists(), 'Never overwrite or implicitly resume a prepared shard')
    run.mkdir(parents=True, exist_ok=True)
    loader = VideoMediaIO(ImageMediaIO(), video_backend='opencv', num_frames=180, fps=-1)
    items = {}; started = time.monotonic()
    for row in rows:
        sid = row['sample_id']; uid = row['case_id']; folder = run / 'media' / uid
        folder.mkdir(parents=True, exist_ok=False)
        source = root / row['source_relative']; require(file_sha256(source) == row['source_sha256'], 'Sealed source differs')
        frames, meta = loader.load_file(source).media
        require(frames.dtype == np.uint8 and frames.shape == (180,512,768,3)
                and meta['frames_indices'] == list(range(180)), 'Official decoder did not preserve all frames and geometry')
        content, hashes = [], []
        for i, frame in enumerate(frames):
            pixel_sha = sha_bytes(frame.tobytes()); hashes.append(pixel_sha)
            path = folder / f'{uid}_{i:03d}.png'; Image.fromarray(frame).save(path)
            with Image.open(path) as loaded:
                require(loaded.mode == 'RGB' and loaded.size == (768,512) and not loaded.info
                        and sha_bytes(np.asarray(loaded).tobytes()) == pixel_sha, 'PNG export altered pixels')
            content.append({'type':'image', 'path':str(path), 'sha256':file_sha256(path),
                            'frame_index':i, 'pixel_sha256':pixel_sha})
        content.append({'type':'text', 'text':row['prompt']})
        items[sid] = {'sample_id':sid, 'task':'T1', 'input_modality':'native_image_list', 'content':content,
            'row':{'sample_id':sid, 'task':'T1', 'truth':row['truth'], 'prompt':row['prompt'],
                   'choices':['LEFT','RIGHT'], 'source_parameters':row['parameters']},
            'source_parameters':row['parameters'], 'accuracy_eligible':True,
            'video_contract':{'frames':180,'fps':60,'width':768,'height':512},
            'image_contract':{'count':180,'width':768,'height':512},
            'decoded_video_sha256':sha_bytes(frames.tobytes()), 'frame_pixel_sha256':hashes,
            'ready':{'video_sha256':row['source_sha256'],'source_qa':'PASS','frames':180,'width':768,'height':512,'fps':60},
            'event_frames':list(range(180)), 'original_source':{'path':str(source),'sha256':row['source_sha256']}}
        del frames
        if len(items) == 1 or len(items) % 20 == 0:
            print(json.dumps({'phase':'CPU_PNG_QA', 'card':card,'done':len(items),'total':len(rows)}), flush=True)
    catalog_path = plans / (card + '-sources.json')
    write_new(catalog_path, {'schema':'official-x-catalog-v1','protocol_id':NAME,'status':'SOURCE_QA_PASS_MODEL_INPUT_PENDING',
                            'items':items,'assigned_ids':list(items),'model_calls':0})
    first = next(iter(items.values())); tested = native_cpu_budget(parent, first)
    attempts = []
    for sid, item in items.items():
        require(item['row']['prompt'] == first['row']['prompt'] and item['image_contract'] == first['image_contract'],
                'Cannot reuse expected prompt under changed template/geometry')
        if sid == first['sample_id']: attempts.append(tested)
        else: attempts.append({**{k:v for k,v in tested.items() if k not in ('pixel_payloads','grids')},
                               'sample_id':sid,'processor_tested_this_id':False,
                               'expected_prompt_basis':'Identical text, 180 image placeholders, and fixed image geometry; every actual request checks expanded prompt and 180 grids.'})
    budget_path = plans / (card + '-cpu-budget.json')
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    write_new(budget_path, {'status':'PASS','model_calls':0,'weights_loaded':False,'CUDA_initialized':False,
                            'coverage':'All source videos decoded once; every PNG byte checked. Native processor only first source per card; every actual request will check prompt/grid.',
                            'attempts':attempts,'elapsed_seconds':time.monotonic()-started})
    p = copy.deepcopy(parent)
    for key in ('media_files','sources','worker_token','implementation','observer_entry','source_catalog'):
        p.pop(key, None)
    p.update(worker='cosmos-'+card+'-X-images', assigned_ids=list(items), maximum_model_calls=len(items),
             protocol_name=NAME, route_label='X ordered 180-image request', scope_change_only=False,
             output=str(run/('cosmos-'+card+'-X-images')), claim_root=str(run/'claims'/card),
             pause_file=str(plans/(card+'-PAUSE.json')), source_catalog=descriptor(catalog_path),
             t1_cpu_budget=descriptor(budget_path), input_arm='images', minimum_kv_capacity=CONTEXT,
             parent_validated_profile=descriptor(parent_path), image_entry=descriptor(__file__),
             t1_entry=descriptor(t1.__file__), selection=plan['selection'], phase='images')
    p['engine_kwargs'].update(max_model_len=CONTEXT, kv_cache_memory_bytes=KV_BYTES,
                             limit_mm_per_prompt={'image':180,'video':0,'audio':0})
    draft = plans/(card+'-draft.json'); write_new(draft,p)
    result = t1.v2.seal(draft, plans/(card+'-profile.json'))
    print(json.dumps({**result,'CPU_prompt_tokens':tested['prompt_tokens'],'CPU_no_model_calls':True}),flush=True)
    return result


def supervise(profile):
    from . import refinement_worker as worker
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try: return worker.supervise(profile)
    finally: worker.subprocess.Popen = native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','supervise'])
    parser.add_argument('--root');parser.add_argument('--card');parser.add_argument('--profile');a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.root,a.card)
    elif a.command=='supervise': result=supervise(a.profile)
    else:
        p=read(a.profile)
        require(p['model']=='cosmos' and p['input_arm']=='images', 'Wrong model/modality')
        for key,path in [('image_entry',__file__),('t1_entry',t1.__file__)]:
            require(file_sha256(path)==p[key]['sha256'], 'Frozen image observer changed')
        require(verified(p['t1_cpu_budget'])['status']=='PASS', 'CPU admission not complete')
        result=t1.v2.run(a.profile)
    print(json.dumps(result))


if __name__=='__main__': main()
