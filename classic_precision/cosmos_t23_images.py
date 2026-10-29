"""Prepare two complete Cosmos X cells for native ordered-image comparison.

No new inference implementation: execute the unchanged Cosmos image entry.
Every source remains on the GPU that produced its baseline video answer.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
from . import cosmos_t1_images as common
from .cosmos_t1_images import native_cpu_budget, sha_bytes, CONTEXT, KV_BYTES, t1
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import file_sha256
NAME='cosmos-t23-images-20261029'

def prepare(root, card):
    import numpy as np
    import torch
    from PIL import Image
    from vllm.multimodal.media.video import VideoMediaIO
    from vllm.multimodal.media.image import ImageMediaIO

    torch.set_num_threads(6)
    root = Path(root); plans = root / 'plans' / NAME; run = root / 'runs/reproducibility' / NAME
    plan = read(plans / 'plan.json'); selection = verified(plan['selection'])
    require(plan['maximum_new_model_calls'] == 144 and plan['automatic_retry'] is False, 'Unexpected study quota')
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
        items[sid] = {'sample_id':sid, 'task':row['task'], 'input_modality':'native_image_list', 'content':content,
            'row':{'sample_id':sid, 'task':row['task'], 'truth':row['truth'], 'prompt':row['prompt'],
                   'choices':['LEFT','RIGHT'], 'source_parameters':row['parameters']},
            'source_parameters':row['parameters'], 'accuracy_eligible':True,
            'video_contract':{'frames':180,'fps':60,'width':768,'height':512},
            'image_contract':{'count':180,'width':768,'height':512},
            'decoded_video_sha256':sha_bytes(frames.tobytes()), 'frame_pixel_sha256':hashes,
            'ready':{'video_sha256':row['source_sha256'],'source_qa':'PASS','frames':180,'width':768,'height':512,'fps':60},
            'event_frames':row['event_frames'], 'original_source':{'path':str(source),'sha256':row['source_sha256']}}
        del frames
        if len(items) == 1 or len(items) % 20 == 0:
            print(json.dumps({'phase':'CPU_PNG_QA', 'card':card,'done':len(items),'total':len(rows)}), flush=True)
    catalog_path = plans / (card + '-sources.json')
    write_new(catalog_path, {'schema':'official-x-catalog-v1','protocol_id':NAME,'status':'SOURCE_QA_PASS_MODEL_INPUT_PENDING',
                            'items':items,'assigned_ids':list(items),'model_calls':0})
    first_by_task = {}; tested_by_task = {}; attempts = []
    for sid, item in items.items():
        task=item['task']
        if task not in tested_by_task:
            first_by_task[task]=item;tested_by_task[task]=native_cpu_budget(parent,item)
            attempts.append(tested_by_task[task])
        else:
            first=first_by_task[task];tested=tested_by_task[task]
            require(item['row']['prompt']==first['row']['prompt'] and item['image_contract']==first['image_contract'], 'Cannot reuse expected task prompt under changed template/geometry')
            attempts.append({**{k:v for k,v in tested.items() if k not in ('pixel_payloads','grids')},'sample_id':sid,'processor_tested_this_id':False,'expected_prompt_basis':'Same task prompt and 180 original-size image placeholders; actual prompt and every grid checked for every request.'})
    budget_path = plans / (card + '-cpu-budget.json')
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    write_new(budget_path, {'status':'PASS','model_calls':0,'weights_loaded':False,'CUDA_initialized':False,
                            'coverage':'All source videos decoded once; every PNG byte checked. Native processor first source per task and card; every actual request will check prompt/grid.',
                            'attempts':attempts,'elapsed_seconds':time.monotonic()-started})
    p = copy.deepcopy(parent)
    for key in ('media_files','sources','worker_token','implementation','observer_entry','source_catalog'):
        p.pop(key, None)
    p.update(worker='cosmos-'+card+'-X-images-T23', assigned_ids=list(items), maximum_model_calls=len(items),
             protocol_name=NAME, route_label='X ordered 180-image request', scope_change_only=False,
             output=str(run/('cosmos-'+card+'-X-images-T23')), claim_root=str(run/'claims'/card),
             pause_file=str(plans/(card+'-PAUSE.json')), source_catalog=descriptor(catalog_path),
             t1_cpu_budget=descriptor(budget_path), input_arm='images', minimum_kv_capacity=CONTEXT,
             parent_validated_profile=descriptor(parent_path), image_entry=descriptor(common.__file__), preparation_entry=descriptor(__file__),
             t1_entry=descriptor(t1.__file__), selection=plan['selection'], phase='images')
    p['engine_kwargs'].update(max_model_len=CONTEXT, kv_cache_memory_bytes=KV_BYTES,
                             limit_mm_per_prompt={'image':180,'video':0,'audio':0})
    draft = plans/(card+'-draft.json'); write_new(draft,p)
    result = t1.v2.seal(draft, plans/(card+'-profile.json'))
    print(json.dumps({**result,'CPU_prompt_tokens':{k:v['prompt_tokens'] for k,v in tested_by_task.items()},'CPU_no_model_calls':True}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--card',choices=['fourier0','fourier1'],required=True);a=p.parse_args();prepare(a.root,a.card)
