"""Scoped T4 policy dispatch over the frozen native static image executor.

No processor/model/normalization forward is replaced. The adapter only extends
scientific item validation, reuses proved endpoint provenance, binds the new
queue start gate, and records unscored zero-coherence metadata before append.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path

from . import static_x_admission as native
from . import static_t4_strict_sources as sources
from . import official_x_runner as base
from .storage import digest


def check_item(item, protocol):
    sources.require(item['task'] == item['row']['task'] == 'T4' and item['domain'] == 'classic', 'ST4 task/domain differs')
    sources.require(protocol == sources.protocol(protocol['source_video_prompt']), 'ST4 protocol differs from literal minimum replacements')
    row = item['row']; entries = native.image_entries(item)
    sources.require([x['type'] for x in item['content']] == ['image', 'image', 'text']
                    and [x['frame_index'] for x in entries] == [0, 179]
                    and item['content'][-1]['text'] == row['prompt'] == protocol['prompt']
                    and row['choices'] == ['LEFT', 'RIGHT']
                    and item['video_contract'] == sources.CONTRACT
                    and item['image_contract'] == {'count': 2, 'width': 768, 'height': 512}, 'ST4 ordered input/prompt/geometry differs')
    params = row['source_parameters']; eligible = params['coherence'] > 0
    sources.require(params['N'] in (8,16) and params['coherence'] in [i/8 for i in range(9)]
                    and params['signal_count'] == params['N'] * params['coherence']
                    and params['diameter_px'] == 56 and params['speed_px_s'] == 32
                    and 0 <= params['layout_index'] < 24, 'ST4 inherited scientific conditions differ')
    sources.require(row['accuracy_eligible'] is eligible and
                    ((row['truth'] == params['direction'] and row['truth'] in ('LEFT','RIGHT')) if eligible else
                     (row['truth'] is None and params['direction'] == 'NONE')), 'ST4 GT/zero scoreability differs')
    hashes = [x['pixel_sha256'] for x in entries]
    sources.require(hashes == item['source_frame_RGB8_sha256']
                    and item['model_input_sha256'] == sources.digest([hashes,row['prompt'],row['choices']]), 'ST4 actual ordered RGB8 identity differs')
    export = item['endpoint_export']
    sources.require(export['status'] == 'PASS_EXACT_ENDPOINT_RGB8' and export['resize_crop_calls'] == 0
                    and export['source_sha256'] == item['original_source']['sha256']
                    and export['source_qa'] == item['source_qa']
                    and [x['frame_index'] for x in export['endpoints']] == [0,179]
                    and [x['pixel_sha256'] for x in export['endpoints']] == hashes
                    and all(x['matches_accepted_dual_decoder_QA'] for x in export['endpoints']), 'ST4 endpoint export evidence missing')
    return 2


@contextmanager
def dispatch(p=None, *, execution=False):
    """All temporary policy bindings are restored even on exceptions."""
    original_check, original_load = native.check_item, native.load_images
    original_gate, original_append = native.start_gate, base._append
    def checked(item, protocol):
        if item.get('task') == 'T4':
            return check_item(item, protocol)
        return original_check(item, protocol)
    def loaded(item, verify_video=False):
        if item.get('task') != 'T4':
            return original_load(item, verify_video=verify_video)
        # Prove against frozen dual-decoder QA and the new exact-PTS export.
        # Do not decode all180 again to satisfy an old qualification helper.
        check_item(item, sources.protocol(item['original_video_prompt']) if 'original_video_prompt' in item
                   else native.cached_verified(p['static_protocol']))
        qa = native.cached_verified(item['source_qa'])
        proofs=item['endpoint_export']['endpoints']
        sources.require(qa['sample_id']==item['original_video_aliases'][0]['sample_id']
                        and qa['video_sha256']==item['original_source']['sha256']
                        and [qa['rgb16_frame_sha256'][i] for i in (0,179)] ==
                            [proof['source_rgb16_sha256'] for proof in proofs],
                        'ST4 strict endpoint RGB16 source proof changed')
        images, receipts = original_load(item, verify_video=False)
        for receipt in receipts:
            receipt.update(matches_original_video_RGB8=True,
                source_validation_basis='Source container and RGB16 endpoint SHA matched, OpenCV RGB8 matched independent decode',
                accepted_source_qa=item['source_qa'])
        return images, receipts
    def appended(path, value):
        if p is not None and str(path) == str(Path(p['output']) / 'raw.jsonl'):
            catalog = native.cached_verified(p['source_catalog'])
            item = catalog['items'][value['sample_id']]
            sources.require(value['task'] == 'T4' and value['source']['item_digest'] == digest(item), 'ST4 raw scientific identity changed')
            eligible = item['row']['accuracy_eligible']
            value.update(variant='CURRENT_X_STATIC_T4_STRICT_FIRST_LAST', accuracy_eligible=eligible,
                         correct=(value['raw_answer'].strip() == value['truth']) if eligible else None)
        return original_append(path, value)
    native.check_item, native.load_images = checked, loaded
    if execution:
        from . import static_t4_strict_queue
        native.start_gate = static_t4_strict_queue.start_gate
        base._append = appended
    try:
        yield
    finally:
        native.check_item, native.load_images = original_check, original_load
        native.start_gate, base._append = original_gate, original_append


def cpu_sample(p, item):
    with dispatch(p):
        return native.cpu_sample(p, item)


def execute(profile):
    from . import static_t4_strict_operations as ops
    p = sources.read(profile)
    with dispatch(p, execution=True):
        return native.execute_bound_profile(profile, policy_validator=ops.full_policy,
            input_validator=ops.full_input, deep_policy=ops.deep_policy, runtime_validator=ops.qualified_deployment)
