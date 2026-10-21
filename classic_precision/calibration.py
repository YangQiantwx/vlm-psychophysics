"""CPU measurement of the actual official processor, before and after BF16.

This is calibration, not label correction. No model weights or VLM calls.
The post-BF16 image estimators do not imply anything about learned features.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

from . import raster
from .manifest import atomic_json, digest, file_sha, state_key


def image_measurement(row, source_codes, restored_srgb):
    p = row['source_parameters']
    baseline, event = 0, p['onset_frame']
    indices = [baseline, event]
    observed_rgb = restored_srgb[indices, :512, :768].double().numpy()
    if not np.isfinite(observed_rgb).all():
        raise ValueError('Nonfinite inverse processor pixels')
    actual_y = raster.linear(observed_rgb) @ np.array([.2126, .7152, .0722])
    from .verify import linear_y
    source_y = linear_y(source_codes[indices, :512, :768])
    result = {'max_sRGB_difference': float(np.max(abs(observed_rgb-source_codes[indices, :512, :768]/65535.))),
              'observed_sRGB_min': float(observed_rgb.min()), 'observed_sRGB_max': float(observed_rgb.max()),
              'task': row['task']}
    if row['task'] == 'T2':
        a, _ = raster.disk_coverage(area=p['disk_alpha_area_px2'])
        if p['target_side'] == 'RIGHT':
            a = np.roll(a, 320, axis=1)
        core = a >= 1-1e-12
        actual = float((actual_y[1]-actual_y[0])[core].mean())
        source = float((source_y[1]-source_y[0])[core].mean())
        result.update(nominal_delta_Y=p['delta_Y'], source_delta_Y=source,
                      BF16_delta_Y=actual, absolute_nominal_error=abs(actual-p['delta_Y']),
                      absolute_source_error=abs(actual-source))
    else:
        axes = []
        for index, side in enumerate(('LEFT', 'RIGHT')):
            cx, cy = p['centers_xy'][index]
            patch = actual_y[1, cy-128:cy+128, cx-128:cx+128]
            # Calibrate opaque foreground/background from observed tensor values.
            bg, fg = float(np.median(patch[:10])), float(patch.max())
            if not fg > bg:
                raise ValueError('BF16 image has no foreground/background separation')
            a = np.clip((patch-bg)/(fg-bg), 0, 1)
            pca, edge = raster.pca_axis(a), raster.edge_axis(a)
            if edge['status'] != 'ok':
                raise ValueError('Post-BF16 image edge measurement failed')
            nominal = 0. if side == p['reference_side'] else p['angle_deg']*p['tilt_sign']
            axes.append({'side': side, 'nominal_deg': nominal,
                         'PCA_deg': pca, 'edge_OLS_deg': edge['axis_deg'],
                         'PCA_error_deg': raster.angle_error(pca, nominal),
                         'edge_OLS_error_deg': raster.angle_error(edge['axis_deg'], nominal),
                         'observed_background_Y': bg, 'observed_foreground_Y': fg})
        result['axes'] = axes
    return result


def calibrate(manifest_path, output_dir, model_root, model_key, ready_path=None,
              asset_root=None, cosmos_package=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from .inference import (ProcessorSession, load_run_inputs, restore_processor_rgb,
                            checkpoint_asset_identity, _versions, _cosmos_package_identity)
    manifest, rows, ready, assets, ready_path = load_run_inputs(manifest_path, ready_path, asset_root)
    spec = manifest['models'][model_key]
    model_path = Path(model_root)/spec['checkpoint_relative']
    unique = {}
    for row in rows:
        unique.setdefault(state_key(row), row)
    if len(unique) != 128:
        raise ValueError('Unexpected unique source state denominator')
    dest = Path(output_dir)/model_key
    dest.mkdir(parents=True, exist_ok=True)
    identity = {'protocol_id': manifest['protocol_id'], 'model': spec,
                'ready_sha256': file_sha(ready_path), 'checkpoint_assets': checkpoint_asset_identity(model_path),
                'runtime_versions': _versions(),
                'cosmos_package_sources': _cosmos_package_identity(cosmos_package) if model_key == 'cosmos' else None,
                'dependent_sources': {name: file_sha(Path(__file__).with_name(name))
                                      for name in ('manifest.py', 'verify.py', 'raster.py')},
                'calibration_source_sha256': file_sha(__file__),
                'processor_runner_sha256': file_sha(Path(__file__).with_name('inference.py'))}
    with ProcessorSession(model_key, model_path, cosmos_package) as session:
        records = []
        for key, row in unique.items():
            path = dest/(key+'.json')
            if path.exists():
                record = json.loads(path.read_text())
                if (record['identity'] != identity or record['sample_id'] != row['sample_id']
                        or record['state_key'] != key or record['source_parameters'] != row['source_parameters']):
                    raise ValueError('Calibration cache identity changed; use a new directory')
            else:
                prepared = session.prepare(row, ready[row['sample_id']], assets)
                pixels = prepared['pixels'].to(torch.bfloat16)
                rgb = restore_processor_rgb(pixels, prepared['processor'], prepared['source_codes'], model_key)
                record = {'identity': identity, 'state_key': key, 'sample_id': row['sample_id'],
                          'source_parameters': row['source_parameters'], 'input_receipt': prepared['input_receipt'],
                          'BF16_measurement': image_measurement(row, prepared['source_codes'], rgb),
                          'new_model_calls': 0}
                atomic_json(path, record)
                del prepared, pixels, rgb
                gc.collect()
            records.append(record)
    t2 = [r['BF16_measurement']['absolute_nominal_error'] for r in records if r['BF16_measurement']['task'] == 'T2']
    axes = [a for r in records if r['BF16_measurement']['task'] == 'T3' for a in r['BF16_measurement']['axes']]
    result = {'status': 'COMPLETE', 'identity': identity, 'unique_states': len(records),
              'new_model_calls': 0, 'weights_loaded': False,
              'T2_BF16_max_absolute_nominal_delta_error': max(t2),
              'T3_BF16_max_image_axis_error_deg': max(max(a['PCA_error_deg'], a['edge_OLS_error_deg']) for a in axes),
              'interpretation': 'Observed deployed-input measurements, not source-label corrections or learned-feature error bounds.'}
    atomic_json(dest/'summary.json', result)
    return result
