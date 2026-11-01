"""Bounded ICC repair of three photographic cutouts, without rendering video.

The existing 300 assets remain immutable. Embedded profiles are audited for all
assets; only the three independently identified non-sRGB objects are converted.
All output geometry and alpha values retain their original scientific meaning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
import PIL
from PIL import Image, ImageCms

from . import natural_t2_stimuli as t2
from . import natural_t13_stimuli as t3

ASSETS = frozenset({'photo_obj_025', 'photo_obj_033', 'photo_obj_048'})
CATALOG = 'plans/natural-t3-spoon-20261030/catalog.json'
STAGING = 'plans/dataset-naming-migration/public-staging'
INTENT = 0  # PERCEPTUAL, matching the independent diagnostic.
FLAGS = 0
SRGB_EQUIVALENT_C2 = '61bb24bd075ae1179b562cb484a2737fa63bcf35133a951843db1f1f7c97ce7f'
read, write, sha, digest = t2.read, t2.write, t2.sha, t2.digest


def raw_sha(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def sdk():
    return {'Pillow': PIL.__version__, 'LittleCMS': ImageCms.core.littlecms_version,
            'rendering_intent': INTENT, 'rendering_intent_name': 'PERCEPTUAL', 'flags': FLAGS,
            'function': 'PIL.ImageCms.profileToProfile', 'output_profile': 'LittleCMS built-in sRGB',
            'output_profile_header': 'creation date fixed 2000-01-01; unused profile ID zeroed for deterministic assets'}


def convert_rgba(image, profile_bytes):
    """Apply the actual source ICC to RGB; preserve alpha and invisible RGB exactly."""
    if not profile_bytes:
        raise ValueError('ICC conversion requires a valid embedded source profile')
    try:
        source = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
        target = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
        canonical = bytearray(target.tobytes())
        canonical[24:36] = struct.pack('>6H', 2000, 1, 1, 0, 0, 0)
        canonical[84:100] = bytes(16)
        result = ImageCms.profileToProfile(image.convert('RGB'), source, target,
                                          renderingIntent=INTENT, outputMode='RGB', flags=FLAGS)
    except (ImageCms.PyCMSError, OSError, ValueError, TypeError) as error:
        raise ValueError('Invalid or unsupported embedded ICC profile') from error
    original = np.array(image.convert('RGBA'))
    out = original.copy(); visible = original[..., 3] > 0
    out[..., :3][visible] = np.asarray(result)[visible]
    assert np.array_equal(out[..., 3], original[..., 3])
    return Image.fromarray(out, 'RGBA'), bytes(canonical)


def profile_info(blob):
    if not blob:
        return {'profile_status': 'missing', 'policy': 'preserve original sRGB-assumed values; no inferred conversion'}
    try:
        profile = ImageCms.ImageCmsProfile(BytesIO(blob))
        description = ImageCms.getProfileDescription(profile).strip()
    except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
        return {'profile_status': 'invalid', 'ICC_sha256': raw_sha(blob),
                'policy': 'preserve original bytes outside selected scope; selected conversion must fail'}
    equivalent = description.lower().startswith('srgb') or raw_sha(blob) == SRGB_EQUIVALENT_C2
    return {'profile_status': 'valid', 'ICC_description': description, 'ICC_sha256': raw_sha(blob),
            'profile_interpretation': 'named_sRGB_or_independently_audited_sRGB_equivalent' if equivalent else 'other_valid_profile',
            'policy': 'retain original; independently measured sRGB-equivalent differences at most one code' if equivalent
                      else 'convert only if explicitly selected by the three-asset source repair allowlist'}


def inspect_asset(args):
    root, asset = args; path = Path(root)/'assets/library'/asset['path']
    assert sha(path) == asset['sha256'], asset['id']
    with Image.open(path) as image:
        blob = image.info.get('icc_profile'); info = profile_info(blob)
        return {'asset_id': asset['id'], 'path': asset['path'], 'source_sha256': asset['sha256'],
                'mode': image.mode, 'dimensions': list(image.size), **info,
                'action': 'convert_actual_embedded_ICC_to_sRGB' if asset['id'] in ASSETS else 'retain_original'}


def prepare_object(root, asset, output, probe):
    root, output = Path(root), Path(output)
    path = root/'assets/library'/asset['path']; raw = root/'assets/library'/asset['raw_path']
    assert sha(path) == asset['sha256'] and sha(raw) == asset['raw_sha256']
    with Image.open(path) as cutout, Image.open(raw) as original:
        blob = cutout.info.get('icc_profile'); raw_blob = original.info.get('icc_profile')
        assert blob and blob == raw_blob, 'Cutout profile must equal embedded photographic source profile'
        pixels = np.array(cutout.convert('RGBA')); alpha = pixels[..., 3]
        transform = asset['transform']; assert transform['resize'] == 'none'
        crop = original.convert('RGB').crop(transform['source_crop_xyxy']).crop(transform['alpha_bbox_crop_xyxy'])
        foreground = alpha > 0
        assert crop.size == cutout.size
        assert np.array_equal(np.asarray(crop)[foreground], pixels[..., :3][foreground])
        converted, output_profile = convert_rgba(cutout, blob)
    new = np.array(converted); diff = abs(new[..., :3].astype(int)-pixels[..., :3].astype(int))
    maximum = int(diff[foreground].max())
    mean = float((diff*alpha[..., None]/255).sum()/(3*(alpha/255).sum()))
    old_y = float((t2.linear(pixels[..., :3]/255)@t2.Y*(alpha/255)).sum()/(alpha/255).sum())
    new_y = float((t2.linear(new[..., :3]/255)@t2.Y*(alpha/255)).sum()/(alpha/255).sum())
    assert maximum == probe['foreground_max_code_difference']
    assert abs(mean-probe['foreground_alpha_weighted_mean_absolute_channel_difference_codes']) < 1e-12
    assert abs(new_y-probe['ICC_to_sRGB_mean_linear_Y']) < 1e-12
    target = output/'assets'/(asset['id']+'.png'); target.parent.mkdir(parents=True, exist_ok=True)
    converted.save(target, icc_profile=output_profile)
    assert np.array_equal(np.asarray(Image.open(target)), new)
    a = alpha.astype(float)/255
    source = np.dstack([t2.linear(new[..., :3]/255)*a[..., None], a])
    # Reuse the exact existing source-to-output affine, not a newly fitted pose.
    oldprep = read(root/'datasets'/t2.NAME/'prepared-vertical-v2'/(asset['id']+'.json'))
    oldfile = root/'datasets'/t2.NAME/'prepared-vertical-v2'/(asset['id']+'.npz')
    assert oldprep['source_sha256'] == asset['sha256'] and sha(oldfile) == oldprep['prepared_sha256']
    z = t3.affine(source, np.asarray(oldprep['source_to_output_matrix']), size=384, samples=8)
    with np.load(oldfile) as old:
        assert np.array_equal(z[..., 3], old['alpha']), 'T2 alpha changed'
    mu = float((z[..., :3]@t2.Y).sum()/z[..., 3].sum()); assert mu+max(t2.DELTAS) < 1
    prepfile = output/'prepared/T2'/(asset['id']+'.npz'); prepfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepfile, pm=z[..., :3], alpha=z[..., 3])
    prep = {**oldprep, 'source_sha256': sha(target), 'parent_source_sha256': asset['sha256'],
            'prepared_sha256': sha(prepfile), 'features': [mu, float(z[..., 3].sum())], 'baseline_mean_Y': mu,
            'alpha_exactly_unchanged': True}
    write(prepfile.with_suffix('.json'), prep)
    prep3 = None
    if asset['id'] == 'photo_obj_025':
        old3 = root/'datasets'/t3.NAME/'objects'; base = read(old3/(asset['id']+'.json'))
        assert base['source_sha256'] == asset['sha256']
        oldsource = np.load(old3/(asset['id']+'-source.npy'), mmap_mode='r')
        assert np.array_equal(source[..., 3], oldsource[..., 3])
        theta = np.deg2rad(90-base['source_axis_deg']); matrix = np.eye(3)
        matrix[:2, :2] = base['scale']*np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        matrix[:2, 2] = 159.5-matrix[:2, :2]@np.array(base['center'])
        zero = t3.affine(source, matrix)
        assert np.array_equal(zero[..., 3], np.load(old3/(asset['id']+'-zero.npy'))[..., 3])
        aa = zero[..., 3]; yy = zero[..., :3]@t2.Y; mu3 = float(yy.sum()/aa.sum())
        var = float(((np.divide(yy, aa, out=np.zeros_like(aa), where=aa>0)-mu3)**2*aa).sum()/aa.sum())
        prep3 = deepcopy(base); prep3.update(source_sha256=sha(target), parent_source_sha256=asset['sha256'],
                 features=[mu3, float(np.sqrt(var)), *base['features'][2:]], alpha_exactly_unchanged=True)
        dest = output/'prepared/T3'; dest.mkdir(parents=True, exist_ok=True)
        np.save(dest/(asset['id']+'-source.npy'), source); np.save(dest/(asset['id']+'-zero.npy'), zero)
        prep3['source_array_sha256'] = sha(dest/(asset['id']+'-source.npy'))
        write(dest/(asset['id']+'.json'), prep3)
    return {'asset_id': asset['id'], 'parent_source_sha256': asset['sha256'], 'source_sha256': sha(target),
            'raw_path': asset['raw_path'], 'raw_sha256': asset['raw_sha256'], 'relative_path': str(target.relative_to(output)),
            **profile_info(blob), 'output_ICC_sha256': raw_sha(output_profile), 'sdk': sdk(),
            'RGBA_alpha_sha256': raw_sha(alpha.tobytes()), 'alpha_byte_equal': True,
            'invisible_RGB_byte_equal': True, 'foreground_original_raw_RGB_byte_equal': True,
            'foreground_max_code_difference': maximum,
            'foreground_alpha_weighted_mean_absolute_channel_difference_codes': mean,
            'unmanaged_sRGB_assumed_mean_linear_Y': old_y, 'ICC_to_sRGB_mean_linear_Y': new_y,
            'T2_preparation': prep, 'T3_preparation': prep3}


def prepare(root, output, independent_audit, jobs=4):
    root, output = Path(root), Path(output)
    audit = read(independent_audit); catalog = read(root/CATALOG)
    assets = catalog['backgrounds']+catalog['objects']; assert len(assets) == 300
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        headers = list(pool.map(inspect_asset, [(str(root), a) for a in assets]))
    probes = {r['asset_id']: r for r in audit['checks']['ICC_profile_to_sRGB_source_probe']}
    converted = [prepare_object(root, a, output, probes[a['id']]) for a in assets if a['id'] in ASSETS]
    ids = audit['affected_current_video_ids']; assert {t:len(ids[t]) for t in ids} == {'T2':116, 'T3':68}
    report = {'schema':'natural-source-ICC-repair-v1', 'status':'CANDIDATE_ASSETS_AND_ALPHA_PASS_NO_VIDEO_RENDER',
        'sdk':sdk(), 'parent_catalog_sha256':sha(root/CATALOG), 'independent_audit_sha256':sha(independent_audit),
        'asset_count':300, 'unchanged_assets':297, 'asset_headers':headers, 'converted_assets':converted,
        'profile_counts':dict(Counter(r.get('ICC_description',r['profile_status']) for r in headers)),
        'exact_current_video_allowlist':ids, 'model_calls':0,
        'scope':'Only source-profile RGB interpretation changes. Alpha, pose, geometry, background, GT and numerical stimulus targets retain their definitions.'}
    # A small explicit patch for source-assets metadata; unchanged provenance,
    # attribution and source records remain in the parent catalog.
    write(output/'source-catalog-delta.json',{'schema':'natural-source-color-catalog-delta-v1',
        'parent_catalog_sha256':sha(root/CATALOG), 'relative_paths_from':'this directory',
        'unchanged_assets':297, 'changes':[dict(source_id=r['asset_id'],
            parent_sha256=r['parent_source_sha256'],sha256=r['source_sha256'],path=r['relative_path'],
            raw_path=r['raw_path'],raw_sha256=r['raw_sha256'],
            color_management={k:r[k] for k in ['ICC_description','ICC_sha256','output_ICC_sha256','sdk']},
            alpha_byte_equal=True, native_downloaded_RGB_unchanged=False,
            operation='Embedded source ICC to sRGB RGB conversion; existing straight alpha copied exactly')
            for r in converted]})
    write(output/'color-repair.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--independent-audit', required=True); p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args(); report = prepare(a.root, a.output, a.independent_audit, a.jobs)
    print(json.dumps({k:report[k] for k in ['status','sdk','asset_count','unchanged_assets','profile_counts']},indent=2))


if __name__ == '__main__':
    main()
