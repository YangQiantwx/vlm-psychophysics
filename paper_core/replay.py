"""Prepare a hash-bound trial or execute its archived local model protocol."""

from argparse import ArgumentParser
import gzip
from hashlib import sha1, sha256 as hash_bytes
import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from .media import decoded_sha, pixel_sha, read_video
from .runtime_recipe import RECIPE_PATH, check_versions, execute, prefill
from .scores import dataset, read_csv, read_json, require, sha256

STATIC_VIEWS = ('black_exact', 'scene_only', 'objects_only', 'black_only')


def checked(root, relative, digest):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Media must be below its designated root')
    require(sha256(path) == digest, (relative, 'media SHA-256'))
    return path


def study_file(args, manifest, name):
    return checked(args.study, name, manifest['files'][name]['sha256'])


def content_entry(kind, path):
    key = kind + '_url'
    return {'type': key, key: {'url': path.as_uri()}}


def frozen_ordered_rgb(manifest, trial):
    digest = trial.get('decoded_stream_sha256')
    if digest is not None:
        return digest
    rev = manifest['api_runner']['png_revision']
    repo = manifest['panel']['repo']
    url = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/api-images/v1/index.json.gz'
    with urlopen(url, timeout=60) as response:
        index_bytes = response.read()
    require(hash_bytes(index_bytes).hexdigest() == manifest['api_runner']['png_index_sha256'],
            'Frozen PNG inventory changed')
    source = json.loads(gzip.decompress(index_bytes))['sources'][trial['video_sha256']]
    require(len(source['png_sha256']) == 180, 'Frozen PNG frame count changed')
    return source['rgb_sha256']


def recipe(args, task):
    model, mode, variant, arm = args.model, args.mode, args.variant, args.arm
    if variant != 'original':
        if variant == 'enlarged':
            require(model == 'qwen' and task == 'T5' and mode == 'video' and arm == 'off', 'Enlargement uses native Qwen T5 Off')
            return 'qwen-enlarged'
        require(task == 'T3' and model in ('qwen', 'cosmos') and arm == 'off', 'T3 guides use Qwen/Cosmos Off')
        if mode == 'static' and variant in ('axis-base', *STATIC_VIEWS):
            return f'{model}-axis-static'
        require(mode == 'video' and variant in ('axis-base', 'axis'), 'Use axis-base/axis for T3 MP4 or a static view')
        return f'{model}-axis-video'
    if model == 'qwen':
        require(arm in ('off', 'low'), 'Qwen uses Off or Low')
        return 'qwen-ordered' if mode == 'png' else f'qwen-native-{mode}'
    if model == 'nemo':
        require(task == 'T5' and mode == 'video' and arm in ('off', 'low'), 'Nemotron control uses T5 native Off/Efficient')
        return 'nemo-t5'
    if model == 'cosmos':
        require(task in ('T4', 'T5') and mode == 'video' and arm == 'cot',
                'Cosmos prompted CoT uses native T4 or T5; use run.py for the archived plain baseline')
        return f'cosmos-cot-{task.lower()}'
    require(arm == 'direct' and mode != 'png', 'Muse uses native label-only video or static input')
    return 'muse'


def expected_axis_prompt(args, manifest, table, view=None):
    path = study_file(args, manifest, table)
    found = [r for r in read_csv(path) if r['sample_id'] == args.id and r['model'] == args.model
             and (view is None or r['view'] == view)]
    record, = found
    return record['prompt_sha256' if view is None else 'prompt_ids_sha256']


def request(args, videos, stills, manifest, config):
    trial = videos[args.id]
    media = []
    source_video_rgb = expected_prompt_sha = None
    if args.mode == 'static':
        selected = stills[args.id]
        prompt = selected['prompt']
        if args.variant == 'original':
            media = [('image', checked(args.data, i['path'], i['sha256'])) for i in selected['images']]
        else:
            panel = read_json(study_file(args, manifest, 't3-static-inputs.json'))['inputs'][args.id]
            view = 'base' if args.variant == 'axis-base' else args.variant
            spec = panel['input_digests'][view]
            if view == 'base':
                path = checked(args.data, panel['base_path'], spec['sha256'])
            else:
                require(args.axis_images is not None, 'Pass --axis-images with the regenerated T3 views')
                path = (args.axis_images / f'{view}.png').resolve()
                with Image.open(path) as image:
                    # The published recipe verifies decoded RGB independently of PNG compression.
                    import numpy as np
                    require(image.mode == 'RGB' and image.size == (768, 512) and
                            pixel_sha(np.asarray(image)) == spec['rgb_sha256'], 'Regenerated T3 image RGB differs')
            media = [('image', path)]
            prompt = config['static_axis_prompt']
            expected_prompt_sha = expected_axis_prompt(args, manifest, 't3-static-results.csv', view)
    else:
        prompt = trial['prompt']
        movie_path = trial.get('path', trial.get('video_path'))
        movie = checked(args.data, movie_path, trial['video_sha256'])
        source_video_rgb = trial.get('decoded_stream_sha256')
        if 'reference' in trial:
            ref = trial['reference']
            media = [('image', checked(args.data, ref['path'], ref['sha256']))]
        if args.variant in ('axis-base', 'axis'):
            axis = read_json(study_file(args, manifest, 't3-axis-inputs.json'))['inputs'][args.id]
            require(axis['base_sha256'] == trial['video_sha256'], 'Axis source SHA')
            if args.variant == 'axis':
                movie = checked(args.study, axis['marked_path'], axis['marked_sha256'])
            source_video_rgb = axis['marked_rgb_sha256' if args.variant == 'axis' else 'base_rgb_sha256']
            expected_prompt_sha = expected_axis_prompt(args, manifest, 't3-axis-results.csv')
        elif args.variant == 'enlarged':
            zoom = read_json(study_file(args, manifest, 't5-enlarged-inputs.json'))['inputs'][args.id]
            require(zoom['original_video_sha256'] == trial['video_sha256'], 'Enlargement source SHA')
            enlarged = [(m['type'], checked(args.study, m['path'], m['sha256'])) for m in zoom['media']]
            require([k for k, _ in enlarged] == ['image', 'video'], 'Enlargement media order')
            media, movie = enlarged[:1], enlarged[1][1]
            source_video_rgb = zoom['rgb_sha256']
        if args.mode == 'png':
            frames = read_video(movie)
            require(decoded_sha(frames) == frozen_ordered_rgb(manifest, trial), 'Ordered video RGB identity')
            frame_dir = args.out / 'frames'
            frame_dir.mkdir(exist_ok=True)
            for i, rgb in enumerate(frames):
                png = frame_dir / f'{i:03d}.png'
                Image.fromarray(rgb).save(png)
                media.append(('image', png.resolve()))
            prompt = config['recipes']['qwen-ordered']['frame_prompt_prefix'] + prompt
        else:
            media.append(('video', movie))
            if source_video_rgb is None:
                source_video_rgb = decoded_sha(read_video(movie))
    choices = ['YES', 'NO'] if trial['task'] == 'T5' else ['LEFT', 'RIGHT']
    return media, prompt, choices, expected_prompt_sha, source_video_rgb


def execute_muse(args, trial, media, config):
    from vlm_psychophysics import assets, data, glimmer

    require(args.weights and args.weights.name == glimmer.REVISION, 'Use the pinned Muse checkpoint')
    check_versions(config['packages'])
    require(sha256(Path(glimmer.__file__)) == config['processor_code']['glimmer'], 'Muse input source changed')
    files = {}
    for item in config['download']['files']:
        name, expected, size = item['name'], item['sha'], item['size']
        path = args.weights / name
        require(path.stat().st_size == size, (name, 'Muse checkpoint size differs'))
        if len(expected) == 64:
            files[name] = {'bytes': size, 'sha256': expected, 'weight': name.endswith('.safetensors')}
        else:
            content = path.read_bytes()
            require(sha1(f'blob {size}\0'.encode() + content).hexdigest() == expected,
                    (name, 'Muse snapshot metadata differs'))
            files[name] = {'bytes': size, 'sha256': hash_bytes(content).hexdigest()}
    material = {'files': files}
    target = assets._check_files(args.weights, material)
    assets._verify_weight_hashes(args.verification_cache, target, material)
    scope = f"{trial['domain']}/{trial['task'].lower()}"
    original, _ = data.build_selection(args.data, (scope,), (args.mode,), sample_id=args.id, offline=True)
    archived, = original['cases']
    case = data.runtime_case(archived, args.data)
    expected = [i['sha256'] for i in case.get('images', [])]
    if args.mode == 'video':
        expected.append(case['video_sha256'])
    require(expected == [sha256(path) for _, path in media], 'Muse source order or bytes differ')
    session = glimmer.Session(args.weights, args.data, args.out)
    try:
        require(session.profile_sha256 == config['profile_sha256'], 'Muse deployment/profile differs from the archive')
        return session.run(case, force_deep=True)
    finally:
        session.close()


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--id', required=True)
    ap.add_argument('--model', choices=('qwen', 'nemo', 'cosmos', 'glimmer'), required=True)
    ap.add_argument('--mode', choices=('video', 'static', 'png'), default='video')
    ap.add_argument('--arm', choices=('off', 'low', 'cot', 'direct'), required=True)
    ap.add_argument('--variant', choices=('original', 'axis-base', 'axis', *STATIC_VIEWS, 'enlarged'), default='original')
    ap.add_argument('--axis-images', type=Path, help='Folder of verified T3 static views produced by paper_core.media')
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--tokenizer', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--verification-cache', type=Path, default=Path.home() / '.cache/vlm-psychophysics')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    manifest = read_json(args.study / 'manifest.json')
    selected = sha256(args.data / 'dataset-index.json')
    require(selected in {manifest[k]['index_sha256'] for k in ('base', 'panel')}, 'Pin the base or panel dataset revision')
    if args.model == 'glimmer':
        require(selected == manifest['base']['index_sha256'], 'Use the exact full-set snapshot for Muse')
    config = read_json(RECIPE_PATH)
    args.out.mkdir(parents=True, exist_ok=True)
    videos, stills = dataset(args.data, selected)
    trial = videos[args.id]
    key = recipe(args, trial['task'])
    profile = config['recipes'].get(key, {})
    media, prompt, choices, prompt_sha, rgb_sha = request(args, videos, stills, manifest, config)
    messages = [{'role': 'user', 'content': [*[content_entry(k, p) for k, p in media], {'type': 'text', 'text': prompt}]}]
    prefill(messages, prompt, profile, args.arm)
    plan = dict(sample_id=args.id, model=args.model, mode=args.mode, arm=args.arm, variant=args.variant,
                execution_recipe=key, media_sha256=[sha256(p) for _, p in media], truth=trial['truth'],
                messages=messages, archived_prompt_tokens_sha256=prompt_sha, video_rgb_sha256=rgb_sha)
    (args.out / 'request.json').write_text(json.dumps(plan, indent=2) + '\n')
    if args.execute:
        if args.model == 'glimmer':
            muse = read_json(study_file(args, manifest, 'model-configs.json'))['muse']
            answer = execute_muse(args, trial, media, muse)
        else:
            answer = execute(args, profile, messages, choices, prompt_sha, rgb_sha)
        (args.out / 'answer.json').write_text(json.dumps(answer, indent=2) + '\n')
    print(f'Prepared {len(media)} verified inputs using {key} for {args.id}.')


if __name__ == '__main__':
    main()
