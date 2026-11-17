"""Verify the pinned Molmo2 BF16 study and pass it to its archived runner."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys

from vlm_psychophysics import assets
from ..runtime_recipe import RECIPE_PATH, check_versions
from ..scores import read_json, require, sha256
from ..scores import dataset

ROOT = Path(__file__).resolve().parent


def prepare(checkpoint, output, cache, spec):
    receipt = spec['checkpoint']
    require(checkpoint.name == receipt['revision'], 'Use the named official BF16 snapshot')
    names = {
        'runner': 'run_molmo2_noresize_subset.py',
        'input_bridge': 'molmo2_noresize.py',
        'prefix_attention': 'molmo2_prefix_attention.py',
    }
    for key, name in names.items():
        require(sha256(ROOT / name) == spec['source_hashes'][key], (name, 'archived source SHA differs'))
    for name, digest in spec['processor'].items():
        require(sha256(checkpoint / name) == digest, (name, 'Molmo2 tokenizer or processor SHA differs'))
    material = {'files': {name: {**item, 'weight': True} for name, item in receipt['weights'].items()}}
    path = assets._check_files(checkpoint, {**material, 'files': {**material['files'], **{
        name: {'bytes': (checkpoint / name).stat().st_size, 'sha256': digest}
        for name, digest in spec['processor'].items()}}})
    assets._verify_weight_hashes(cache, path, material)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoint-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu-lock', type=Path)
    ap.add_argument('--verification-cache', type=Path, default=Path.home() / '.cache/vlm-psychophysics')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    spec = read_json(RECIPE_PATH)['molmo']
    require(args.dataset.name == spec['dataset_revision'], 'Use a dataset directory named for the pinned panel revision')
    dataset(args.dataset, spec['dataset_index_sha256'])
    if args.execute:
        require(args.gpu_lock is not None and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),
                'Select one GPU and its local lock file')
        check_versions(spec['package_versions'])
    prepare(args.checkpoint, args.output, args.verification_cache, spec)
    if args.execute:
        subprocess.run([sys.executable, str(ROOT / 'run_molmo2_noresize_subset.py'),
                        '--checkpoint', str(args.checkpoint), '--dataset', str(args.dataset),
                        '--output', str(args.output), '--gpu-lock', str(args.gpu_lock), *spec['flags']], check=True)
    print('Verified the archived Molmo2 BF16 model and execution sources.')


if __name__ == '__main__':
    main()
