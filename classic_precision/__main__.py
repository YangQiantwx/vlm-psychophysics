"""One command line for construction, inspection, inference and analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m classic_precision')
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('manifest', help='Freeze the exact 3504 scientific rows')
    init.add_argument('--run', type=Path, required=True)
    init.add_argument('--protocol', type=Path)
    gen = sub.add_parser('generate', help='Build/verify lossless masters with existing CPU cores')
    gen.add_argument('--run', type=Path, required=True)
    gen.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    gen.add_argument('--disk-floor-gib', type=float, default=10.)
    dl = sub.add_parser('download', help='Fetch pinned model snapshots; never start inference')
    dl.add_argument('--model-root', type=Path, required=True)
    dl.add_argument('--models', nargs='+', choices=['nemo', 'qwen', 'cosmos', 'gemma'])
    for name in ['run', 'supervise', 'calibrate']:
        p = sub.add_parser(name)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--ready', type=Path)
        p.add_argument('--assets', type=Path)
        p.add_argument('--model-root', type=Path, required=True)
        p.add_argument('--cosmos-package', type=Path)
        p.add_argument('--output', type=Path, required=True)
        if name == 'calibrate':
            p.add_argument('--model', choices=['nemo', 'qwen', 'cosmos', 'gemma'], required=True)
        else:
            p.add_argument('--gpu', required=True)
            p.add_argument('--shard-index', type=int, required=True)
            p.add_argument('--shard-count', type=int, default=3)
            if name == 'run':
                p.add_argument('--model', choices=['nemo', 'qwen', 'cosmos', 'gemma'], required=True)
                p.add_argument('--retry-failed', action=argparse.BooleanOptionalAction, default=True)
                p.add_argument('--retry-interrupted', action=argparse.BooleanOptionalAction, default=True)
            else:
                p.add_argument('--worker-python')
                p.add_argument('--wait-seconds', type=int, default=60)
    s = sub.add_parser('summarize', help='No model calls; refuse duplicate/mismatched answers')
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--inputs', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--require-complete', action='store_true')
    s.add_argument('--plot', action='store_true')
    status = sub.add_parser('status', help='Compact local journal snapshot; no GPU query')
    status.add_argument('--manifest', type=Path, required=True)
    status.add_argument('--inputs', type=Path, nargs='+', required=True)
    args = parser.parse_args(argv)
    if args.command == 'manifest':
        from .manifest import write_manifest, load_protocol
        result = write_manifest(args.run/'manifest.json', load_protocol(args.protocol))
        print(json.dumps({'protocol_id': result['protocol_id'], 'counts': result['counts']}))
    elif args.command == 'generate':
        from .manifest import validate_manifest
        from .stimuli import generate
        manifest = validate_manifest(json.loads((args.run/'manifest.json').read_text()))
        print(json.dumps(generate(manifest, args.run, args.workers, args.disk_floor_gib)))
    elif args.command == 'download':
        from .scheduler import download_models
        print(json.dumps(download_models(args.model_root, args.models)))
    elif args.command in ('run', 'supervise', 'calibrate'):
        common = dict(manifest_path=args.manifest, ready_path=args.ready, asset_root=args.assets,
                      output_dir=args.output, model_root=args.model_root, cosmos_package=args.cosmos_package)
        if args.command == 'run':
            from .inference import run_manifest
            result = run_manifest(**common, model_key=args.model, gpu=args.gpu, shard_index=args.shard_index,
                                  shard_count=args.shard_count, retry_failed=args.retry_failed,
                                  retry_interrupted=args.retry_interrupted)
        elif args.command == 'supervise':
            from .scheduler import supervise
            result = supervise(**common, gpu=args.gpu, shard_index=args.shard_index,
                               shard_count=args.shard_count, worker_python=args.worker_python,
                               wait_seconds=args.wait_seconds)
        else:
            from .calibration import calibrate
            result = calibrate(**common, model_key=args.model)
        print(json.dumps(result))
    elif args.command == 'summarize':
        from .summary import summarize, plot
        result = summarize(args.manifest, args.inputs, args.output, args.require_complete)
        if args.plot:
            plot(result, args.output)
        print(json.dumps({'status': result['status'], 'models': result['models'], 'cost': result['cost']}))
    else:
        from .summary import model_snapshot, technical_passed, attempt_metrics
        from .manifest import validate_manifest
        manifest = validate_manifest(json.loads(args.manifest.read_text()))
        result = {}
        for root in args.inputs:
            result[str(root)] = {}
            for family in manifest['models']:
                directory = root/family
                if not (directory/'identity.json').exists():
                    result[str(root)][family] = {'completed': 0, 'status': 'NOT_STARTED'}
                    continue
                _, rows, events, tail, event_tail = model_snapshot(directory)
                passed = technical_passed(rows, events)
                result[str(root)][family] = {'completed': len(rows), 'technical_passed': len(passed),
                    'technical_unresolved': len(rows)-len(passed), 'partial_tail': tail or event_tail,
                    'attempts': attempt_metrics(rows, events),
                    'last20_mean_call_s': sum(r['seconds'] for r in rows[-20:])/len(rows[-20:]) if rows else None}
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
