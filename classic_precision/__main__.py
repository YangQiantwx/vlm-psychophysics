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
    gen.add_argument('--workers', '--cpu-max-workers', type=int)
    gen.add_argument('--disk-floor-gib', type=float, default=10.)
    gen.add_argument('--shard-count', type=int, default=3)
    validate = sub.add_parser('validate', help='Independently decode published RGB16 videos against the source states')
    validate.add_argument('--run', type=Path, required=True)
    validate.add_argument('--workers', '--cpu-max-workers', type=int)
    validate.add_argument('--require-complete', action='store_true')
    integrated = sub.add_parser('pipeline', help='Local finite CPU/input/calibration/GPU process pipeline')
    integrated.add_argument('--run', type=Path, required=True)
    integrated.add_argument('--model-root', type=Path, required=True)
    integrated.add_argument('--gpu-shards', nargs='+', required=True, help='GPU:shard pairs, e.g.0:0 1:1')
    integrated.add_argument('--shard-count', type=int, default=3)
    integrated.add_argument('--produce', action='store_true')
    integrated.add_argument('--output', type=Path)
    integrated.add_argument('--worker-python')
    integrated.add_argument('--cosmos-package', type=Path)
    integrated.add_argument('--cpu-max-workers', type=int)
    integrated.add_argument('--start-ready', type=int, default=100)
    integrated.add_argument('--input-poll-seconds', type=float, default=10)
    integrated.add_argument('--sync-destination-root')
    integrated.add_argument('--destination-ssh')
    integrated.add_argument('--destination-python')
    integrated.add_argument('--destination-code-root')
    for command in (gen, validate, integrated):
        command.add_argument('--cpu-memory-gib', type=float, default=32)
        command.add_argument('--ram-reserve-gib', type=float, default=4)
    sync = sub.add_parser('sync-ready', help='Incrementally copy only qualified inputs; local or SSH endpoints')
    sync.add_argument('--manifest', type=Path, required=True)
    sync.add_argument('--source-root', required=True)
    sync.add_argument('--destination-root', required=True)
    for endpoint in ('source', 'destination'):
        sync.add_argument('--'+endpoint+'-ssh')
        sync.add_argument('--'+endpoint+'-python')
        sync.add_argument('--'+endpoint+'-code-root')
    sync.add_argument('--poll-seconds', type=float, default=10)
    sync.add_argument('--once', action='store_true')
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
        p.add_argument('--input-poll-seconds', type=float, default=10)
        if name == 'calibrate':
            p.add_argument('--model', choices=['nemo', 'qwen', 'cosmos', 'gemma'], required=True)
        else:
            p.add_argument('--start-ready', type=int, default=100)
            p.add_argument('--calibration-root', type=Path)
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
    # CPU children inherit one codec/BLAS thread before importing numerical SDKs.
    from .cpu import configure_threads
    configure_threads()
    if args.command in ('generate', 'validate', 'calibrate', 'sync-ready', 'supervise', 'run'):
        import signal
        def stop(signum, frame):
            raise KeyboardInterrupt('Received signal '+str(signum))
        signal.signal(signal.SIGTERM, stop)
    if args.command == 'manifest':
        from .manifest import write_manifest, load_protocol
        result = write_manifest(args.run/'manifest.json', load_protocol(args.protocol))
        print(json.dumps({'protocol_id': result['protocol_id'], 'counts': result['counts']}))
    elif args.command == 'generate':
        from .manifest import validate_manifest
        from .stimuli import generate
        manifest = validate_manifest(json.loads((args.run/'manifest.json').read_text()))
        print(json.dumps(generate(manifest, args.run, args.workers, args.disk_floor_gib, args.shard_count,
                                  args.cpu_memory_gib, args.ram_reserve_gib)))
    elif args.command == 'validate':
        from .stimuli import validate_inputs
        print(json.dumps(validate_inputs(args.run, args.workers, args.require_complete,
                                         args.cpu_memory_gib, args.ram_reserve_gib)))
    elif args.command == 'pipeline':
        from .pipeline import pipeline
        pairs = [pair.rsplit(':', 1) for pair in args.gpu_shards]
        print(json.dumps(pipeline(args.run, args.model_root, pairs, args.shard_count, args.produce,
            args.output, args.worker_python, args.cosmos_package, args.cpu_max_workers, args.cpu_memory_gib,
            args.ram_reserve_gib, args.start_ready, args.input_poll_seconds, args.sync_destination_root,
            args.destination_ssh, args.destination_python, args.destination_code_root)))
    elif args.command == 'sync-ready':
        from .ready import sync_ready
        print(json.dumps(sync_ready(args.manifest, args.source_root, args.destination_root,
            args.source_ssh, args.destination_ssh, args.source_python, args.destination_python,
            args.source_code_root, args.destination_code_root, args.poll_seconds, args.once)))
    elif args.command == 'download':
        from .scheduler import download_models
        print(json.dumps(download_models(args.model_root, args.models)))
    elif args.command in ('run', 'supervise', 'calibrate'):
        common = dict(manifest_path=args.manifest, ready_path=args.ready, asset_root=args.assets,
                       output_dir=args.output, model_root=args.model_root, cosmos_package=args.cosmos_package,
                       input_poll_seconds=args.input_poll_seconds)
        if args.command != 'calibrate':
            common.update(start_ready=args.start_ready, calibration_root=args.calibration_root)
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
