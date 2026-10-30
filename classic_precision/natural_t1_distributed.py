"""CPU shard orchestration for the already validated natural T1 RGB renderer.

This module never changes allocation, stimulus math, encoding or inference.
All shards use natural_t1_rgb.render_one and retain its full RGB/timestamp audit.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from . import natural_t1_rgb as t
from . import natural_t13_stimuli as s

HOSTS = ('fourier', 'radon')


def prepare(root):
    p, d, r = t.paths(root)
    dest = p / 'distributed.json'
    assert not dest.exists(), 'Keep the existing sealed distribution'
    blocks = s.read(p / 'allocation.json')['blocks']
    pilot = s.read(r / 'pilot-CPU-summary.json')
    pilot_ids = {b['block_id'] for b in pilot['rows']}
    assert len(pilot_ids) == 12
    remaining = [b for b in blocks if b['block_id'] not in pilot_ids]
    # Exact disjoint union; keep the previously rendered pilot on Fourier.
    radon_ids = {b['block_id'] for b in remaining[::3][:640]}
    # 1908 remaining / 3 gives636; add the next four unassigned blocks.
    for b in remaining:
        if len(radon_ids) == 640:
            break
        radon_ids.add(b['block_id'])
    assignments = {'radon': [b['block_id'] for b in blocks if b['block_id'] in radon_ids],
                   'fourier': [b['block_id'] for b in blocks if b['block_id'] not in radon_ids]}
    assert [len(assignments[h]) for h in HOSTS] == [1280, 640]
    assert set(assignments['fourier']).isdisjoint(assignments['radon'])
    assert set().union(*map(set, assignments.values())) == {b['block_id'] for b in blocks}
    record = {'status': 'SEALED_CPU_ONLY', 'videos': 3840, 'blocks': 1920,
              'assignments': assignments,
              'allocation_sha256': s.sha(p / 'allocation.json'),
              'criteria_sha256': s.sha(p / 'criteria.json'),
              'renderer_sha256': s.sha(t.__file__), 'shared_io_sha256': s.sha(s.__file__),
              'orchestrator_sha256': s.sha(__file__),
              'pilot_reference_sha256': s.sha(r / 'pilot-CPU-summary.json'),
              'model_calls': 0, 'method': 'Same frozen LR blocks, split only by execution host',
              'authorization': 'User approved fixed global gain0.5 and remote CPU generation on both hosts, complementary sync, then validated HF publication.'}
    s.write(dest, record)
    return {'status': record['status'], 'videos': {h: 2*len(v) for h,v in assignments.items()}}


def validated(root, host):
    p,d,r = t.paths(root)
    plan = s.read(p/'distributed.json')
    for path,key in [(p/'allocation.json','allocation_sha256'),(p/'criteria.json','criteria_sha256'),
                     (Path(t.__file__),'renderer_sha256'),(Path(s.__file__),'shared_io_sha256'),
                     (Path(__file__),'orchestrator_sha256'),(r/'pilot-CPU-summary.json','pilot_reference_sha256')]:
        assert s.sha(path) == plan[key], (path,key)
    decision=s.read(r/'execution-decision.json')
    assert decision['status']=='AUTHORIZED_DISTRIBUTED_CPU_GENERATION'
    assert decision['allocation_sha256']==plan['allocation_sha256']
    assert decision['global_linear_gain']==.5
    pilot=s.read(r/'pilot-acceptance.json')
    assert pilot['status']=='PASS_RGB_MULTIPLICATIVE_PILOT'
    assert pilot['visual_review']=='PASS_RAW_RGB_REPRESENTATIVE_INSPECTION'
    assert pilot['source_sha256']==plan['renderer_sha256']
    assert pilot['criteria_sha256']==plan['criteria_sha256']
    ids=set(plan['assignments'][host])
    return plan,[b for b in s.read(p/'allocation.json')['blocks'] if b['block_id'] in ids]


def pilot_identity(root):
    import numpy as np
    p,d,r=t.paths(root)
    features=s.read(p/'background-features.json')
    matches=[]
    for b in s.read(r/'pilot-CPU-summary.json')['rows']:
        block=next(x for x in s.read(p/'allocation.json')['blocks'] if x['block_id']==b['block_id'])
        source=features[block['background_id']]
        bgpath=Path(root)/source['prepared_relative']
        assert s.sha(bgpath)==source['prepared_sha256']
        bank,_=t.frame_bank(np.load(bgpath),block['sf'],block['m'],np.deg2rad(7.5+15*block['phase_index']))
        for c in b['cases']:
            direction=1 if c['truth']=='RIGHT' else -1
            digest=hashlib.sha256()
            for f in range(180): digest.update(bank[direction*f%12].tobytes())
            assert digest.hexdigest()==c['RGB_stream_sha256'],c['sample_id']
            matches.append(c['sample_id'])
    return {'status':'PASS_CURRENT_HOST_RGB_PARITY_WITH_ACCEPTED_PILOT','videos':len(matches),
            'sample_ids':matches,'reference_sha256':s.sha(r/'pilot-CPU-summary.json')}


def memory_available():
    fields=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0])*1024


def run_shard(root,host,workers):
    p,d,r=t.paths(root); plan,blocks=validated(root,host)
    import cv2
    cv2.setNumThreads(1)
    os.nice(5)
    start=time.monotonic()
    parity=pilot_identity(root)
    s.write(r/f'{host}-pilot-parity.json',parity)
    total=len(blocks); closed=0; existing=0; durations=[]; failures=[]
    # Each process is single-threaded; leave RAM for active model engines.
    reserve=(8 if host=='fourier' else 4)*1024**3
    estimate=400*1024**2
    ceiling=min(workers,max(1,(memory_available()-reserve)//estimate))
    s.write(r/f'{host}-worker-start.json',{'pid':os.getpid(),'hostname':platform.node(),
        'workers_requested':workers,'workers_effective':ceiling,'blocks':total,
        'allocation_sha256':plan['allocation_sha256'],'monotonic':start,
        'env_threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        'model_calls':0})
    def progress(active):
        out={'status':'RUNNING_CPU_ONLY','host':host,'closed_blocks':closed,'videos':2*closed,
             'total_blocks':total,'active':active,'existing_blocks':existing,'workers':ceiling,
             'elapsed_seconds':time.monotonic()-start,'recent_pair_seconds':durations[-20:],
             'failures':failures,'monotonic':time.monotonic(),'model_calls':0}
        s.write(r/f'{host}-progress.json',out)
    # Bounded submission: on failure finish in-flight work but issue nothing new.
    pending={}; iterator=iter(blocks); exhausted=False
    with ProcessPoolExecutor(max_workers=ceiling) as pool:
        while pending or not exhausted:
            while not exhausted and not failures and len(pending)<ceiling:
                try: b=next(iterator)
                except StopIteration: exhausted=True;break
                pending[pool.submit(t.render_one,(str(root),b))]=b['block_id']
            if failures: exhausted=True
            if not pending: break
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                block_id=pending.pop(future)
                try:
                    result=future.result();closed+=1
                    if result.get('existing'):existing+=1
                    else:durations.append(result['seconds'])
                except Exception:
                    failures.append({'block_id':block_id,'traceback':traceback.format_exc()})
            progress(len(pending))
    result={'status':'FAILED_DRAINED' if failures else 'COMPLETE_CPU_SHARD',
            'host':host,'closed_blocks':closed,'total_blocks':total,
            'videos':2*closed,'failures':failures,'elapsed_seconds':time.monotonic()-start,
            'allocation_sha256':plan['allocation_sha256'],'model_calls':0}
    s.write(r/f'{host}-result.json',result)
    assert not failures and closed==total,result
    return result


def supervise(root,host,workers):
    p,d,r=t.paths(root);validated(root,host);r.mkdir(parents=True,exist_ok=True)
    lockpath=r/f'{host}.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inode=os.fstat(lock.fileno()).st_ino
        assert not (r/f'{host}-launch.json').exists(),'Existing attempt retained; inspect before resuming'
        env=os.environ.copy()
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='1'
        env['CUDA_VISIBLE_DEVICES']=''
        command=[sys.executable,'-B','-m','classic_precision.natural_t1_distributed','worker','--root',str(root),'--host',host,'--workers',str(workers)]
        with (r/f'{host}-worker.log').open('xb') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            s.write(r/f'{host}-launch.json',{'supervisor_pid':os.getpid(),'child_pid':child.pid,
                'command':command,'lock_inode':inode,'orchestrator_sha256':s.sha(__file__),'model_calls':0})
            rc=child.wait()
        fcntl.flock(lock,fcntl.LOCK_UN)
    with lockpath.open('a+') as check:
        assert os.fstat(check.fileno()).st_ino==inode
        fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(check,fcntl.LOCK_UN)
    result={'actual_child_wait':rc,'original_lock_inode':inode,'original_lock_free':True,
            'worker_returned_after_pool_join':True,'host':host,'model_calls':0}
    s.write(r/f'{host}-wait.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','supervise','worker'])
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--host',choices=HOSTS)
    ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    if args.command=='prepare':result=prepare(args.root)
    elif args.command=='supervise':result=supervise(args.root,args.host,args.workers)
    else:result=run_shard(args.root,args.host,args.workers)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
