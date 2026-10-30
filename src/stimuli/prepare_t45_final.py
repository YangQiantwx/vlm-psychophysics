"""Finite T4/T5 classic-natural construction and full RGB8/PTS CPU verification.

No model imports or answers. Historical source files are read-only. Only completed,
fully decoded sources enter the append-only, atomically replaced ready catalogs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import av
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

NAME = 't45-final-20261030'
W, H, FPS, FRAMES = 768, 512, 60, 180
BG, FG = .10, .82
GROUPS = (('photo', 'photo'), ('photo', 'gpt'), ('gpt', 'photo'), ('gpt', 'gpt'))
CENTERS = ((160, 256), (608, 256), (384, 128), (384, 384))
Y = np.array([.2126, .7152, .0722])
T4_PROMPT = ('Watch the moving elements against the stationary background and follow their positions over time. '
    'Your task is to identify the horizontal motion signal and decide whether it points left or right relative to the background. '
    'Some or all elements may move in that direction; other elements, when present, can move in different directions. '
    'The signal does not have to be carried by a majority of the elements. Consider their movements across the video, '
    'not which side contains more elements in a single image. Object appearance, orientation, and identity do not determine the answer. '
    'Judge displacement across successive moments rather than where an element happens to finish. '
    'Give your final answer as exactly one uppercase word: LEFT or RIGHT.')
T5_PROMPT = ('The first input is a single reference image showing the target object in the orientation that counts as a match. '
    'In the video, objects appear in one or two separate events. Determine whether any event shows that object in exactly the reference orientation, '
    'comparing orientation in the image plane while ignoring position, size, and background. A rotated version does not count as a match. '
    'Judge each object during its visible presentation. If a line pattern replaces an object, this mask marks the end of that presentation; '
    'it is not another target orientation. A match still counts if it appeared earlier and is no longer visible at the end. '
    'Answer YES if any event matches, or NO if none does. Give your final answer as exactly one uppercase word: YES or NO.')


def read(p): return json.loads(Path(p).read_text())
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); q.replace(p)
def linear(x): return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
def rgb8(x):
    assert np.isfinite(x).all() and x.min() >= -1e-10 and x.max() <= 1 + 1e-10
    x = np.clip(x, 0, 1)
    return np.rint(255 * np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)).astype(np.uint8)
def paths(root):
    r = Path(root); return r/'plans'/NAME, r/'datasets'/NAME, r/'runs/stimulus-preparation'/NAME


def initialize(root):
    """Freeze exact classic conditions and input bindings; no media generation."""
    root = Path(root); p, d, out = paths(root)
    for folder in (p, d/'videos', d/'qa', d/'references', d/'prepared', out): folder.mkdir(parents=True, exist_ok=True)
    inputs = p/'inputs'; inputs.mkdir(exist_ok=True)
    source = root/'runs/stimulus-preparation/t4-coherence-final-20261030/geometry-audit.json'
    ref = root/'runs/stimulus-preparation/t5-square-reference-20261030/T5-reference.png'
    mask = root/'assets/scene-pilot/assets/L/mask.png'
    for src, name in ((source, 'geometry-audit.json'), (ref, 'T5-reference.png'), (mask, 'mask.png')):
        dst = inputs/name
        if dst.exists(): assert sha(dst) == sha(src)
        else: shutil.copy2(src, dst)
    assert sha(inputs/'T5-reference.png') == '40592f247066d5ce225e106f06c5fbef6906ea745b697ecd88b91028cf179ce3'
    assert sha(inputs/'mask.png') == '9649d9a37d10a0086971cf42fd6873c8003cc0ecf6e190be1bd408be639fd3c6'
    geometry = read(inputs/'geometry-audit.json'); assert geometry['passed_seeds'] == 24
    rows = []
    for layout in geometry['layouts']:
        for c in layout['conditions']:
            rows.append(dict(task='T4', seed=layout['seed'], coherence=c['coherence'], direction=c['direction'],
                truth=None if c['coherence'] == 0 else c['direction'], accuracy_eligible=c['coherence'] > 0,
                choices=['LEFT', 'RIGHT'], prompt=T4_PROMPT, paired_block=f'T4-s{layout["seed"]}',
                positions_xy_px=layout['positions_xy_px'], velocity_xy_px_s=c['velocity_xy_px_s'],
                signal_indices=c['signal_indices'], source_parameters=dict(seed=layout['seed'], coherence=c['coherence'],
                signal_count=c['signal_count'], direction=c['direction'], diameter_px=56, N=16, speed_px_s=32,
                minimum_continuous_disc_gap_px=c['minimum_disc_gap_px'], minimum_edge_margin_px=c['minimum_edge_margin_px'])))
    for k in range(12):
        for a in range(2):
            for scenario, presentation, truth in (('target_first', 'dual1', 'YES'), ('target_second', 'dual2', 'YES'),
                    ('target_absent_a', 'dual1', 'NO'), ('target_absent_b', 'dual2', 'NO')):
                for mask_name in ('none', 'persistent'):
                    probe = int(presentation[-1])-1; angles = [180, 180]; angles[probe] = 0 if truth == 'YES' else 90
                    params = dict(k=k, position_pair=a, onset=18+3*k, presentation=presentation, scenario=scenario,
                        mask=mask_name, angles_deg=angles, exposure_frames=30, ISI_frames=30, SOA_frames=60,
                        guide_size_px=160, target_size_px=128, target_stroke_px=28, mask_size_px=160,
                        centers_xy=[list(CENTERS[a]), list(CENTERS[2+(a ^ (k%2))])])
                    rows.append(dict(task='T5', **params, truth=truth, accuracy_eligible=True, choices=['YES', 'NO'],
                        prompt=T5_PROMPT, paired_block=f'T5-k{k:02d}-p{a}', source_parameters=params))
    assert Counter(r['task'] for r in rows) == {'T4': 408, 'T5': 192}
    # Exact finite card allocation, preserving every LR pair and all eight T5
    # occurrence/mask conditions at a timing-position block.
    cards = ('fourier0','fourier1','radon0')
    pairs = sorted({(r['coherence'],r['seed']) for r in rows if r['task']=='T4' and r['coherence']>0})
    pair_cards = dict(zip(pairs, ['fourier0']*76+['fourier1']*76+['radon0']*40))
    # A fixed shuffle breaks contiguous coherence/card coupling without looking
    # at pixels, source features, or any model outputs.
    order = np.random.default_rng(451030).permutation(len(pairs))
    pool=['fourier0']*76+['fourier1']*76+['radon0']*40
    pair_cards={pairs[int(j)]:pool[i] for i,j in enumerate(order)}
    blocks=[(k,a) for k in range(12) for a in range(2)]
    order=np.random.default_rng(551030).permutation(24);pool=['fourier0']*9+['fourier1']*9+['radon0']*6
    block_cards={blocks[int(j)]:pool[i] for i,j in enumerate(order)}
    for row in rows:
        if row['task']=='T4': row['card']=cards[(row['seed']-101)%3] if row['coherence']==0 else pair_cards[row['coherence'],row['seed']]
        else: row['card']=block_cards[row['k'],row['position_pair']]
        row['domain'] = 'classic'; row['sample_id'] = 'c45_' + digest({'protocol':NAME, **row})[:24]
    assert Counter(r['card'] for r in rows)=={'fourier0':232,'fourier1':232,'radon0':136}
    # Put both tasks near each shard's beginning for first-scientific-item input
    # admission, while retaining the complete fixed assignment.
    reordered=[]
    for card in cards:
        shard=[r for r in rows if r['card']==card]
        first_t4=[r for r in shard if r['task']=='T4'][:2]
        first_t5=[r for r in shard if r['task']=='T5'][:8]
        chosen={r['sample_id'] for r in first_t4+first_t5}
        reordered+=first_t4+first_t5+[r for r in shard if r['sample_id'] not in chosen]
    rows=reordered
    manifest = dict(schema='t45-final-manifest-v1', protocol_id=NAME, items=rows, classic_count=600,
        source_geometry_sha256=sha(inputs/'geometry-audit.json'), classic_reference_sha256=sha(inputs/'T5-reference.png'),
        mask_source_sha256=sha(inputs/'mask.png'), implementation_sha256=sha(__file__),
        codec='libx264rgb CRF0 preset ultrafast RGB8 gbrp; every frame decoded and byte-compared',
        photometry=dict(nominal_linear_Y={'background':BG, 'foreground':FG},
            effective_RGB8={'background':int(rgb8(np.array(BG))), 'foreground':int(rgb8(np.array(FG)))},
            effective_decoded_linear_Y={'background':float(linear(rgb8(np.array(BG))/255)),
                'foreground':float(linear(rgb8(np.array(FG))/255))}),
        model_calls=0, no_answer_based_selection=True)
    dest = p/'manifest.json'
    if dest.exists(): assert read(dest) == manifest, 'Frozen manifest differs'
    else: write(dest, manifest)
    shutil.copy2(inputs/'T5-reference.png', d/'references/T5-classic.png')
    return manifest


@lru_cache(maxsize=1)
def geometry_by_seed(root):
    p, _, _ = paths(root); return {x['seed']:x for x in read(p/'inputs/geometry-audit.json')['layouts']}


def circle_area(x0, x1, y0, y1, radius=28.):
    """Exact circle-box integral; same formula as the certified classic renderer."""
    lo, hi = max(x0, -radius), min(x1, radius)
    if lo >= hi: return 0.
    cuts = [lo, hi]
    for y in (y0, y1):
        if abs(y) < radius:
            q = math.sqrt(radius*radius-y*y); cuts.extend(x for x in (-q, q) if lo < x < hi)
    cuts = sorted(set(cuts))
    def primitive(x):
        return .5*(x*math.sqrt(max(0.,radius*radius-x*x))+radius*radius*math.asin(max(-1.,min(1.,x/radius))))
    area = 0.
    for a, b in zip(cuts[:-1], cuts[1:]):
        h = math.sqrt(max(0.,radius*radius-((a+b)/2)**2))
        if min(y1,h) <= max(y0,-h): continue
        tc, th = (y1,0) if y1 <= h else (0,1)
        bc, bh = (y0,0) if y0 >= -h else (0,-1)
        area += (tc-bc)*(b-a)+(th-bh)*(primitive(b)-primitive(a))
    return min(1.,max(0.,area))


def disc_patch(cx, cy):
    x0,y0 = math.floor(cx-28),math.floor(cy-28)
    x1,y1 = math.ceil(cx+28),math.ceil(cy+28)
    yy,xx = np.mgrid[y0:y1,x0:x1]
    far = np.maximum(abs(xx-cx),abs(xx+1-cx))**2 + np.maximum(abs(yy-cy),abs(yy+1-cy))**2
    near = np.maximum(np.maximum(xx-cx,cx-xx-1),0)**2 + np.maximum(np.maximum(yy-cy,cy-yy-1),0)**2
    alpha = (far<=28**2).astype(float)
    for y,x in np.argwhere((near<28**2)&(far>28**2)):
        alpha[y,x] = circle_area(x+x0-cx,x+x0+1-cx,y+y0-cy,y+y0+1-cy)
    assert abs(alpha.sum()-math.pi*28**2) < 1e-7
    return x0,y0,alpha


def stamp(out, patch, cx, cy, fractional=False):
    """Linear-light premultiplied source-over, with an optional translation."""
    h,w = patch.shape[:2]
    if fractional:
        # Pixel index centers at .5 for both continuous geometry and output.
        left,top = cx-w/2,cy-h/2; x0,y0 = math.floor(left),math.floor(top)
        yy,xx = np.indices((h+2,w+2),dtype=float)
        srcy,srcx = yy-(top-y0),xx-(left-x0)
        patch = np.stack([map_coordinates(patch[:,:,c], [srcy,srcx], order=1,
            mode='grid-constant',cval=0,prefilter=False) for c in range(4)],axis=-1)
        h,w = patch.shape[:2]
    else: x0,y0 = int(cx)-w//2,int(cy)-h//2
    assert 0 <= x0 < x0+w <= W and 0 <= y0 < y0+h <= H
    dest = out[y0:y0+h,x0:x0+w]
    dest[:] = patch[:,:,:3]+(1-patch[:,:,3,None])*dest


def base_t5(background):
    base = rgb8(background)
    for x,y in CENTERS:
        base[y-80:y+80,x-80:x-78] = 115; base[y-80:y+80,x+78:x+80] = 115
        base[y-80:y-78,x-80:x+80] = 115; base[y+78:y+80,x-80:x+80] = 115
    base[255:257,379:390] = 170; base[251:262,383:385] = 170
    return base


@lru_cache(maxsize=1)
def mask_pixels(root):
    p,_,_ = paths(root); original = Image.open(p/'inputs/mask.png').convert('RGBA')
    canvas = Image.new('RGBA',original.size,(0,0,0,255)); canvas.alpha_composite(original)
    return np.array(canvas.convert('RGB').resize((160,160),Image.Resampling.NEAREST))


def t5_frames(root, row):
    _,d,_ = paths(root)
    if row['domain'] == 'classic':
        background = np.full((H,W,3),BG); canonical = np.full((128,128,3),int(rgb8(np.array(BG))),np.uint8)
        canonical[:,:28] = int(rgb8(np.array(FG))); canonical[-28:,:] = int(rgb8(np.array(FG)))
        sprites = None; refpath = d/'references/T5-classic.png'
        original = np.array(Image.open(refpath).convert('RGB'))
        assert np.array_equal(original[16:144,16:144],canonical)
    else:
        background = np.load(d/'prepared'/(row['background_id']+'.npy'))
        zero = np.load(d/'prepared'/(row['object_id']+'-T5.npy')); sprites = [np.rot90(zero,-a//90) for a in row['angles_deg']]
        refpath = d/'references'/(row['object_id']+'.png')
    base = base_t5(background); mask = mask_pixels(root); cache={}; schedule=[]
    starts = [row['onset'],row['onset']+60]; stops=[s+30 for s in starts]
    for f in range(FRAMES):
        key = tuple('target' if starts[i]<=f<stops[i] else 'mask' if row['mask']=='persistent' and f>=stops[i] else 'blank' for i in range(2))
        schedule.append(key)
        if key in cache: continue
        frame=base.copy()
        for i,state in enumerate(key):
            cx,cy=row['centers_xy'][i]
            if state=='target':
                if sprites is None: frame[cy-64:cy+64,cx-64:cx+64]=np.rot90(canonical,-row['angles_deg'][i]//90)
                else:
                    b=linear(frame.astype(float)/255); stamp(b,sprites[i],cx,cy); frame=rgb8(b)
            elif state=='mask': frame[cy-80:cy+80,cx-80:cx+80]=mask
        cache[key]=frame
    assert sum(any(a==0 for a in row['angles_deg']) for _ in [0]) == (row['truth']=='YES')
    assert all(sum(k[i]=='target' for k in schedule)==30 for i in (0,1))
    return lambda f:cache[schedule[f]],dict(target_intervals_frames=list(map(list,zip(starts,stops))),
        mask_intervals_frames=[[s,180] if row['mask']=='persistent' else [s,s] for s in stops],
        angles_deg=row['angles_deg'],centers_xy=row['centers_xy'],all_events=2,
        reference_path=str(refpath),reference_sha256=sha(refpath), schedule=schedule,
        no_mask_post_event_restores_background_and_guide=True, persistent_mask_replaces_entire_guide=True)


def t4_frames(root, row):
    _,d,_=paths(root); start=np.array(row['positions_xy_px']); v=np.array(row['velocity_xy_px_s'])
    assert np.max(abs(np.linalg.norm(v,axis=1)-32)) < 1e-10
    # Independent quadratic continuous distance check; source row is not trusted alone.
    minimum=float('inf'); T=179/60
    for i in range(16):
        for j in range(i):
            r=start[i]-start[j]; dv=v[i]-v[j]; t=np.clip(-r@dv/max(dv@dv,1e-15),0,T)
            minimum=min(minimum,float(np.linalg.norm(r+t*dv)-56))
    assert minimum>=4-1e-6
    if row['domain']=='natural':
        bg=np.load(d/'prepared'/(row['background_id']+'.npy'))
        patch=np.load(d/'prepared'/(row['object_id']+'-T4.npy'))
    def frame_at(f):
        points=start+v*(f/60)
        if row['domain']=='classic':
            result=np.full((H,W,3),int(rgb8(np.array(BG))),np.uint8)
            for cx,cy in points:
                x,y,a=disc_patch(cx,cy); q=rgb8(BG+(FG-BG)*a)
                # A transparent bounding-box corner can cover another disk.
                # Only actual analytic coverage writes into the shared frame.
                region=result[y:y+a.shape[0],x:x+a.shape[1]];active=a>0
                region[active]=q[active,None]
            return result
        result=bg.copy()
        for cx,cy in points: stamp(result,patch,cx,cy,True)
        return rgb8(result)
    return frame_at,dict(minimum_continuous_disc_gap_px=minimum,all_motion_frames=[0,179],
        trajectory_sha256=hashlib.sha256((start[None]+np.arange(180)[:,None,None]*v[None]/60).tobytes()).hexdigest(),
        positions_shared_across_all_coherences_LR=True,natural_identity_same_for_all16_elements=True,
        positions_xy_px=start.tolist(),velocity_xy_px_s=v.tolist(),signal_indices=row['signal_indices'])


def render_one(payload):
    root,row=payload; root=Path(root);p,d,out=paths(root);sid=row['sample_id']; receipt=d/'qa'/(sid+'.json')
    cv2.setNumThreads(1); started=time.monotonic()
    if receipt.exists():
        q=read(receipt); assert q['row']==row and sha(q['video'])==q['video_sha256']; return q
    before=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss
    frame_at,science=t4_frames(root,row) if row['task']=='T4' else t5_frames(root,row)
    video=d/'videos'/(sid+'.mp4'); assert not video.exists(),'Partial source preserved for diagnosis'
    hashes=[]; streamhash=hashlib.sha256(); temp=video.with_suffix('.partial.mp4')
    assert not temp.exists(),'Previous failed partial source preserved for diagnosis'
    with av.open(str(temp),'w') as container:
        stream=container.add_stream('libx264rgb',rate=FPS);stream.width=W;stream.height=H;stream.pix_fmt='rgb24'
        stream.codec_context.thread_count=1;stream.options={'crf':'0','preset':'ultrafast','threads':'1'}
        for i in range(FRAMES):
            rgb=np.ascontiguousarray(frame_at(i)); assert rgb.shape==(H,W,3) and rgb.dtype==np.uint8
            b=rgb.tobytes(); hashes.append(hashlib.sha256(b).hexdigest());streamhash.update(b)
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,FPS)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    decodedhash=hashlib.sha256(); actualhashes=[];maxpts=0.;n=0
    with av.open(str(temp)) as container:
        assert len(container.streams.video)==1
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        assert (stream.width,stream.height)==(W,H) and stream.average_rate==60
        pixel_format=stream.codec_context.pix_fmt
        assert pixel_format=='gbrp',pixel_format
        for i,frame in enumerate(container.decode(video=0)):
            assert i<FRAMES
            rgb=frame.to_ndarray(format='rgb24');b=rgb.tobytes(); actualhashes.append(hashlib.sha256(b).hexdigest());decodedhash.update(b)
            assert actualhashes[-1]==hashes[i],(sid,i,'RGB mismatch')
            error=abs(float(frame.pts*frame.time_base)-i/60);maxpts=max(maxpts,error);assert error<=1e-6
            if row['task']=='T5':
                for j,state in enumerate(science['schedule'][i]):
                    if state=='mask':
                        x,y=row['centers_xy'][j];assert np.array_equal(rgb[y-80:y+80,x-80:x+80],mask_pixels(str(root)))
            n+=1
    assert n==FRAMES and decodedhash.hexdigest()==streamhash.hexdigest()
    # OpenCV is the official uint8 reader family used by X, checked in a second decode pass.
    cap=cv2.VideoCapture(str(temp));cvhash=hashlib.sha256();cvframes=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);b=rgb.tobytes();cvframes.append(hashlib.sha256(b).hexdigest());cvhash.update(b)
    cap.release();assert cvframes==hashes
    temp.replace(video)
    q=dict(status='PASS_ALL180_RGB8_PTS_OPENCV',sample_id=sid,task=row['task'],row=row,science=science,
        video=str(video),video_sha256=sha(video),stream_sha256=streamhash.hexdigest(),opencv_stream_sha256=cvhash.hexdigest(),
        decoded_RGB8_frame_hashes=hashes,frames=n,width=W,height=H,fps=FPS,pix_fmt=pixel_format,max_PTS_error_seconds=maxpts,
        codec_threads=1,elapsed_seconds=time.monotonic()-started,
        peak_RSS_MiB=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024,
        peak_RSS_before_MiB=before/1024,implementation_sha256=sha(__file__),model_calls=0)
    write(receipt,q);return q


def ready_item(q):
    row=q['row'];params=dict(row['source_parameters'])
    params.update(domain=row['domain'],paired_block=row['paired_block'],protocol_id=NAME)
    for key in ('background_id','object_id','group','source_asset_hashes'):
        if key in row:params[key]=row[key]
    params['source_geometry_qa']={k:v for k,v in q['science'].items() if k!='schedule'}
    r=dict(sample_id=row['sample_id'],task=row['task'],prompt=row['prompt'],truth=row['truth'],
        choices=row['choices'],accuracy_eligible=row['accuracy_eligible'],source_parameters=params)
    content=[]
    if row['task']=='T5':content.append(dict(type='image',path=q['science']['reference_path'],sha256=q['science']['reference_sha256']))
    content += [dict(type='video',path=q['video'],sha256=q['video_sha256']),dict(type='text',text=row['prompt'])]
    return dict(sample_id=row['sample_id'],task=row['task'],row=r,ready={k:q[k] for k in
        ('video_sha256','stream_sha256','frames','width','height','fps')},content=content,
        video_contract=dict(frames=180,width=W,height=H,fps=60,no_resize=True),
        source_qa_path=str(Path(q['video']).parent.parent/'qa'/(row['sample_id']+'.json')))


def render(root,host,workers,domain,task,limit,offset,sample_ids=None):
    root=Path(root);p,d,out=paths(root);m=read(p/'manifest.json')
    candidates=list(m['items'])
    if domain in ('natural','all') and (p/'natural-manifest.json').exists():
        candidates+=read(p/'natural-manifest.json')['items']
    assert len({x['sample_id'] for x in candidates})==len(candidates),'Duplicate source identity across manifests'
    rows=[x for x in candidates if (domain=='all' or x['domain']==domain) and (task=='all' or x['task']==task)]
    rows=[x for x in rows if x['card'].startswith(host)]
    if sample_ids:
        requested=set(sample_ids);rows=[x for x in rows if x['sample_id'] in requested]
        assert {x['sample_id'] for x in rows}==requested,'Requested ID absent from this current host/domain/task'
    rows=rows[offset:None if limit is None else offset+limit]
    assert rows
    catalog=p/(host+'-sources.json');existing=read(catalog) if catalog.exists() else dict(schema='t45-final-source-catalog-v1',protocol_id=NAME,items={})
    started=time.monotonic();closed=[]
    write(out/(host+'-active.json'),dict(status='RUNNING',host=socket.gethostname(),pid=os.getpid(),workers=workers,domain=domain,task=task,
        selected_ids=[r['sample_id'] for r in rows],started_unix=time.time(),implementation_sha256=sha(__file__)))
    with ProcessPoolExecutor(workers) as pool:
        jobs={pool.submit(render_one,(str(root),r)):r for r in rows}
        for future in as_completed(jobs):
            q=future.result();item=ready_item(q);sid=q['sample_id']
            if sid in existing['items']:assert existing['items'][sid]==item
            else:existing['items'][sid]=item
            write(catalog,existing);closed.append(q)
            progress=dict(completed=len(closed),selected=len(rows),catalog_total=len(existing['items']),
                elapsed_seconds=time.monotonic()-started,peak_worker_RSS_MiB=max(q['peak_RSS_MiB'] for q in closed),last_id=sid)
            write(out/(host+'-progress.json'),progress);print(json.dumps(progress),flush=True)
    final=dict(status='PASS_SELECTED_SHARD',host=host,selected=len(rows),frames_verified=180*len(rows),
        catalog_sha256=sha(catalog),process_pool_closed=True,model_calls=0,worker_count=workers,
        peak_worker_RSS_MiB=max(q['peak_RSS_MiB'] for q in closed),elapsed_seconds=time.monotonic()-started,
        ids=[q['sample_id'] for q in closed],video_logical_bytes=sum(Path(q['video']).stat().st_size for q in closed))
    write(out/(f'{host}-{domain}-{task}-{offset}-{len(rows)}-closure.json'),final)
    write(out/(host+'-active.json'),dict(final,pid=os.getpid(),status='COMPLETE'))
    print(json.dumps(final),flush=True)


def current_paths(root):
    r=Path(root);return (r/'plans'/NAME/'current-reproduction',r/'datasets'/NAME/'current-reproduction',
        r/'runs/stimulus-preparation'/NAME/'current-reproduction')


def initialize_current(root,domain,task='all',host=None,sample_ids=None):
    """Use the published current identities; never silently fall back to v1.

    Frozen sources remain read-only. Reproductions have their own output tree.
    The released metadata and prepared input assets must be staged under root.
    """
    root=Path(root);original_plan=root/'plans'/NAME;original_data=root/'datasets'/NAME
    manifest=original_plan/'current1200-manifest.json'
    if not manifest.is_file():raise FileNotFoundError('Stage the released plans/t45-final-20261030/current1200-manifest.json; historical manifests are not a current-source fallback')
    current=read(manifest);rows=current['items']
    assert len(rows)==len({x['sample_id'] for x in rows})==1200
    assert Counter((x['domain'],x['task']) for x in rows)=={('classic','T4'):408,('classic','T5'):192,('natural','T4'):408,('natural','T5'):192}
    selected=[x for x in rows if (domain=='all' or x['domain']==domain) and (task=='all' or x['task']==task)
        and (host is None or x['card'].startswith(host)) and (not sample_ids or x['sample_id'] in set(sample_ids))]
    assert selected,'No current sources selected'
    needs_t5=any(x['task']=='T5' for x in selected)
    needs_natural=any(x['domain']=='natural' for x in selected)
    needs_natural_t5=any(x['domain']=='natural' and x['task']=='T5' for x in selected)
    p,d,out=current_paths(root)
    for folder in (p,d/'videos',d/'qa',d/'references',out):folder.mkdir(parents=True,exist_ok=True)
    target=p/'manifest.json'
    if target.exists():assert sha(target)==sha(manifest),'Existing current reproduction manifest differs'
    else:shutil.copy2(manifest,target)
    if needs_t5:
        inputs=p/'inputs'
        if not inputs.exists():inputs.symlink_to(original_plan/'inputs',target_is_directory=True)
        assert sha(inputs/'T5-reference.png')=='40592f247066d5ce225e106f06c5fbef6906ea745b697ecd88b91028cf179ce3'
        assert sha(inputs/'mask.png')=='9649d9a37d10a0086971cf42fd6873c8003cc0ecf6e190be1bd408be639fd3c6'
        ref=d/'references/T5-classic.png'
        if ref.exists():assert sha(ref)==sha(inputs/'T5-reference.png')
        else:ref.symlink_to(inputs/'T5-reference.png')
    if needs_natural:
        prepared=original_data/'natural-fit56-v2/prepared'
        if not prepared.is_dir():raise FileNotFoundError('Stage the released natural-fit56-v2 prepared inputs; no v1 object-size fallback is allowed')
        dest=d/'prepared'
        if dest.exists():assert dest.resolve()==prepared.resolve()
        else:dest.symlink_to(prepared,target_is_directory=True)
    if needs_natural_t5:
        references=original_data/'natural-fit56-v2/references'
        for source in references.glob('*.png'):
            dest=d/'references'/source.name
            if dest.exists():assert sha(dest)==sha(source)
            else:dest.symlink_to(source)
    return current


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['initialize','render']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--host',choices=['fourier','radon'],default='fourier');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--domain',choices=['classic','natural','all'],default='classic');parser.add_argument('--task',choices=['T4','T5','all'],default='all')
    parser.add_argument('--limit',type=int);parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--sample-id',action='append',help='Limit CPU reproduction to a named current source; repeat for multiple IDs')
    args = parser.parse_args()
    current=initialize_current(args.root,args.domain,args.task,args.host,args.sample_id)
    if args.action=='initialize':print(json.dumps({'current_sources':len(current['items']),'manifest':'current1200-manifest.json'}))
    else:
        global paths
        paths=current_paths
        render(args.root,args.host,args.workers,args.domain,args.task,args.limit,args.offset,args.sample_id)


if __name__=='__main__':main()
