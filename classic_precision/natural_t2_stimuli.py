"""Finite natural T2 preparation; no model dispatch or answer-based selection.

Absolute alpha-area mean linear luminance increments, original object baselines,
crossed photographic/generated sources, and matched left/right trials.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

NAME = 'natural-t2-qwen-x-20261029'
DELTAS = [.05,.06,.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.19,.20,.25,.30,.35,.40]
ONSETS = list(range(18,134,5))
Y = np.array([.2126,.7152,.0722])
GROUPS = [('photo','photo'),('gpt','gpt'),('photo','gpt'),('gpt','photo')]
CENTERS = [(224,302),(544,302)]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');temp.replace(p)

def read(p):return json.loads(Path(p).read_text())
def linear(v):return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)

def rgb8(v):
    assert np.isfinite(v).all() and v.min()>=-1e-12 and v.max()<=1+1e-12
    v=np.clip(v,0,1) # arithmetic epsilon only, after range rejection
    return np.rint(255*np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)).astype(np.uint8)

def helpers(root):
    src=Path(root)/'src/stimuli';sys.path.insert(0,str(src))
    import natural_signal_reference as ref
    import prepare_next_tasks as prep
    return ref,prep

def asset_job(args):
    root,asset,kind=args;root=Path(root);ref,prep=helpers(root)
    out=root/'datasets'/NAME/('prepared' if kind=='background' else 'prepared-vertical-v2');out.mkdir(parents=True,exist_ok=True)
    p=(root/'assets/library'/asset['path']).resolve();assert sha(p)==asset['sha256'],asset['id']
    target=out/(asset['id']+'.npz');receipt=out/(asset['id']+'.json')
    if receipt.exists():
        r=read(receipt);assert r['source_sha256']==asset['sha256'] and r['prepared_sha256']==sha(target)
        return r
    if kind=='background':
        values,extra=ref.background(p);features=[]
        for cx,cy in CENTERS:
            z=values[cy-128:cy+128,cx-128:cx+128]@Y;features.extend([float(z.mean()),float(z.std())])
        np.savez_compressed(target,rgb=values)
    else:
        rgba=np.asarray(Image.open(p).convert('RGBA'),np.float64)/255;a=rgba[...,3]
        axis,center,ratio=prep.axis(a);yy,xx=np.where(a>1e-6)
        points=asset.get('axis_points_px')
        if points:
            dx,dy=np.subtract(points[1],points[0]);theta=90-float(np.degrees(np.arctan2(dy,dx)))
            axis_rule='existing annotated physical shaft/body axis'
        elif ratio>=4:
            theta=90-axis;axis_rule='elongated silhouette principal axis; placement only, not T3 ground truth'
        else:theta=0.;axis_rule='native pose for non-elongated object'
        t=np.deg2rad(theta);rotation=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        rotated=rotation@np.stack([xx,yy]);low=rotated.min(axis=1);high=rotated.max(axis=1)
        area_scale=np.sqrt(5118.7529296875/a.sum());scale=min(area_scale,300/(high[0]-low[0]+2),380/(high[1]-low[1]+2))
        matrix=np.eye(3);matrix[:2,:2]=scale*rotation;matrix[:2,2]=191.5-scale*(low+high)/2
        source=np.concatenate([linear(rgba[...,:3])*a[...,None],a[...,None]],axis=-1)
        transformed=prep.sample_affine(source,matrix,size=384,samples=8)
        pm,a=transformed[...,:3],transformed[...,3]
        assert max(a[0].max(),a[-1].max(),a[:,0].max(),a[:,-1].max())==0
        mu=float((pm@Y).sum()/a.sum());assert mu+max(DELTAS)<1,('Insufficient absolute increment headroom',asset['id'],mu)
        features=[mu,float(a.sum())]
        extra={'baseline_mean_Y':mu,'alpha_area_px2':float(a.sum()),'source_to_output_matrix':matrix.tolist(),
            'rotation_degrees':theta,'placement_axis_rule':axis_rule,'scale':scale,'area_scale_capped':bool(scale<area_scale),
            'source_axis_eigenvalue_ratio':ratio,'policy':'elongated objects vertical, physical annotation preferred; target alpha area 5118.75; cap width300/height380 to prevent crop/overlap; fixed across conditions',
            'resampling':'single FP64 premultiplied-linear-RGB affine, 8x8 subpixel integration','baseline_rgb_gain':1}
        np.savez_compressed(target,pm=pm,alpha=a)
    r={'asset_id':asset['id'],'kind':kind,'pair_id':asset['pair_id'],'source_sha256':asset['sha256'],
       'prepared_sha256':sha(target),'features':features,**extra}
    write(receipt,r);return r

def prepare_assets(root,jobs):
    root=Path(root);out=root/'plans'/NAME;out.mkdir(parents=True,exist_ok=True)
    source=root/'assets/library/extensions/t2-natural-20261029/catalog.json';pool=read(source)
    spec={'schema':'natural-T2-allocation-criteria-v1','groups':GROUPS,'deltas':DELTAS,'onsets':ONSETS,
        'frames':180,'fps':60,'width':768,'height':512,'exposure_frames':30,'maximum_new_model_calls':3840,
        'seed_rule':'first seed 0..9999 passing predeclared feature balance, never any model answers',
        'feature_balance':'Within each photographic/generated source independently: every delta mean versus pooled mean, standardized by pooled feature SD, absolute value <=0.5; background local mean/SD at both target sites, object mean luminance and alpha area',
        'max_standardized_mean_deviation':.5,'background_reuse_per_semantic_pair':[4,5],
        'object_reuse_per_semantic_pair':[9,10],'within_delta_unique_backgrounds':24,'within_delta_unique_objects':24,
        'LR_pair':'same background, object, onset, delta; only brightened side changes',
        'catalog_sha256':sha(source),'stimulus_implementation_sha256':sha(__file__),
        'original_baselines':True,'geometry_version':'vertical-v2','object_canvas':384,
        'increment':'k=delta/(1-mu); pm_event=pm+k*(alpha-pm); alpha unchanged',
        'declared_side_effects':'texture contrast scales by 1-k, color moves toward white; source appearance and local context are joint transfer factors',
        'encoding':'lossless RGB8 H.264; no YUV/chroma subsampling; every decoded frame must match the rendered uint8 RGB array byte-for-byte',
        'not_authorized':['T3 model calls','classic reruns','automatic scientific retry']}
    cp=out/'allocation-criteria.json'
    if cp.exists():assert read(cp)==spec,'Criteria changed after preparation'
    else:write(cp,spec)
    tasks=[(str(root),a,'background') for a in pool['backgrounds']]+[(str(root),a,'object') for a in pool['objects']]
    rows=[]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i,r in enumerate(ex.map(asset_job,tasks)):
            rows.append(r)
            if (i+1)%25==0:print(json.dumps({'prepared_assets':i+1,'total':len(tasks)}),flush=True)
    write(out/'asset-preparation.json',{'status':'CPU_ASSETS_PREPARED','catalog_sha256':sha(source),'criteria_sha256':sha(cp),'assets':rows})
    return {'assets':len(rows),'model_calls':0}

def allocate(root):
    root=Path(root);out=root/'plans'/NAME;assert not (out/'allocation.json').exists()
    criteria=read(out/'allocation-criteria.json');prep=read(out/'asset-preparation.json');by={x['asset_id']:x for x in prep['assets']}
    objpairs=sorted({x['pair_id'] for x in prep['assets'] if x['kind']=='object'});assert len(objpairs)==50
    pool=read(root/'assets/library/extensions/t2-natural-20261029/catalog.json');lookup={(a['pair_id'],a['id'].split('_')[0]):a['id'] for a in pool['objects']}
    bfeatures=np.array([[by[f'{source}_bg_{i+1:03}']['features'] for source in ['photo','gpt']] for i in range(100)]).reshape(100,-1)
    ofeatures=np.array([[by[lookup[(pair,source)]]['features'] for source in ['photo','gpt']] for pair in objpairs]).reshape(50,-1)
    checks=[]
    for seed in range(10000):
        rng=np.random.default_rng(seed);bp=rng.permutation(100);op=rng.permutation(50)
        idx=np.arange(480);bi=bp[idx%100];oi=op[(idx+11*(idx//100))%50]
        features=np.concatenate([bfeatures[bi],ofeatures[oi]],axis=1);sd=features.std(axis=0);assert (sd>0).all()
        z=(features.reshape(20,24,-1).mean(axis=1)-features.mean(axis=0))/sd
        metric=float(np.max(abs(z)));checks.append({'seed':seed,'max_standardized_mean_deviation':metric})
        if metric<=criteria['max_standardized_mean_deviation']:break
    else:raise RuntimeError('No allocation passes fixed criteria; do not relax silently')
    blocks=[];counts=Counter();rank=rng.permutation(480);cards=['fourier0']*172+['fourier1']*172+['radon0']*136
    cardmap={int(k):cards[i] for i,k in enumerate(rank)}
    for d,delta in enumerate(DELTAS):
        assert len(set(bi[d*24:(d+1)*24]))==len(set(oi[d*24:(d+1)*24]))==24
        time_order=rng.permutation(24)
        for j in range(24):
            k=d*24+j;block={'block':k,'delta_Y':delta,'onset_index':int(time_order[j]),'onset_frame':ONSETS[time_order[j]],
                'offset_frame':ONSETS[time_order[j]]+30,'background_pair':f'bg_{bi[k]+1:03}','object_pair':objpairs[oi[k]],'card':cardmap[k],'cases':[]}
            for bs,osrc in GROUPS:
                for side in ['LEFT','RIGHT']:
                    c={'background_id':f'{bs}_{block["background_pair"]}','object_id':lookup[(block['object_pair'],osrc)],
                       'group':f'{bs}_bg__{osrc}_obj','truth':side,'delta_Y':delta,'onset_frame':block['onset_frame'],'offset_frame':block['offset_frame'],'block':k,
                       'stimulus_recipe_sha256':sha(out/'allocation-criteria.json')}
                    c['sample_id']='nt2_'+digest(c)[:24];block['cases'].append(c);counts[block['card']]+=1
            blocks.append(block)
    allids=[c['sample_id'] for b in blocks for c in b['cases']];assert len(allids)==len(set(allids))==3840
    allocation={'schema':'natural-T2-crossed-four-source-groups-v1','status':'ALLOCATED_NO_MODEL_CALLS','criteria_sha256':sha(out/'allocation-criteria.json'),
        'asset_preparation_sha256':sha(out/'asset-preparation.json'),'selected_seed':seed,'seed_checks':checks,'card_counts':dict(counts),'blocks':blocks,
        'background_semantic_pair_counts':dict(Counter(f'bg_{i+1:03}' for i in bi)),
        'object_semantic_pair_counts':dict(Counter(objpairs[i] for i in oi))}
    write(out/'allocation.json',allocation);return {'seed':seed,'checks':len(checks),'balance':metric,'counts':dict(counts)}

def stamp(bg,pm,a,side):
    cx,cy=CENTERS[0 if side=='LEFT' else 1];half=a.shape[0]//2;s=(slice(cy-half,cy+half),slice(cx-half,cx+half))
    bg[s]=pm+(1-a[...,None])*bg[s]

def encode(path,baseline,event,onset,offset):
    import av
    out=av.open(str(path),'w');stream=out.add_stream('libx264rgb',rate=60)
    stream.width=768;stream.height=512;stream.pix_fmt='rgb24';stream.codec_context.thread_count=1
    stream.options={'crf':'0','preset':'veryfast','x264-params':'keyint=180:min-keyint=180:scenecut=0:bframes=0'}
    for i in range(180):
        frame=av.VideoFrame.from_ndarray(event if onset<=i<offset else baseline,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,60)
        for packet in stream.encode(frame):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
    out.close()

def render_block(args):
    root,k=args;root=Path(root);plan=root/'plans'/NAME;data=root/'datasets'/NAME
    block=read(plan/'allocation.json')['blocks'][k];out=data/'videos';out.mkdir(exist_ok=True);qa=data/'qa';qa.mkdir(exist_ok=True)
    receipt=qa/f'block-{k:03}.json'
    if receipt.exists():
        r=read(receipt);assert all(sha(out/(c['sample_id']+'.mp4'))==c['video_sha256'] for c in r['cases']);return r
    cases=[]
    for c in block['cases']:
        b=np.load(data/'prepared'/(c['background_id']+'.npz'))['rgb'];o=np.load(data/'prepared-vertical-v2'/(c['object_id']+'.npz'));pm,a=o['pm'],o['alpha']
        mu=float((pm@Y).sum()/a.sum());gain=c['delta_Y']/(1-mu);assert 0<gain<1
        b=b.copy();stamp(b,pm,a,'LEFT');stamp(b,pm,a,'RIGHT');e=b.copy()
        cx,cy=CENTERS[0 if c['truth']=='LEFT' else 1]
        e[cy-192:cy+192,cx-192:cx+192] += gain*(a[...,None]-pm)
        delta=float(((e-b)@Y).sum()/a.sum());assert abs(delta-c['delta_Y'])<1e-12
        base,event=rgb8(b),rgb8(e);observed=float(((linear(event/255.)-linear(base/255.))@Y).sum()/a.sum())
        # Keep the exact float64 stimulus. Accept only bounded, measured RGB8
        # quantization error: <0.002 absolute and <2% of the requested increment.
        # No post-quantization fitting, dithering or model-score selection.
        assert abs(observed-c['delta_Y'])<min(.002,.02*c['delta_Y']),(c['sample_id'],observed,c['delta_Y'])
        cx,cy=CENTERS[0 if c['truth']=='LEFT' else 1];changed=np.any(base!=event,axis=-1);roi=np.zeros((512,768),bool);roi[cy-192:cy+192,cx-192:cx+192]=a>0
        assert changed.any() and not changed[~roi].any()
        path=out/(c['sample_id']+'.mp4');temp=path.with_suffix('.partial.mp4');encode(temp,base,event,c['onset_frame'],c['offset_frame'])
        import cv2
        cap=cv2.VideoCapture(str(temp),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert abs(cap.get(cv2.CAP_PROP_FPS)-60)<1e-5;count=0;stream_hash=hashlib.sha256()
        while True:
            ok,frame=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);expected=event if c['onset_frame']<=count<c['offset_frame'] else base
            assert rgb.shape==(512,768,3) and np.array_equal(rgb,expected),('Lossless/source/timing mismatch',c['sample_id'],count)
            stream_hash.update(rgb.tobytes())
            count+=1
        cap.release();assert count==180;temp.replace(path)
        record={**c,'video_sha256':sha(path),'stream_sha256':stream_hash.hexdigest(),'render_implementation_sha256':sha(__file__),'baseline_rgb8_sha256':hashlib.sha256(base.tobytes()).hexdigest(),
            'event_rgb8_sha256':hashlib.sha256(event.tobytes()).hexdigest(),'decoded_frames_exact':count,
            'decoded_delta_Y':observed,'float64_delta_Y':delta,'baseline_object_Y':mu,'white_blend_k':gain,'alpha_area_px2':float(a.sum()),
            'outside_target_identical':True,'video_bytes':path.stat().st_size};cases.append(record)
        if k in (0,24,48,72):
            preview=data/'review';preview.mkdir(exist_ok=True)
            Image.fromarray(base).save(preview/(c['sample_id']+'-before.png'));Image.fromarray(event).save(preview/(c['sample_id']+'-event.png'))
    r={'block':k,'status':'FULL180_LOSSLESS_RGB_QA_PASS','cases':cases};write(receipt,r);return r

def render(root,jobs,start,stop):
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i,r in enumerate(ex.map(render_block,[(root,k) for k in range(start,stop)])):
            print(json.dumps({'completed_blocks_this_process':i+1,'block':r['block'],'videos':len(r['cases'])}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare-assets','allocate','render']);p.add_argument('--root',required=True);p.add_argument('--jobs',type=int,default=2);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=480);a=p.parse_args()
    if a.command=='prepare-assets':r=prepare_assets(a.root,a.jobs)
    elif a.command=='allocate':r=allocate(a.root)
    else:r=render(a.root,a.jobs,a.start,a.stop)
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
