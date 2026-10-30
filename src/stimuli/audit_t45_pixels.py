"""Read-only pixel measurements of frozen T4/T5 streams; never imports renderer.

Classic T4 uses connected components and measured alpha-mass centroids. T5
classifies observed slot rasters against the reference and original line mask.
Natural checks use independently coded four-neighbor alpha compositing and RGB8
quantization intervals, not an assumption that blending can always be inverted.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import av
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

NAME='t45-final-20261030';V2='natural-fit56-v2';W,H=768,512
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
LOW=linear(np.clip((np.arange(256)-.5)/255,0,1))
HIGH=linear(np.clip((np.arange(256)+.5)/255,0,1))
def compatible(actual,predicted):
    return bool(np.all(predicted>=LOW[actual]-1e-10) and np.all(predicted<=HIGH[actual]+1e-10))
def rgb8(x):return np.rint(255*np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)).astype('uint8')


def classic_t4(rgb,expected):
    assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2])
    gray=rgb[:,:,0];background=89
    count,labels,stats,_=cv2.connectedComponentsWithStats((gray>background).astype('uint8'),8)
    assert count-1==16,('merged/missing components',count-1)
    measured=[];diameters=[];boxes=[]
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        assert x>0 and y>0 and x+w<W and y+h<H,('component touches canvas edge',x,y,w,h)
        weights=np.clip((linear(gray[y:y+h,x:x+w]/255)-linear(89/255))/(linear(234/255)-linear(89/255)),0,1)
        weights*=labels[y:y+h,x:x+w]==label;mass=weights.sum();assert mass>0
        xx=np.arange(w)+x+.5;yy=np.arange(h)+y+.5
        measured.append([(weights*xx[None]).sum()/mass,(weights*yy[:,None]).sum()/mass])
        diameters.append(math.sqrt(4*mass/math.pi));boxes.append([int(x),int(y),int(w),int(h)])
    measured=np.array(measured);_,match=linear_sum_assignment(np.linalg.norm(expected[:,None]-measured[None],axis=2))
    measured=measured[match];diameters=np.array(diameters)[match]
    error=float(np.linalg.norm(measured-expected,axis=1).max());diam_error=float(abs(diameters-56).max())
    assert error<=.03,('centroid error',error)
    assert diam_error<=.08,('alpha-mass diameter error',diam_error)
    return measured,error,diam_error


def translated_four_neighbors(z,fx,fy):
    """Independent forward splat; producer uses inverse-map SciPy sampling."""
    h,w=z.shape[:2];result=np.zeros((h+2,w+2,4))
    result[:h,:w]+=z*((1-fx)*(1-fy));result[:h,1:w+1]+=z*(fx*(1-fy))
    result[1:h+1,:w]+=z*((1-fx)*fy);result[1:h+1,1:w+1]+=z*(fx*fy)
    return result


def natural_t4(rgb,bg,bg8,z,positions):
    allowed=np.zeros((H,W),bool);visible=[]
    for cx,cy in positions:
        left,top=cx-28,cy-28;x0,y0=math.floor(left),math.floor(top)
        patch=translated_four_neighbors(z,left-x0,top-y0)
        expected=patch[:,:,:3]+(1-patch[:,:,3,None])*bg[y0:y0+58,x0:x0+58]
        region=rgb[y0:y0+58,x0:x0+58]
        own=patch[:,:,3]>0
        # Rectangular ROIs can overlap at transparent corners. Another object's
        # pixels there are valid; color identity is checked only on this alpha
        # support, and the separate union check tests all remaining pixels.
        assert compatible(region[own],expected[own]),('natural object color/alpha identity mismatch',cx,cy)
        yy,xx=np.mgrid[y0:y0+58,x0:x0+58]
        # Whole pixel-cell corner support, stricter than center-only support.
        far=np.maximum(abs(xx-cx),abs(xx+1-cx))**2+np.maximum(abs(yy-cy),abs(yy+1-cy))**2
        allowed[y0:y0+58,x0:x0+58]|=far<=28**2+1e-9
        changes=int((np.any(region!=bg8[y0:y0+58,x0:x0+58],axis=2)&own).sum())
        assert changes>0,('no observed object pixels',cx,cy)
        visible.append(changes)
    changed=np.any(rgb!=bg8,axis=2)
    assert not np.any(changed&~allowed),'actual RGB changes outside strict56px supports'
    return min(visible),int(changed.sum())


def mask_pixels(root):
    original=Image.open(root/'plans'/NAME/'inputs/mask.png').convert('RGBA')
    canvas=Image.new('RGBA',original.size,(0,0,0,255));canvas.alpha_composite(original)
    return np.array(canvas.convert('RGB').resize((160,160),Image.Resampling.NEAREST))


def intervals(indices):
    if not indices:return []
    starts=[indices[0]];ends=[]
    for a,b in zip(indices[:-1],indices[1:]):
        if b!=a+1:ends.append(a+1);starts.append(b)
    ends.append(indices[-1]+1);return list(map(list,zip(starts,ends)))


def audit_one(payload):
    root,item=payload;root=Path(root);cv2.setNumThreads(1);started=time.monotonic()
    q=read(item['source_qa_path']);row=q['row'];sid=row['sample_id'];assert item['sample_id']==sid
    video=next(c['path'] for c in item['content'] if c['type']=='video');assert sha(video)==item['ready']['video_sha256']
    streamhash=hashlib.sha256();frames=0;max_centroid=max_diam=0.;tracks=[];min_visible=10**9;states=[[],[]];cache={}
    maxpts=0.;natural=row['domain']=='natural'
    old_container=None;old_hash=hashlib.sha256();delta_counts=[];delta_examples=[]
    if row['task']=='T4':
        positions=np.array(row['positions_xy_px']);velocities=np.array(row['velocity_xy_px_s'])
        if natural:
            d=root/'datasets'/NAME/V2;bg=np.load(d/'prepared'/(row['background_id']+'.npy'));bg8=rgb8(bg)
            z=np.load(d/'prepared'/(row['object_id']+'-T4.npy'))
        elif row.get('parent_v1_source_id'):
            oldq=read(root/'datasets'/NAME/'qa'/(row['parent_v1_source_id']+'.json'))
            old_container=av.open(oldq['video']);old_container.streams.video[0].codec_context.thread_count=1
            old_frames=old_container.decode(video=0)
    else:
        reference=next(c for c in item['content'] if c['type']=='image');assert sha(reference['path'])==reference['sha256']
        ref=np.asarray(Image.open(reference['path']).convert('RGB'));assert ref.shape==(160,160,3)
        mask=mask_pixels(root);centers=row['centers_xy'];outside=np.ones((H,W),bool)
        for x,y in centers:outside[y-80:y+80,x-80:x+80]=False
        if natural:
            d=root/'datasets'/NAME/V2;z=np.load(d/'prepared'/(row['object_id']+'-T5.npy'))
            sprites=[np.rot90(z,-a//90) for a in (0,90,180)]
            expected_ref=z[:,:,:3]+(1-z[:,:,3,None])*.1
            assert compatible(ref[16:144,16:144],expected_ref),'reference not canonical cutout on.1'
        else:templates=[np.rot90(ref[16:144,16:144],-a//90) for a in (0,90,180)]
    with av.open(video) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        for f,frame in enumerate(container.decode(video=0)):
            rgb=frame.to_ndarray(format='rgb24');assert rgb.shape==(H,W,3) and f<180
            streamhash.update(rgb.tobytes());maxpts=max(maxpts,abs(float(frame.pts*frame.time_base)-f/60))
            if old_container is not None:
                old_rgb=next(old_frames).to_ndarray(format='rgb24');old_hash.update(old_rgb.tobytes())
                different=np.any(old_rgb!=rgb,axis=2);delta_counts.append(int(different.sum()))
                if different.any() and not delta_examples:
                    delta_examples=[dict(frame=f,xy=[int(x),int(y)],old_RGB8=old_rgb[y,x].tolist(),current_RGB8=rgb[y,x].tolist()) for y,x in np.argwhere(different)[:6]]
            if row['task']=='T4':
                expected=positions+velocities*f/60
                if natural:
                    minimum,_=natural_t4(rgb,bg,bg8,z,expected);min_visible=min(min_visible,minimum)
                else:
                    measured,ce,de=classic_t4(rgb,expected);tracks.append(measured);max_centroid=max(max_centroid,ce);max_diam=max(max_diam,de)
            else:
                if f==0:baseline=rgb.copy()
                assert np.array_equal(rgb[outside],baseline[outside]),'changed pixels outside scheduled slots'
                for i,(x,y) in enumerate(centers):
                    region=rgb[y-80:y+80,x-80:x+80];h=hashlib.sha256(region.tobytes()).hexdigest();key=(i,h)
                    if key not in cache:
                        if np.array_equal(region,baseline[y-80:y+80,x-80:x+80]):state='blank'
                        elif np.array_equal(region,mask):state='mask'
                        else:
                            crop=region[16:144,16:144];basecrop=linear(baseline[y-64:y+64,x-64:x+64]/255)
                            matches=[]
                            for angle in (0,90,180):
                                if natural:
                                    sprite=sprites[angle//90];expected=sprite[:,:,:3]+(1-sprite[:,:,3,None])*basecrop
                                    match=compatible(crop,expected)
                                else:match=np.array_equal(crop,templates[angle//90])
                                if match:matches.append(angle)
                            assert len(matches)==1,('unrecognized/ambiguous orientation raster',sid,f,i,matches)
                            assert np.array_equal(region[:16],baseline[y-80:y-64,x-80:x+80])
                            assert np.array_equal(region[-16:],baseline[y+64:y+80,x-80:x+80])
                            assert np.array_equal(region[:, :16],baseline[y-80:y+80,x-80:x-64])
                            assert np.array_equal(region[:, -16:],baseline[y-80:y+80,x+64:x+80])
                            state='target'+str(matches[0])
                        cache[key]=state
                    states[i].append(cache[key])
            frames+=1
    assert frames==180 and maxpts<=1e-6 and streamhash.hexdigest()==item['ready']['stream_sha256']
    if old_container is not None:
        assert old_hash.hexdigest()==oldq['stream_sha256'];old_container.close()
    result=dict(status='PASS_INDEPENDENT_PIXEL_AUDIT',sample_id=sid,task=row['task'],domain=row['domain'],frames=frames,
        video_sha256=item['ready']['video_sha256'],stream_sha256=streamhash.hexdigest(),max_PTS_error_seconds=maxpts,
        elapsed_seconds=time.monotonic()-started,implementation_sha256=sha(__file__),model_calls=0)
    if row['task']=='T4':
        if natural:result.update(method='independent forward four-neighbor alpha splat + RGB8 quantization intervals + static-background difference support',
            minimum_changed_RGB8_pixels_per_object_per_frame=min_visible,all180_strict56_pixelcell_support=True,
            no_claim='No assumption that all blends are invertible; color/identity checked against observed quantization intervals')
        else:
            tracks=np.array(tracks);t=np.arange(180)/60;tc=t-t.mean();measured_v=(tc[:,None,None]*tracks).sum(0)/(tc@tc)
            error=float(np.linalg.norm(measured_v-velocities,axis=1).max());speed_error=float(abs(np.linalg.norm(measured_v,axis=1)-32).max())
            assert error<=.03 and speed_error<=.03,(sid,error,speed_error)
            signal=row['signal_indices']
            if signal:
                wanted=-32 if row['truth']=='LEFT' else 32
                assert np.max(abs(measured_v[signal,0]-wanted))<=.03 and np.max(abs(measured_v[signal,1]))<=.03
            result.update(method='16 connected components per frame; decoded linear-light alpha-mass centroids and diameters; OLS velocity across180 independently measured positions',
                maximum_centroid_error_px=max_centroid,maximum_mass_diameter_error_px=max_diam,
                maximum_velocity_vector_error_px_s=error,maximum_speed_error_px_s=speed_error,measured_velocity_xy_px_s=measured_v.tolist(),
                all180_component_counts=16,no_merged_or_edge_touching_components=True,
                alpha_support='Classical analytic circles have radius28; antialiased positive pixel cells can extend beyond the analytic circle')
            result['old_current_pixel_difference']=dict(parent_v1_source_id=row.get('parent_v1_source_id',sid),
                changed_pixel_counts_per_frame=delta_counts if delta_counts else [0]*180,
                total_changed_pixels=sum(delta_counts),maximum_changed_pixels_per_frame=max(delta_counts,default=0),
                frames_changed=sum(x>0 for x in delta_counts),first_examples=delta_examples,
                original_stream_sha256=oldq['stream_sha256'] if old_container is not None else item['ready']['stream_sha256'])
    else:
        observed=[]
        for i in (0,1):
            target=intervals([f for f,s in enumerate(states[i]) if s.startswith('target')]);masks=intervals([f for f,s in enumerate(states[i]) if s=='mask'])
            start=row['onset']+60*i;stop=start+30
            assert target==[[start,stop]],(sid,i,target)
            assert {s for s in states[i] if s.startswith('target')}=={'target'+str(row['angles_deg'][i])}
            assert masks==([[stop,180]] if row['mask']=='persistent' else []),(sid,i,masks)
            assert states[i][stop-1].startswith('target') and states[i][stop]==('mask' if row['mask']=='persistent' else 'blank')
            observed.append(dict(target_intervals=target,mask_intervals=masks,orientation_deg=row['angles_deg'][i],immediate_offset_transition=True))
        assert any('target0' in s for s in states)==(row['truth']=='YES')
        result.update(method='Observed slot raster classification versus supplied reference/canonical quarter-turns, pre-event background+guide, and original opaque line-mask raster',
            independently_observed_events=observed,all180_complete_guide_mask_regions=True,reference_sha256=reference['sha256'],
            target_present=(row['truth']=='YES'),two_events=True,no_mask_restores_baseline=True)
    return result


def select_catalog(root,host,phase):
    p=root/'plans'/NAME
    if phase=='natural':return p/V2/(host+'-sources.json')
    current=p/'classic-disk-union-v2'/(host+'-current-sources.json')
    return current if current.exists() else p/(host+'-sources.json')


def seal_cards(root,host,phase):
    source=select_catalog(root,host,phase);catalog=read(source)['items']
    catalog={sid:x for sid,x in catalog.items() if x['row']['source_parameters']['domain']==phase}
    out=root/'runs/stimulus-preparation'/NAME/'independent-pixel-audit';file=out/(host+'-'+phase+'.jsonl')
    measurements=[]
    for line in file.read_text().splitlines():
        try:measurements.append(json.loads(line))
        except json.JSONDecodeError:pass  # only an in-progress final append may be partial
    passed={x['sample_id']:x for x in measurements if x['status']=='PASS_INDEPENDENT_PIXEL_AUDIT'}
    cards={}
    for sid,item in catalog.items():cards.setdefault(read(item['source_qa_path'])['row']['card'],[]).append(sid)
    sealed=[]
    for card,assigned in sorted(cards.items()):
        ids=sorted(assigned);complete=all(sid in passed for sid in ids)
        print(json.dumps({'card':card,'phase':phase,'passed':sum(sid in passed for sid in ids),'total':len(ids)}),flush=True)
        if not complete:continue
        for sid in ids:
            assert passed[sid]['video_sha256']==catalog[sid]['ready']['video_sha256']
            assert passed[sid]['stream_sha256']==catalog[sid]['ready']['stream_sha256']
            assert passed[sid]['frames']==180
        value=dict(status='PASS_INDEPENDENT_PIXELS_COMPLETE_CARD',card=card,sources=len(ids),frames=len(ids)*180,
            source_catalog=str(source),source_catalog_sha256=sha(source),sample_ids=ids,measurements=[passed[sid] for sid in ids],model_calls=0)
        dest=out/(card+'-'+phase+'-card-pass.json')
        if dest.exists():assert read(dest)==value,'Existing immutable card receipt differs'
        else:dest.write_text(json.dumps(value,indent=2)+'\n')
        sealed.append(str(dest))
    return sealed


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--host',choices=['fourier','radon'],required=True);parser.add_argument('--phase',choices=['classic','natural'],required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--limit',type=int);parser.add_argument('--seal-cards',action='store_true');args=parser.parse_args()
    if args.seal_cards:
        print(json.dumps({'sealed':seal_cards(args.root,args.host,args.phase)}));return
    p=args.root/'plans'/NAME;out=args.root/'runs/stimulus-preparation'/NAME/'independent-pixel-audit';out.mkdir(exist_ok=True)
    if args.phase=='natural':p=p/V2
    source=select_catalog(args.root,args.host,args.phase)
    catalog=read(source)['items']
    items=[x for x in catalog.values() if x['row']['source_parameters']['domain']==args.phase]
    if args.limit is not None:
        t4=[x for x in items if x['task']=='T4'];t5=[x for x in items if x['task']=='T5']
        items=t4[:args.limit//2]+t5[:args.limit-args.limit//2]
    assert items
    stem=f'{args.host}-{args.phase}'+('' if args.limit is None else '-smoke'+str(args.limit))
    dest=out/(stem+'.jsonl');summary=out/(stem+'-summary.json');assert not dest.exists(),'Preserve completed or partial audit'
    started=time.monotonic();results=[]
    with dest.open('w') as file,ProcessPoolExecutor(args.workers) as pool:
        pending={pool.submit(audit_one,(str(args.root),x)):x for x in items}
        for future in as_completed(pending):
            try:result=future.result()
            except Exception as error:result=dict(status='FAILED',sample_id=pending[future]['sample_id'],error=repr(error))
            results.append(result);file.write(json.dumps(result)+'\n');file.flush()
            if len(results)%20==0:print(json.dumps({'independently_measured':len(results),'total':len(items),'failed':sum(r['status']=='FAILED' for r in results)}),flush=True)
    status='PASS_ALL_SELECTED_SOURCE_PIXELS' if all(r['status']!='FAILED' for r in results) else 'HOLD_PIXEL_MEASUREMENT_FAILURE'
    result=dict(status=status,host=args.host,phase=args.phase,sources=len(items),frames_independently_measured=sum(r.get('frames',0) for r in results),
        failures=[r for r in results if r['status']=='FAILED'],by_task=dict(Counter(x['task'] for x in items)),
        per_source_records_sha256=sha(dest),source_catalog_sha256=sha(source),implementation_sha256=sha(__file__),
        process_pool_closed=True,elapsed_seconds=time.monotonic()-started,model_calls=0)
    summary.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if args.limit is None:seal_cards(args.root,args.host,args.phase)
    if status.startswith('HOLD'):raise SystemExit(2)


if __name__=='__main__':main()
