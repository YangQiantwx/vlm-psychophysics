"""CPU reference for natural-context signal definitions, never model dispatch.

Uses the unchanged certified T1 area renderer and original natural assets.
The finite review matrix checks newly introduced compositing/photometric math;
it is not full-video or actual-processor qualification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix

Y = np.array([.2126, .7152, .0722], np.float64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def linear(a):
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def quantize(a):
    if not np.isfinite(a).all() or a.min() < -1e-12 or a.max() > 1 + 1e-12:
        raise ValueError('Out-of-gamut signal: do not silently clip')
    a = np.maximum(a, 0)  # arithmetic epsilon only; upper range is checked
    s = np.where(a <= .0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - .055)
    return np.rint(s * 65535).astype(np.uint16)


def decode(a):
    return linear(a.astype(np.float64) / 65535)


def area_weights(source, dest):
    """Exact overlap of output boxes with piecewise-constant source pixels."""
    edges = np.linspace(0, source, dest + 1)
    ii, jj, ww = [], [], []
    for i in range(dest):
        for j in range(int(np.floor(edges[i])), min(source, int(np.ceil(edges[i+1])))):
            v = max(0., min(edges[i+1], j+1) - max(edges[i], j))
            if v:
                ii.append(i); jj.append(j); ww.append(v / (edges[i+1]-edges[i]))
    weights = csr_matrix((ww, (ii, jj)), shape=(dest, source))
    assert np.max(np.abs(np.asarray(weights.sum(axis=1)).ravel() - 1)) < 1e-12
    return weights


def background(path, width=768, height=512):
    im = np.asarray(Image.open(path).convert('RGB'), np.float64) / 255
    h, w = im.shape[:2]
    if w/h > width/height:
        cw = int(h * width/height); x = (w-cw)//2; crop = (x, 0, x+cw, h)
    else:
        ch = int(w * height/width); y = (h-ch)//2; crop = (0, y, w, y+ch)
    x0,y0,x1,y1 = crop
    im = linear(im[y0:y1,x0:x1])
    wy, wx = area_weights(im.shape[0], height), area_weights(im.shape[1], width)
    out = np.stack([(wx @ (wy @ im[:,:,k]).T).T for k in range(3)], axis=-1)
    return out, {'crop_xyxy': list(crop), 'resize': 'FP64 linear-RGB exact box overlap',
                 'input_dimensions': [w,h], 'output_dimensions': [width,height]}


def contrast(a, roi):
    v = (a @ Y)[roi]
    mu = float(v.mean())
    return {'mean_Y': mu, 'RMS_contrast': float(v.std()/mu),
            'Michelson_extrema': float((v.max()-v.min())/(v.max()+v.min()))}


def controlled_object(root, asset, prep, texture_mode='classic_Y0p5'):
    path = (root/'assets/library'/asset['path']).resolve()
    assert sha(path) == asset['sha256']
    rgba = np.asarray(Image.open(path).convert('RGBA'), np.float64)/255
    a = rgba[:,:,3]
    axis, center, ratio = prep.axis(a)
    scale = np.sqrt(5118.7529296875 / a.sum())
    src = np.concatenate([linear(rgba[:,:,:3])*a[:,:,None], a[:,:,None]], axis=-1)
    transform = prep.matrix(scale, 90-axis, center)
    transformed = prep.sample_affine(src, transform, size=256, samples=8)
    pm, a = transformed[:,:,:3], transformed[:,:,3]
    assert max(a[0].max(), a[-1].max(), a[:,0].max(), a[:,-1].max()) == 0, 'Object crop'
    mu = float((pm @ Y).sum()/a.sum())
    active = a > 1e-8
    q = pm[active] / a[active,None] - mu
    # One constant transform per asset, shared by every delta/side/background.
    if texture_mode=='classic_Y0p5':
        gain = min(1., .09/max(float(q.max()), 1e-12), .49/max(float(-q.min()), 1e-12))
        pm = .5*a[:,:,None] + gain*(pm-mu*a[:,:,None]); target_mean=.5
    elif texture_mode=='native_texture':
        gain=min(1., .59/float((pm[active]/a[active,None]).max()))
        pm=pm*gain; target_mean=mu*gain
    elif texture_mode=='original_baseline':
        gain=1.; target_mean=mu
    else:
        raise ValueError(texture_mode)
    assert abs(float((pm@Y).sum()/a.sum())-target_mean) < 1e-12
    return pm, a, {'asset_id':asset['id'], 'source_sha256':sha(path),
                   'source_axis_ratio':ratio, 'source_to_output_matrix':transform.tolist(),
                   'alpha_area_px2':float(a.sum()), 'area_relative_error':float(a.sum()/5118.7529296875-1),
                   'original_mean_Y':mu, 'fixed_texture_gain':gain, 'target_mean_Y':target_mean,'texture_mode':texture_mode,
                   'channel_range_base':[float((pm[active]/a[active,None]).min()),float((pm[active]/a[active,None]).max())]}


def stamp(base, pm, a, cx, cy):
    out = base.copy(); h,w = a.shape
    x,y = cx-w//2,cy-h//2
    out[y:y+h,x:x+w] = pm + (1-a[:,:,None])*out[y:y+h,x:x+w]
    return out


def run(root, output):
    started=time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    producer=module(root/'runs/t1-rgb16/producer.py','natural_review_t1')
    prep=module(root/'src/stimuli/prepare_next_tasks.py','natural_review_prep')
    catalog=json.loads((root/'assets/library/catalog.json').read_text())
    wanted=['photo_bg_001','gpt_bg_001','photo_bg_002','gpt_bg_002']
    bgs=[]
    for sid in wanted:
        a=next(x for x in catalog['backgrounds'] if x['id']==sid)
        path=(root/'assets/library'/a['path']).resolve(); assert sha(path)==a['sha256']
        b, transform=background(path)
        bgs.append((sid,b,{'asset_id':sid,'source_sha256':sha(path),**transform}))
    yy,xx=np.indices((512,768),dtype=np.float64)
    core=np.hypot(xx+.5-384, yy+.5-256)<180
    allroi=np.ones((512,768),bool)
    w,_=producer.build_basis(0)
    rows=[]; arrays={}; tiles=[]
    for sf in (.5,4.,32.,64.):
        cb,sb=producer.build_basis(sf)
        phase=np.deg2rad(7.5); p=cb*np.cos(phase)-sb*np.sin(phase)
        # Independent sinusoidal regression on decoded differential core pixels.
        k=2*np.pi*sf/448
        design=np.column_stack([np.cos(k*(xx[core]+.5-384)),np.sin(k*(xx[core]+.5-384)),np.ones(core.sum())])
        inverse=np.linalg.pinv(design)
        for sid,b,_ in bgs:
            for opacity in (1.,.5):
                base=(1-opacity*w[:,:,None])*b+opacity*w[:,:,None]*.31
                base_code=quantize(base); base_dec=decode(base_code)
                for m in (.05,.4,1.):
                    im=base+opacity*.31*m*p[:,:,None]
                    code=quantize(im); observed=decode(code)
                    d=(observed-base_dec)@Y
                    fit=inverse@d[core]
                    measured_m=float(np.hypot(fit[0],fit[1])/(opacity*.31*np.sinc(sf/448)))
                    residual=float(np.max(np.abs(d-opacity*.31*m*p)))
                    external=bool(np.array_equal(code[w==0],base_code[w==0]))
                    row={'task':'T1','background':sid,'object':'grating','sf':sf,'nominal':m,'opacity':opacity,
                         'measured':measured_m,'error':measured_m-m,'pixel_residual_Y':residual,'outside_unchanged':external,
                         'core_Michelson':contrast(observed,core)['Michelson_extrema'],
                         'whole_Michelson':contrast(observed,allroi)['Michelson_extrema'],
                         'signal_RMS_over_baseline_mean':float(np.sqrt(np.mean(d[core]**2))/((base_dec@Y)[core].mean())),
                         'passed':abs(measured_m-m)<=1e-4 and residual<=4e-5 and external}
                    rows.append(row)
                    if sid=='photo_bg_001' and sf==4 and m==1:
                        name=f'T1_opacity_{opacity}';arrays[name]=code;tiles.append((name,code))
    objects=[]
    for sid in ('photo_obj_021','photo_obj_027'):
        asset=next(x for x in catalog['objects'] if x['id']==sid)
        pm,a,record=controlled_object(root,asset,prep);objects.append(record)
        for bid,b,_ in bgs:
            baseline=stamp(stamp(b,pm,a,224,302),pm,a,544,302)
            basecode=quantize(baseline); basedec=decode(basecode)
            fulla=np.zeros((512,768),np.float64);fulla[174:430,96:352]=a
            for delta in (.05,.1,.4):
                event=baseline.copy()
                # Do not alpha-compose twice: event replaces the left object's
                # region against its original background, not against itself.
                event[174:430,96:352]=(pm+delta*a[:,:,None])+(1-a[:,:,None])*b[174:430,96:352]
                eventcode=quantize(event);d=(decode(eventcode)-basedec)@Y
                measured=float(np.sum(fulla*d)/np.sum(fulla*fulla))
                residual=float(np.max(np.abs(d-delta*fulla)))
                external=bool(np.array_equal(eventcode[fulla==0],basecode[fulla==0]))
                rows.append({'task':'T2','background':bid,'object':sid,'nominal':delta,'opacity':1.,'measured':measured,
                             'error':measured-delta,'pixel_residual_Y':residual,'outside_unchanged':external,
                             'core_Michelson':None,'whole_Michelson':contrast(decode(eventcode),allroi)['Michelson_extrema'],
                             'passed':abs(measured-delta)<=1e-4 and residual<=4e-5 and external})
                if bid=='photo_bg_001' and delta==.1:
                    arrays[sid+'_baseline']=basecode;arrays[sid+'_delta_0p1']=eventcode
                    tiles.extend([(sid+' baseline',basecode),(sid+' delta .1',eventcode)])
    np.savez_compressed(output/'reference-states.npz',**arrays)
    sheet=Image.new('RGB',(768,3*278),(246,246,242));draw=ImageDraw.Draw(sheet)
    for i,(name,code) in enumerate(tiles):
        x=(i%2)*384;y=(i//2)*278
        thumb=Image.fromarray((code>>8).astype(np.uint8)).resize((384,256),Image.Resampling.LANCZOS)
        sheet.paste(thumb,(x,y+22));draw.text((x+8,y+4),name,fill=(20,20,20))
    sheet.save(output/'natural-signal-review.png')
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    old=json.loads((root/'runs/stimulus-preparation/preparation.json').read_text())
    result={'schema':'natural_signal_reference/v1','status':'COMPONENT_CPU_PASS' if all(r['passed'] for r in rows) else 'FAILED',
            'source_sha256':sha(__file__),'unchanged_T1_renderer_sha256':sha(root/'runs/t1-rgb16/producer.py'),
            'catalog_sha256':sha(root/'assets/library/catalog.json'),'GPU_calls':0,
            'scope':'Finite compositing/photometric matrix, not full-video or actual-processor qualification',
            'backgrounds':[x[2] for x in bgs],'objects':objects,'cases':len(rows),'passed_cases':sum(r['passed'] for r in rows),
            'max_parameter_error':{t:max(abs(r['error']) for r in rows if r['task']==t) for t in ('T1','T2')},
            'max_pixel_residual_Y':max(r['pixel_residual_Y'] for r in rows),'gamut_clipping_applied':False,
            'T3_reused_receipt':{'path':'../preparation.json','sha256':sha(root/'runs/stimulus-preparation/preparation.json'),
                'objects':[{k:x[k] for k in ('asset_id','status')} for x in old['object_angle']['objects']]},
            'T4_reused_receipt':{'clips':len(old['T4']['clips']),'status':old['T4']['status']},
            'T5':{'status':'SCHEDULE_DESIGN_ONLY','RGB16_sequence_and_processor':'NOT_YET_QUALIFIED'},
            'files':['reference-states.npz','measurements.csv','natural-signal-review.png'],
            'elapsed_CPU_wall_seconds':time.monotonic()-started}
    (output/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cases','passed_cases','max_parameter_error','max_pixel_residual_Y','elapsed_CPU_wall_seconds')},ensure_ascii=False))
    if result['status']!='COMPONENT_CPU_PASS':raise SystemExit(1)


def extend_native(root, output, texture_mode='native_texture'):
    """Validate only the changed T2 appearance mode; reuse prior T1 checks."""
    receipt=output/'validation.json'; old=json.loads(receipt.read_text())
    record_key='native_texture_extension' if texture_mode=='native_texture' else 'original_baseline_extension'
    assert old['status']=='COMPONENT_CPU_PASS' and record_key not in old
    started=time.monotonic(); prep=module(root/'src/stimuli/prepare_next_tasks.py','natural_extend_prep')
    cat=json.loads((root/'assets/library/catalog.json').read_text())
    with (output/'measurements.csv').open() as f:rows=list(csv.DictReader(f))
    with np.load(output/'reference-states.npz') as bank:arrays={k:bank[k] for k in bank}
    tiles=[('T1 opaque target',arrays['T1_opacity_1.0']),('T1 half-opacity target',arrays['T1_opacity_0.5'])]
    bgs=[]
    for sid in ('photo_bg_001','gpt_bg_001','photo_bg_002','gpt_bg_002'):
        asset=next(x for x in cat['backgrounds'] if x['id']==sid)
        path=(root/'assets/library'/asset['path']).resolve();assert sha(path)==asset['sha256']
        bgs.append((sid,background(path)[0]))
    new=[];objects=[]
    for sid in ('photo_obj_021','photo_obj_027'):
        asset=next(x for x in cat['objects'] if x['id']==sid)
        pm,a,record=controlled_object(root,asset,prep,texture_mode);objects.append(record)
        for bid,b in bgs:
            base=stamp(stamp(b,pm,a,224,302),pm,a,544,302)
            basecode=quantize(base);basedec=decode(basecode)
            mask=np.zeros((512,768));mask[174:430,96:352]=a
            for delta in (.05,.1,.4):
                if texture_mode=='original_baseline':
                    white_mix=delta/(1-record['target_mean_Y']);assert 0<=white_mix<=1
                    event_pm=pm+white_mix*(a[:,:,None]-pm)
                    expected=np.zeros((512,768));expected[174:430,96:352]=white_mix*(a-pm@Y)
                else:
                    event_pm=pm+delta*a[:,:,None];expected=delta*mask
                event=base.copy();event[174:430,96:352]=event_pm+(1-a[:,:,None])*b[174:430,96:352]
                code=quantize(event);d=(decode(code)-basedec)@Y
                measured=float(d.sum()/mask.sum()) if texture_mode=='original_baseline' else float(np.sum(mask*d)/np.sum(mask*mask))
                residual=float(np.max(np.abs(d-expected)))
                external=bool(np.array_equal(code[mask==0],basecode[mask==0]))
                new.append({'task':'T2_'+texture_mode,'background':bid,'object':sid,'nominal':delta,'opacity':1.,
                            'measured':measured,'error':measured-delta,'pixel_residual_Y':residual,'outside_unchanged':external,
                            'passed':abs(measured-delta)<=1e-4 and residual<=4e-5 and external})
                if bid=='photo_bg_001' and delta==.1:
                    arrays[sid+'_'+texture_mode+'_baseline']=basecode;arrays[sid+'_'+texture_mode+'_delta_0p1']=code
                    tiles.extend([(sid+' '+texture_mode,basecode),(sid+' mean delta .1',code)])
    assert all(x['passed'] for x in new)
    rows.extend(new);columns=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'measurements.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader();wr.writerows(rows)
    np.savez_compressed(output/'reference-states.npz',**arrays)
    sheet=Image.new('RGB',(768,834),(246,246,242));draw=ImageDraw.Draw(sheet)
    for i,(name,code) in enumerate(tiles):
        x=(i%2)*384;y=(i//2)*278
        sheet.paste(Image.fromarray((code>>8).astype(np.uint8)).resize((384,256),Image.Resampling.LANCZOS),(x,y+22))
        draw.text((x+8,y+4),name,fill=(20,20,20))
    sheet.save(output/'natural-signal-review.png')
    old[record_key]={'source_sha256':sha(__file__),'objects':objects,'cases':len(new),
       'passed_cases':len(new),'max_increment_error':max(abs(x['error']) for x in new),
       'max_pixel_residual_Y':max(x['pixel_residual_Y'] for x in new),'elapsed_CPU_wall_seconds':time.monotonic()-started,
       'T1_repeated':False,'original_classic_Y0p5_states_preserved':True,'GPU_calls':0}
    old['current_source_sha256']=sha(__file__);old['total_passed_component_cases']=len(rows)
    receipt.write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(old[record_key],ensure_ascii=False))


def check_t5(output):
    receipt=output/'validation.json';d=json.loads(receipt.read_text());assert 'T5_schedule_validation' not in d
    from itertools import product
    schedules=[]
    scenarios={'PRESENT_FIRST':(0,180),'PRESENT_SECOND':(90,0),'ABSENT_A':(90,180),'ABSENT_B':(180,90)}
    for duration,mask_frames,(scenario,angles),mode in product((10,30,60),(3,6),scenarios.items(),('both','first_only','second_only')):
        onsets=(24,24+duration+30);events=np.full((2,180),-1);mask=np.zeros((2,180),bool)
        for i,start in enumerate(onsets):
            if mode=='both' or mode==('first_only' if i==0 else 'second_only'):events[i,start:start+duration]=angles[i]
            mask[i,start+duration:start+duration+mask_frames]=True
            assert mask[i].sum()==mask_frames and not np.any(mask[i] & (events[i]>=0))
        target_present=bool(np.any(events==0))
        expected=any(angle==0 and (mode=='both' or mode==('first_only' if i==0 else 'second_only')) for i,angle in enumerate(angles))
        assert target_present==expected and max(onsets)+duration+mask_frames<=180
        schedules.append({'duration_frames':duration,'mask_frames':mask_frames,'scenario':scenario,'mode':mode,'answer':'YES' if target_present else 'NO'})
    rng=np.random.default_rng(20261024);mask=np.array([.4,.6]*14450);rng.shuffle(mask);decoded=decode(quantize(mask.reshape(170,170)))
    assert abs(decoded.mean()-.5)<2e-5 and abs(decoded.std()/decoded.mean()-.2)<1e-4
    d['T5_schedule_validation']={'source_sha256':sha(__file__),'cases':len(schedules),'passed_cases':len(schedules),'schedules':schedules,
      'mask_mean_Y':float(decoded.mean()),'mask_RMS_contrast':float(decoded.std()/decoded.mean()),'RGB16_video_and_processor':'PENDING','GPU_calls':0}
    receipt.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'T5_schedules_PASS':len(schedules),'mask_mean_Y':float(decoded.mean()),'mask_RMS_contrast':float(decoded.std()/decoded.mean())}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extend-native-t2',action='store_true')
    p.add_argument('--extend-original-t2',action='store_true');p.add_argument('--check-t5-schedules',action='store_true')
    a=p.parse_args()
    if a.check_t5_schedules:check_t5(a.output.resolve())
    elif a.extend_original_t2:extend_native(a.root.resolve(),a.output.resolve(),'original_baseline')
    elif a.extend_native_t2:extend_native(a.root.resolve(),a.output.resolve())
    else:run(a.root.resolve(),a.output.resolve())
