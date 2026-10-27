"""CPU measurements of actual RGB16/FP32 and official uint8 reader outputs.

All nominal T1-T3 cells are represented. Repeated onset/phase offsets are not
counted as independent spatial measurements. No model or processor is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .storage import atomic_json, file_sha256, _append
from .refinement_worker import read, require, write_new

NAME='cosmos-t3-neighbors-20261027'


def linear(s):
    return np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)


def luminance(s):
    return linear(s) if s.ndim==2 else linear(s)@np.array([.2126,.7152,.0722])


def axis(s,center):
    cx,cy=center;y=luminance(s[cy-128:cy+128,cx-128:cx+128])
    bg,fg=float(np.median(y[:16])),float(y.max())
    alpha=np.clip((y-bg)/(fg-bg),0,1);edge=raster.edge_axis(alpha)
    require(edge['status']=='ok','Axis measurement failed')
    return {'pca_deg':raster.pca_axis(alpha),'edge_deg':edge['axis_deg'],
        'edge_rms_px':edge['edge_fit_rms_px']}


def views(codes,native):
    return {'source16':codes.astype(np.float64)/65535.,
        'fp32':(codes.astype(np.float32)*(255./65535.)).astype(np.float64)/255.,
        'uint8':native.astype(np.float64)/255.}


def errors(paths):
    out={};ref=paths['source16']
    for name in ['fp32','uint8']:
        delta=paths[name]-(ref[...,None] if paths[name].ndim==3 else ref)
        out[name+'_srgb_max_abs']=float(np.abs(delta).max())
        out[name+'_srgb_MAE']=float(np.abs(delta).mean())
        out[name+'_linear_Y_max_abs']=float(np.abs(luminance(paths[name])-linear(ref)).max())
    out['fp32_code_mismatches']=int(np.count_nonzero(np.rint(paths['fp32']*65535).astype(np.uint16)!=np.rint(ref*65535).astype(np.uint16)))
    return out


def measure(item,path,io,t1audit):
    start=time.monotonic();row=item['row'];task=item['task']
    source_sha=item['ready']['video_sha256'];require(file_sha256(path)==source_sha,'Source identity')
    media=io.load_file(path);native,meta=media
    require(native.shape==(180,512,768,3) and str(native.dtype)=='uint8' and meta['frames_indices']==list(range(180)), 'Native frame coverage')
    nominal=row.get('source_parameters',row.get('params'));states={};source_lines=[];fp_lines=[]
    frame_count=0;max_pts=0.;pixel=[];p=nominal
    selected=set(range(12)) if task=='T1' else {0,p['onset_frame']}
    if task!='T1':
        event=native[p['onset_frame']];base=native[0]
        expected=list(range(p['onset_frame'],p['offset_frame']))
        require([i for i in range(180) if not np.array_equal(native[i],base)]==expected,'uint8 exposure changed')
        require(all(np.array_equal(native[i],event if i in expected else base) for i in range(180)),'uint8 state changed')
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        require(stream.codec_context.name=='ffv1' and stream.codec_context.format.name=='gbrp16le','Source codec')
        for i,f in enumerate(container.decode(video=0)):
            rgb=f.to_ndarray(format='rgb48le')
            require(i<180 and rgb.shape==(512,768,3),'Source frame count/geometry')
            require(np.array_equal(rgb[...,0],rgb[...,1]) and np.array_equal(rgb[...,0],rgb[...,2]),'Source achromatic expectation')
            codes=rgb[...,0]
            if task=='T1':
                source_lines.append(codes[255,194:574].copy())
                fp_lines.append((codes[255,194:574].astype(np.float32)*(255./65535.)).copy())
            if i in selected:
                vv=views(codes,native[i]);pixel.append(errors(vv))
                if task!='T1':states[i]=vv
            if task!='T1' and i>=p['onset_frame']:
                expected_codes=states[p['onset_frame'] if i<p['offset_frame'] else 0]['source16']
                require(np.array_equal(codes,np.rint(expected_codes*65535).astype(np.uint16)),'RGB16 exposure changed')
            max_pts=max(max_pts,abs(float(f.pts*f.time_base)-i/60));frame_count+=1
    require(frame_count==180 and max_pts<=.001,'Source timing')
    out={'sample_id':item['sample_id'],'task':task,'source_video_sha256':source_sha,'nominal':nominal,
        'frames':180,'frame_indices':list(range(180)),'dimensions_wh':[768,512],
        'source_PTS_max_error_s':max_pts,'pixel_state_frames_measured':sorted(selected),
        'pixel_errors':{k:max(e[k] for e in pixel) for k in pixel[0]},'routes':{}}
    if task=='T1':
        lines={'source16':linear(np.asarray(source_lines,dtype=np.float64)/65535.),
            'fp32':linear(np.asarray(fp_lines,dtype=np.float64)/255.),
            'uint8':luminance(native[:,255,194:574,:].astype(np.float64)/255.)}
        # Reuse the frozen blind estimator; its input conversion alone is bypassed
        # because these three inputs already are measured linear luminance.
        producer=t1audit._producer;original=producer.linear_decode
        producer.linear_decode=lambda x:np.asarray(x,dtype=np.float64)
        try:
            for route,line in lines.items():out['routes'][route]=producer.measure_line(line,row)
        finally:producer.linear_decode=original
        out['cell']=['T1',p['sf_cycles_per_aperture'],p['modulation_depth']]
        out['truth']=row['correct_answer'];out['phase_index']=row['phase_index']
    elif task=='T2':
        yy,xx=np.indices((512,768),dtype=float);xx+=.5;yy+=.5
        for route in ['source16','fp32','uint8']:
            before=luminance(states[0][route]);event=luminance(states[p['onset_frame']][route]);values={}
            for side,(cx,cy) in zip(['LEFT','RIGHT'],p['centers_xy']):
                core=(xx-cx)**2+(yy-cy)**2<(p['disk_diameter_px']*.35)**2
                values[side]={'base_Y':float(before[core].mean()),'event_Y':float(event[core].mean()),
                    'delta_Y':float((event-before)[core].mean())}
            out['routes'][route]=values
        out['cell']=['T2',p['background_Y'],p['delta_Y'],p['offset_frame']-p['onset_frame']]
        out['truth']=row['truth'];out['onset_frame']=p['onset_frame'];out['offset_frame']=p['offset_frame']
    else:
        for route in ['source16','fp32','uint8']:
            out['routes'][route]={side:axis(states[p['onset_frame']][route],center)
                for side,center in zip(['LEFT','RIGHT'],p['centers_xy'])}
        out['cell']=['T3',p['angle_deg']];out['truth']=row['truth']
        out['onset_frame']=p['onset_frame'];out['offset_frame']=p['offset_frame']
    out['cpu_seconds']=time.monotonic()-start
    return out


def run(root,shard,total):
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
    os.nice(15)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[-4+2*(shard%2):-2+2*(shard%2)] if shard%2==0 else cpus[-2:])
    global np,av,raster
    import numpy as np
    import av
    from . import raster
    from .source_audit_t1 import T1Audit
    import torch
    torch.set_num_threads(1)
    from vllm.multimodal.media.video import VideoMediaIO
    from vllm.multimodal.media.image import ImageMediaIO
    root=Path(root);plan_path=root/'plans'/NAME/'pixel-plan.json';plan=read(plan_path)
    require(plan['representative_video_count']==682 and total==4,'Changed CPU scope')
    ids=sorted(plan['items'])[shard::total]
    out=root/'runs/reproducibility'/NAME/('pixel-'+str(shard));out.mkdir(parents=True,exist_ok=False)
    roots={'original':root/'runs/t2_t3_classic_rgb16_production_20261020','t1':root/'runs/t1-rgb16',
        't2-extension':root/'runs/t2-duration-extension','t3-refinement':root/'runs/t3_threshold_refinement_v1',
        'classic-refinement':root/'runs/classic-refinement-20261026'}
    audit=T1Audit(roots['t1']);io=VideoMediaIO(ImageMediaIO(),num_frames=180,video_backend='opencv',backend='opencv',fps=-1)
    require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    atomic_json(out/'identity.json',{'plan_sha256':file_sha256(plan_path),'implementation_sha256':file_sha256(__file__),
        'shard':shard,'total':total,'assigned_ids':ids,'model_calls':0,'processor_calls':0,'source_producer_sha256':audit.producer_sha256,
        'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'versions':{k:importlib.metadata.version(k) for k in ['vllm','av','numpy','torch']},
        'measurement':'actual full180 official reader vs RGB16; frozen image estimators, no processor/model calls'})
    started=time.monotonic()
    for i,sid in enumerate(ids):
        item=plan['items'][sid];ready=item['ready'];relative=ready.get('video_relative',ready.get('video_path'))
        result=measure(item,roots[item['source_family']]/relative,io,audit)
        _append(out/'measurements.jsonl',result)
        atomic_json(out/'progress.json',{'completed':i+1,'expected':len(ids),'last':sid,'elapsed_seconds':time.monotonic()-started})
    require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    result={'status':'COMPLETE_PASS','shard':shard,'ids':ids,'count':len(ids),'model_calls':0,'processor_calls':0,
        'CUDA_initialized':False,'elapsed_seconds':time.monotonic()-started}
    atomic_json(out/'completion.json',result)
    return {'status':'COMPLETE_PASS','count':len(ids),'shard':shard}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--shard',required=True,type=int);p.add_argument('--total',default=4,type=int);a=p.parse_args()
    print(json.dumps(run(a.root,a.shard,a.total)))


if __name__=='__main__':main()
