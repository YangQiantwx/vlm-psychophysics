"""Versioned one-asset repair. Reuse frozen stimuli; never select by model scores.

Only the forty natural-T2 and seventy-two natural-T3 cases that used the
photographic spoon are rebuilt. Other assets, assignments and answers survive.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

from . import natural_t2_stimuli as t2
from . import natural_t13_stimuli as t3

OLD2='natural-t2-qwen-x-20261029'
OLD3='natural-t13-20261029'
NEW2='natural-t2-spoon-20261030'
NEW3='natural-t3-spoon-20261030'
NAME='natural-spoon-replacement-20261030'
ASSET='photo_obj_021'
read,write,sha,digest=t2.read,t2.write,t2.sha,t2.digest

def dirs(root):
    root=Path(root)
    return root/'assets/library/extensions'/NAME,root/'runs/stimulus-preparation'/NAME

def asset(root,raw):
    """Otsu color seed plus GrabCut; retain native photographic RGB samples."""
    import cv2
    root=Path(root);ext,out=dirs(root);ext.mkdir(parents=True,exist_ok=True)
    target=ext/'photograph.jpg'
    if target.exists():assert sha(target)==sha(raw),'Do not replace frozen asset bytes'
    else:shutil.copy2(raw,target)
    rgb=np.array(Image.open(target).convert('RGB'));h,w=rgb.shape[:2]
    signal=np.clip(rgb[:,:,0].astype(float)-rgb[:,:,2],0,255).astype('uint8')
    threshold,mask=cv2.threshold(signal,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    mask=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype('uint8')
    yy,xx=np.where(mask);padding=48
    box=[max(0,int(xx.min())-padding),max(0,int(yy.min())-padding),min(w,int(xx.max())+padding+1),min(h,int(yy.max())+padding+1)]
    x0,y0,x1,y1=box;crop=rgb[y0:y1,x0:x1];mask=mask[y0:y1,x0:x1]
    kernel=np.ones((15,15),np.uint8);g=np.full(mask.shape,cv2.GC_BGD,np.uint8)
    g[cv2.dilate(mask,kernel)>0]=cv2.GC_PR_BGD;g[mask>0]=cv2.GC_PR_FGD
    g[cv2.erode(mask,kernel)>0]=cv2.GC_FGD
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    cv2.grabCut(cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),g,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
    a=np.isin(g,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype('uint8')
    contours,_=cv2.findContours(a,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    a[:]=0;cv2.drawContours(a,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    assert not any(np.any(x) for x in [a[0],a[-1],a[:,0],a[:,-1]])
    rgba=np.dstack([crop,a*255]);Image.fromarray(rgba).save(ext/'spoon.png');Image.fromarray(a*255).save(ext/'mask.png')
    old=next(x for x in read(root/'assets/library/extensions/t2-natural-20261029/catalog.json')['objects'] if x['id']==ASSET)
    row={'id':ASSET,'pair_id':'obj_021','kind':'object','category':'spoon','family':'kitchen','label':'spoon',
        'path':str((ext/'spoon.png').relative_to(root/'assets/library')),'sha256':sha(ext/'spoon.png'),
        'raw_path':str(target.relative_to(root/'assets/library')),'raw_sha256':sha(target),
        'mask_path':str((ext/'mask.png').relative_to(root/'assets/library')),'mask_sha256':sha(ext/'mask.png'),
        'provenance':'photo','author':'Aditya Sethia','source_group':'unsplash_cXog7oetbbw',
        'source_url':'https://unsplash.com/photos/a-wooden-spoon-with-a-long-handle-on-a-white-background-cXog7oetbbw',
        'download_url':'https://images.unsplash.com/photo-1723361749862-22d1362a1c70?fm=jpg&q=95',
        'license':'Unsplash License','license_url':'https://unsplash.com/license','source_publication_date':'2024-08-11',
        'source_dimensions':[w,h],'dimensions':[x1-x0,y1-y0],'alpha_mode':'straight',
        'segmentation':{'method':'OpenCV Otsu R-minus-B largest component; automatic erosion/dilation seeds; GrabCut five iterations; largest external contour filled',
            'otsu_threshold':float(threshold),'seed':0,'iterations':5,'seed_kernel_px':15,'crop_xyxy':box,'crop_padding_px':padding,
            'opencv_version':cv2.__version__,'RGB_modified':False,'manual_source_polygon':False,
            'reference':'https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html'},
        'placement':'existing elongated-silhouette PCA fallback; no manually chosen shaft endpoints',
        'supersedes_source_sha256':old['sha256'],'replacement_reason':'visible red background fringe and cropped handle in prior source',
        'selection_uses_model_answers':False,'implementation_sha256':sha(__file__)}
    out.mkdir(parents=True,exist_ok=True);write(ext/'asset.json',row)
    Image.fromarray(np.where(a[:,:,None],crop,np.array([105,105,105],dtype='uint8'))).save(out/'asset-review.png')
    return row

def link(src,dst):
    src,dst=Path(src),Path(dst);assert src.is_file(),str(src);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert dst.resolve()==src.resolve()
    else:dst.symlink_to(src)

def prepare(root):
    root=Path(root);ext,out=dirs(root);a=read(ext/'asset.json');assert sha(root/'assets/library'/a['path'])==a['sha256']
    t2.NAME=NEW2;t3.NAME=NEW3
    prep2=t2.asset_job((str(root),a,'object'));prep3=t3.object_prepare((str(root),a));assert prep3['accepted']
    old2=read(root/'plans'/OLD2/'allocation.json');old3=read(root/'plans'/OLD3/'allocation.json')
    oldcatalog=read(root/'plans'/OLD3/'catalog.json');newcatalog=copy.deepcopy(oldcatalog)
    newcatalog['objects']=[a if v['id']==ASSET else v for v in newcatalog['objects']]
    newcatalog.update(parent_catalog_sha256=sha(root/'plans'/OLD3/'catalog.json'),revision_reason=a['replacement_reason'])
    write(ext/'catalog.json',newcatalog)
    crit2=read(root/'plans'/OLD2/'allocation-criteria.json');crit3=read(root/'plans'/OLD3/'criteria.json')
    for crit in [crit2,crit3]:crit['one_asset_revision']={'asset_id':ASSET,'new_source_sha256':a['sha256'],'old_source_sha256':a['supersedes_source_sha256'],'assignment_changed':False}
    write(root/'plans'/NEW2/'allocation-criteria.json',crit2);write(root/'plans'/NEW3/'criteria.json',crit3);write(root/'plans'/NEW3/'catalog.json',newcatalog)
    new2=copy.deepcopy(old2);mapping=[]
    for b in new2['blocks']:
        b['cases']=[v for v in b['cases'] if v['object_id']==ASSET]
        for v in b['cases']:
            v['parent_sample_id']=v['sample_id'];v['source_asset_sha256']=a['sha256'];v['stimulus_recipe_sha256']=sha(root/'plans'/NEW2/'allocation-criteria.json')
            v['sample_id']='nt2_'+digest({'parent':v['parent_sample_id'],'asset_sha256':a['sha256'],'recipe_sha256':v['stimulus_recipe_sha256']})[:24]
            mapping.append({'task':'T2','parent_sample_id':v['parent_sample_id'],'sample_id':v['sample_id'],'card':b['card']})
            for suffix in ['.npz','.json']:link(root/'datasets'/OLD2/'prepared'/(v['background_id']+suffix),root/'datasets'/NEW2/'prepared'/(v['background_id']+suffix))
    new2.update(parent_allocation_sha256=sha(root/'plans'/OLD2/'allocation.json'),criteria_sha256=sha(root/'plans'/NEW2/'allocation-criteria.json'),revision=True)
    new3=copy.deepcopy(old3);new3['blocks']=[b for b in new3['blocks'] if b.get('object_id')==ASSET]
    for b in new3['blocks']:
        b['parent_block_id']=b['block_id'];b['source_asset_sha256']=a['sha256'];b['block_id']=digest({'parent':b['parent_block_id'],'asset':a['sha256']})[:20]
        for v in b['cases']:
            v['parent_sample_id']=v['sample_id'];v['sample_id']='n'+digest({'parent':v['parent_sample_id'],'asset':a['sha256'],'criteria':sha(root/'plans'/NEW3/'criteria.json')})[:24]
            mapping.append({'task':'T3','parent_sample_id':v['parent_sample_id'],'sample_id':v['sample_id']})
        for suffix in ['.npy','.json']:link(root/'datasets'/OLD3/'backgrounds'/(b['background_id']+suffix),root/'datasets'/NEW3/'backgrounds'/(b['background_id']+suffix))
    new3.update(parent_allocation_sha256=sha(root/'plans'/OLD3/'allocation.json'),criteria_sha256=sha(root/'plans'/NEW3/'criteria.json'),catalog_sha256=sha(root/'plans'/NEW3/'catalog.json'),revision=True)
    assert Counter(x['task'] for x in mapping)=={'T2':40,'T3':72}
    write(root/'plans'/NEW2/'allocation.json',new2);write(root/'plans'/NEW3/'allocation.json',new3)
    for name in [NEW2,NEW3]:
        (root/'datasets'/name/'qa').mkdir(exist_ok=True);(root/'datasets'/name/'review').mkdir(exist_ok=True)
    # Recompute original feature-balance gates with the same assignments. No re-randomization.
    oldprep=read(root/'plans'/OLD2/'asset-preparation.json')['assets'];lookup={x['asset_id']:x for x in oldprep};lookup[ASSET]=prep2
    objpairs=sorted({x['pair_id'] for x in oldprep if x['kind']=='object'});cat=read(root/'assets/library/extensions/t2-natural-20261029/catalog.json')
    bypair={(x['pair_id'],x['id'].split('_')[0]):x['id'] for x in cat['objects']}
    features=[]
    for b in old2['blocks']:
        f=[]
        for source in ['photo','gpt']:f+=lookup[source+'_'+b['background_pair']]['features']
        for source in ['photo','gpt']:f+=lookup[bypair[b['object_pair'],source]]['features']
        features.append(f)
    f=np.array(features);z=(f.reshape(20,24,-1).mean(1)-f.mean(0))/f.std(0);balance2=float(abs(z).max())
    elig=read(root/'plans'/OLD3/'object-eligibility.json');objects={x['asset_id']:x for x in elig['objects']};objects[ASSET]=prep3
    pairs=elig['eligible_pairs'];pool=np.array([objects['photo_'+x]['features']+objects['gpt_'+x]['features'] for x in pairs]);standard=(pool-pool.mean(0))/np.maximum(pool.std(0),1e-10)
    lookup3={p:i for i,p in enumerate(pairs)};means=[]
    for cell in range(29):
        bs=[b for b in old3['blocks'] if b['task']=='T3' and b['cell']==cell and b['group']=='photo_photo']
        means.append(standard[[lookup3[b['object_id'].split('_',1)[1]] for b in bs]].mean(0))
    balance3=float(abs(np.array(means)).max())
    balance={'status':'PASS' if max(balance2,balance3)<=.5 else 'FAIL','T2_max_mean_shift_SD':balance2,'T3_object_max_mean_shift_SD':balance3,
        'gate':.5,'categorical_assignment_unchanged':True,'all_onsets_signs_LR_backgrounds_unchanged':True,'seed_unchanged':True,
        'T2_asset_preparation':prep2,'T3_asset_preparation':prep3,'counts':{'T2':40,'T3':72},'scope':'Original predeclared feature gates rechecked; not a claim of zero latent confounding.'}
    write(out/'balance-audit.json',balance);write(out/'replacement-map.json',mapping)
    assert balance['status']=='PASS',balance
    return balance

def render_job(job):
    task,root,x=job;t2.NAME=NEW2;t3.NAME=NEW3
    return t2.render_block((root,x)) if task=='T2' else t3.render_one((root,x))

def render(root,jobs):
    root=Path(root);_,out=dirs(root);assert read(out/'balance-audit.json')['status']=='PASS'
    a2=read(root/'plans'/NEW2/'allocation.json');a3=read(root/'plans'/NEW3/'allocation.json')
    tasks=[('T2',str(root),i) for i,b in enumerate(a2['blocks']) if b['cases']]+[('T3',str(root),dict(b,review=(i==0))) for i,b in enumerate(a3['blocks'])]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i,_ in enumerate(ex.map(render_job,tasks)):print(json.dumps({'closed_source_blocks':i+1,'total':len(tasks)}),flush=True)
    return finish(root)

def finish(root):
    from .natural_t13_review import cv_axis
    root=Path(root);_,out=dirs(root);records={'T2':{},'T3':{}};geometry=[]
    a2=read(root/'plans'/NEW2/'allocation.json');a3=read(root/'plans'/NEW3/'allocation.json')
    for b in a2['blocks']:
        if not b['cases']:continue
        q=read(root/'datasets'/NEW2/'qa'/f'block-{b["block"]:03}.json');assert len(q['cases'])==4
        for x in q['cases']:
            assert x['decoded_frames_exact']==180 and x['outside_target_identical']
            assert abs(x['decoded_delta_Y']-x['delta_Y'])<min(.002,.02*x['delta_Y'])
            assert sha(root/'datasets'/NEW2/'videos'/(x['sample_id']+'.mp4'))==x['video_sha256']
            records['T2'][x['sample_id']]={**x,'card':b['card']}
    for b in a3['blocks']:
        q=read(root/'datasets'/NEW3/'qa'/(b['block_id']+'.json'));assert len(q['cases'])==2;axes=[]
        for p in q['patches']:
            tag=f'{ASSET}_{p["theta"]:+g}';z=np.load(root/'datasets'/NEW3/'patches'/(tag+'.npy'));measured=cv_axis(z[:,:,3]);error=t3.angle_diff(measured,90+p['theta']);assert abs(error)<=.01
            geometry.append({'tag':tag,'axis_error_deg':error,'alpha_area':float(z[:,:,3].sum())});axes.append(measured)
        separation=abs(t3.angle_diff(axes[1],axes[0]));assert abs(separation-b['angle'])<=.02
        assert abs(t3.angle_diff(axes[0],90))<abs(t3.angle_diff(axes[1],90))
        for x in q['cases']:
            assert x['frames_byte_equal']==x['timestamps_checked']==180
            assert sha(root/'datasets'/NEW3/'videos/T3'/(x['sample_id']+'.mp4'))==x['video_sha256']
            records['T3'][x['sample_id']]={**b,**x,'actual_angle_deg':separation}
    assert [len(records[t]) for t in ['T2','T3']]==[40,72]
    mapping=read(out/'replacement-map.json');assert {x['sample_id'] for x in mapping}==set(records['T2'])|set(records['T3'])
    assert all(x['parent_sample_id']!=x['sample_id'] for x in mapping)
    write(out/'source-records.json',records);write(out/'geometry-audit.json',geometry)
    oldindex=read(root/'plans'/OLD3/'source-index.json')
    replacements={x['parent_sample_id']:x for x in records['T3'].values()};canonical=[]
    for row in oldindex['rows']:
        if row['task']!='T3':continue
        if row['sample_id'] in replacements:
            new=replacements[row['sample_id']]
            canonical.append({**row,**{k:v for k,v in new.items() if k not in ('cases','patches')},
                'video_root_relative':f'datasets/{NEW3}/videos/T3/{new["sample_id"]}.mp4',
                'video_relative':f'datasets/{NEW3}/videos/T3/{new["sample_id"]}.mp4',
                'qa_sha256':sha(root/'datasets'/NEW3/'qa'/(new['block_id']+'.json')),'source_dataset':NEW3})
        else:canonical.append({**row,'video_root_relative':row['video_relative'],'source_dataset':OLD3})
    assert len(canonical)==len({x['sample_id'] for x in canonical})==2736
    write(root/'plans'/NEW3/'canonical-source-index.json',{'status':'CANONICAL_T3_SOURCE_REVISION','count':2736,
        'replaced':72,'original_source_index_sha256':sha(root/'plans'/OLD3/'source-index.json'),'rows':canonical})
    result={'status':'COMPLETE_SPOON_SOURCE_REVISION','counts':{'T2':40,'T3':72},'decoded_frames':112*180,
        'model_calls':0,'source_records_sha256':sha(out/'source-records.json'),'replacement_map_sha256':sha(out/'replacement-map.json'),
        'balance_audit_sha256':sha(out/'balance-audit.json'),'old_sources_and_results_preserved':True,
        'T2_max_relative_delta_error':max(abs(x['decoded_delta_Y']/x['delta_Y']-1) for x in records['T2'].values()),
        'T3_max_axis_error_deg':max(abs(x['axis_error_deg']) for x in geometry),
        'source_T2_generator_sha256':sha(t2.__file__),'source_T3_generator_sha256':sha(t3.__file__),
        'revision_wrapper_sha256':sha(__file__)}
    write(out/'source-audit.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['asset','prepare','render','finish']);p.add_argument('--root',required=True);p.add_argument('--raw');p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    r=asset(a.root,a.raw) if a.command=='asset' else prepare(a.root) if a.command=='prepare' else render(a.root,a.jobs) if a.command=='render' else finish(a.root)
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
