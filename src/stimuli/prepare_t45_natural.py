"""CPU-only natural asset preparation, fixed allocation and covariate audit.

T4 covers all200 backgrounds and all100 objects, including the repaired spoon.
T4 LR pairs share assets; across coherences assets may differ. T5's eight
occurrence/mask cases share one background/object at each timing-position block.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

import prepare_t45_final as base

read,write,sha,digest=base.read,base.write,base.sha,base.digest
ORIGINAL_PATHS=base.paths
V2='natural-fit56-v2'
CLASSIC_V2='classic-disk-union-v2'
ORIGINAL_T4_FRAMES=base.t4_frames


def sample(source,matrix,size):
    inv=np.linalg.inv(matrix);yy,xx=np.indices((size,size),dtype=float);out=np.zeros((size,size,4))
    for dy in (np.arange(8)+.5)/8-.5:
        for dx in (np.arange(8)+.5)/8-.5:
            sx=inv[0,0]*(xx+dx)+inv[0,1]*(yy+dy)+inv[0,2]
            sy=inv[1,0]*(xx+dx)+inv[1,1]*(yy+dy)+inv[1,2]
            for c in range(4):out[:,:,c]+=map_coordinates(source[:,:,c],[sy,sx],order=1,mode='constant',cval=0,prefilter=False)
    return out/64


def prepare_object(payload):
    root,asset=payload[:2];fit_radius=payload[2] if len(payload)>2 else 26
    root=Path(root);p,d,out=base.paths(root)
    dest=d/'prepared'/(asset['id']+'.json')
    if dest.exists():
        old=read(dest);assert old['source_sha256']==asset['sha256'];return old
    path=d/'inputs/library'/asset['path'];assert sha(path)==asset['sha256']
    rgba=np.asarray(Image.open(path).convert('RGBA'),float)/255;a=rgba[:,:,3]
    # Every positive alpha sample is included; no threshold-based trimming.
    yy,xx=np.where(a>0);center=np.array([(xx.min()+xx.max())/2,(yy.min()+yy.max())/2])
    source=np.dstack([base.linear(rgba[:,:,:3])*a[:,:,None],a])
    dy,dx=np.indices(a.shape,dtype=float);mass=a.sum();cx=(a*dx).sum()/mass;cy=(a*dy).sum()/mass
    cov=np.array([[(a*(dx-cx)**2).sum(),(a*(dx-cx)*(dy-cy)).sum()],[(a*(dx-cx)*(dy-cy)).sum(),(a*(dy-cy)**2).sum()]])/mass
    eig=np.linalg.eigvalsh(cov);ratio=float(eig[-1]/max(eig[0],1e-12))
    radius=float(np.hypot(abs(xx-center[0])+.5,abs(yy-center[1])+.5).max())
    scale=fit_radius/radius;m=np.eye(3);m[:2,:2]*=scale;m[:2,2]=27.5-scale*center
    z=sample(source,m,56);ay,ax=np.where(z[:,:,3]>0)
    support=float(np.hypot(abs(ax-27.5)+.5,abs(ay-27.5)+.5).max())
    assert support<=27.999999,(asset['id'],support)
    assert not any(np.any(t) for t in (z[0],z[-1],z[:,0],z[:,-1]))
    np.save(d/'prepared'/(asset['id']+'-T4.npy'),z)
    alpha=z[:,:,3];mu=float((z[:,:,:3]@base.Y).sum()/alpha.sum())
    luminance=np.divide(z[:,:,:3]@base.Y,alpha,out=np.zeros(alpha.shape),where=alpha>0)
    std=float(np.sqrt((((luminance-mu)**2)*alpha).sum()/alpha.sum()))
    features=[float(alpha.sum()),mu,std,math.log(ratio),float(a.sum()/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)))]
    record=dict(asset_id=asset['id'],pair_id=asset['pair_id'],source_sha256=asset['sha256'],family=asset.get('family','unknown'),
        source_path=asset['path'],features=features,T4=dict(scale=scale,source_to_output_matrix=m.tolist(),
            source_fit_radius_px=fit_radius,
            prepared_alpha_area_px2=float(alpha.sum()),prepared_positive_alpha_enclosing_radius_px=support,
            trajectory_enclosing_circle_diameter_px=56,translation='bilinear premultiplied-linear fractional placement; positive-alpha support remains within28px circle',
            native_RGB_gain=1,source_orientation_retained=True,prepared_sha256=sha(d/'prepared'/(asset['id']+'-T4.npy'))),T5_eligible='T5' in asset.get('tasks',[]))
    if record['T5_eligible']:
        # Native canonical pose is copied into reference and every trial. Only
        # exact quarter-turns follow this single source affine.
        scale=124/max(xx.max()-xx.min()+2,yy.max()-yy.min()+2);m=np.eye(3);m[:2,:2]*=scale;m[:2,2]=63.5-scale*center
        z=sample(source,m,128);assert not any(np.any(t) for t in (z[0],z[-1],z[:,0],z[:,-1]))
        np.save(d/'prepared'/(asset['id']+'-T5.npy'),z)
        ref=np.full((160,160,3),base.BG);ref[16:144,16:144]=z[:,:,:3]+(1-z[:,:,3,None])*base.BG
        ref=base.rgb8(ref);refpath=d/'references'/(asset['id']+'.png');Image.fromarray(ref).save(refpath)
        rotations=[base.rgb8(q[:,:,:3]+(1-q[:,:,3,None])*base.BG) for q in [np.rot90(z,-a//90) for a in (0,90,180)]]
        assert len({x.tobytes() for x in rotations})==3,(asset['id'],'indistinguishable raster rotations')
        record['T5']=dict(scale=scale,source_to_output_matrix=m.tolist(),canonical_rotation_deg=0,canonical_rule='native source pose',
            reference_sha256=sha(refpath),reference_wh=[160,160],object_canvas_wh=[128,128],
            prepared_sha256=sha(d/'prepared'/(asset['id']+'-T5.npy')),quarter_turns_exact=True)
    write(dest,record);return record


def prepare(root,workers):
    root=Path(root);p,d,out=base.paths(root);cat=read(p/'asset-catalog.json')
    for q in (d/'prepared',d/'references'):q.mkdir(parents=True,exist_ok=True)
    bg=[]
    for asset in cat['backgrounds']:
        src=root/'datasets/natural-t13-20261029/backgrounds'/(asset['id']+'.npy')
        receipt=read(src.with_suffix('.json'));assert receipt['source_sha256']==asset['sha256'] and sha(src)==receipt['prepared_sha256']
        target=d/'prepared'/(asset['id']+'.npy')
        if target.exists():assert target.resolve()==src.resolve()
        else:target.symlink_to(src)
        bg.append(dict(asset_id=asset['id'],family=asset.get('family','unknown'),source_sha256=asset['sha256'],
            prepared_sha256=sha(src),features=receipt['T3'],preparation=receipt.get('reused_preparation')))
    write(p/'background-features.json',bg)
    with ProcessPoolExecutor(workers) as pool:
        records=[]
        for i,r in enumerate(pool.map(prepare_object,[(str(root),a) for a in cat['objects']])):
            records.append(r)
            if (i+1)%10==0:print(json.dumps({'prepared_objects':i+1,'total':100}),flush=True)
    write(p/'object-features.json',records)
    write(out/'natural-preparation.json',dict(status='PASS_PREPARED_200BG_100OBJ',backgrounds=len(bg),objects=len(records),
        latest_wood_spoon_sha256=next(a['sha256'] for a in cat['objects'] if a['id']=='photo_obj_021'),
        implementation_sha256=sha(__file__),model_calls=0))


def allocation_search(features,families,sizes,seed,steps=300000):
    """Frequency-preserving pre-outcome swaps; fixed standardized mean gate .5."""
    rng=np.random.default_rng(seed);n=len(features);f0=np.array(features,float)
    z=(f0-f0.mean(0))/np.maximum(f0.std(0),1e-10);names=sorted(set(families))
    cats=np.array([[float(f==g) for g in names] for f in families]);prop=cats.mean(0);f=np.c_[z,cats]
    off=np.r_[0,np.cumsum(sizes)];cell=np.repeat(np.arange(len(sizes)),sizes);a=np.resize(rng.permutation(n),off[-1])
    groups=[set(a[off[i]:off[i+1]]) for i in range(len(sizes))];assert all(len(g)==s for g,s in zip(groups,sizes))
    sums=np.array([f[a[off[i]:off[i+1]]].sum(0) for i in range(len(sizes))])
    def cost(s,size):
        q=s[:z.shape[1]]/size;r=s[z.shape[1]:]-size*prop
        return 100*np.maximum(abs(q)-.45,0)@np.maximum(abs(q)-.45,0)+.02*(q@q)+5*np.sum(np.maximum(abs(r)-1.8,0)**2)
    costs=np.array([cost(s,size) for s,size in zip(sums,sizes)])
    for step in range(steps):
        i,j=rng.integers(0,len(a),2);ci,cj=cell[i],cell[j]
        if ci==cj or a[j] in groups[ci] or a[i] in groups[cj]:continue
        delta=f[a[j]]-f[a[i]];si=sums[ci]+delta;sj=sums[cj]-delta
        ni,nj=cost(si,sizes[ci]),cost(sj,sizes[cj]);change=ni+nj-costs[ci]-costs[cj]
        if change<0 or rng.random()<math.exp(-min(700,change/max(.001*(1-step/steps),1e-10))):
            groups[ci].remove(a[i]);groups[cj].remove(a[j]);groups[ci].add(a[j]);groups[cj].add(a[i]);a[i],a[j]=a[j],a[i]
            sums[ci],sums[cj],costs[ci],costs[cj]=si,sj,ni,nj
        if step%1000==0:
            shift=sums[:,:z.shape[1]]/np.array(sizes)[:,None];dev=sums[:,z.shape[1]:]-np.array(sizes)[:,None]*prop
            if abs(shift).max()<=.5 and abs(dev).max()<=2:break
    shift=sums[:,:z.shape[1]]/np.array(sizes)[:,None];dev=sums[:,z.shape[1]:]-np.array(sizes)[:,None]*prop
    diag=dict(seed=seed,steps=step+1,max_mean_shift_SD=float(abs(shift).max()),max_family_count_deviation=float(abs(dev).max()),
        cell_feature_mean_shift_SD=shift.tolist(),family_names=names,cell_family_counts=sums[:,z.shape[1]:].tolist(),
        frequencies=dict(Counter(map(int,a))),passed=bool(abs(shift).max()<=.5 and abs(dev).max()<=2))
    return [a[off[i]:off[i+1]].tolist() for i in range(len(sizes))],diag


def allocate(root):
    root=Path(root);p,d,out=base.paths(root);classic=read(p/'manifest.json');cat=read(p/'asset-catalog.json')
    bf={r['asset_id']:r for r in read(p/'background-features.json')};of={r['asset_id']:r for r in read(p/'object-features.json')}
    assets={a['id']:a for a in cat['backgrounds']+cat['objects']};cells={};blocks={}
    for row in classic['items']:
        if row['task']=='T4':
            key=('T4',row['coherence'],row['seed']); group=(int(row['coherence']*8)+(row['seed']-101))%4
            cell=('T4',row['coherence'],group)
        else:
            key=('T5',row['k'],row['position_pair']);group=(row['k']+2*row['position_pair'])%4;cell=('T5',0,group)
        blocks[key]=dict(group=group,cell=cell)
        cells.setdefault(cell,set()).add(key)
    cells={key:sorted(value) for key,value in cells.items()};assert len(cells)==40 and all(len(v)==6 for v in cells.values())
    assigned={k:{} for k in blocks};audits={}
    for kind,table in (('background',bf),('object',of)):
        for source in ('photo','gpt'):
            for task in (('T4','T5') if kind=='object' else ('both',)):
                selected_cells=sorted(c for c in cells if (task=='both' or c[0]==task) and base.GROUPS[c[2]][0 if kind=='background' else 1]==source)
                ids=sorted(a['id'] for a in cat['backgrounds' if kind=='background' else 'objects'] if a['id'].startswith(source+'_')
                    and (task!='T5' or of[a['id']]['T5_eligible']))
                features=[table[sid]['features'] for sid in ids];families=[assets[sid].get('family','unknown') for sid in ids]
                for seed in range(20):
                    selections,diag=allocation_search(features,families,[6]*len(selected_cells),seed)
                    if diag['passed']:break
                assert diag['passed'],(kind,source,task,diag)
                for cell,selection in zip(selected_cells,selections):
                    for key,index in zip(cells[cell],selection):assigned[key][kind+'_id']=ids[index]
                audits[f'{kind}_{source}_{task}']={**diag,'asset_ids':ids,'cells':[list(c) for c in selected_cells]}
                print(json.dumps({'allocated':f'{kind}_{source}_{task}','gate':diag['max_mean_shift_SD'],'steps':diag['steps']}),flush=True)
    rows=[]
    for old in classic['items']:
        key=('T4',old['coherence'],old['seed']) if old['task']=='T4' else ('T5',old['k'],old['position_pair'])
        row=dict(old);row['domain']='natural';row['parent_classic_id']=old['sample_id'];row['sample_id']='n45_'+digest([base.NAME,old['sample_id']])[:24]
        row.update(assigned[key]);row['group']='_'.join(base.GROUPS[blocks[key]['group']]);row['source_parameters']=dict(old['source_parameters'])
        row['source_asset_hashes']={k:assets[row[k]]['sha256'] for k in ('background_id','object_id')}
        if row['task']=='T4':row['source_parameters']['object_preparation']=of[row['object_id']]['T4']
        else:row['source_parameters']['object_preparation']=of[row['object_id']]['T5']
        rows.append(row)
    assert len(rows)==600 and Counter(r['group'] for r in rows)=={ '_'.join(g):150 for g in base.GROUPS}
    assert Counter(r['card'] for r in rows)=={'fourier0':232,'fourier1':232,'radon0':136}
    # Exact identity coverage is a deliverable, separate from frequency counts.
    used_bg={r['background_id'] for r in rows};used_obj={r['object_id'] for r in rows}
    assert len(used_bg)==200 and len(used_obj)==100
    rotation_audit=[]
    for sid in sorted({r['object_id'] for r in rows if r['task']=='T5'}):
        z=np.load(d/'prepared'/(sid+'-T5.npy'));images=[base.rgb8(q[:,:,:3]+(1-q[:,:,3,None])*base.BG) for q in (np.rot90(z,-a//90) for a in (0,90,180))]
        pairs=[]
        for i,a in enumerate((0,90,180)):
            for j,b in enumerate((0,90,180)[:i]):
                delta=images[i].astype(float)-images[j].astype(float)
                different=int(np.any(delta!=0,axis=2).sum());assert different>0,(sid,a,b,'exact rotational symmetry')
                pairs.append(dict(angles_deg=[b,a],different_RGB8_pixels=different,
                    mean_absolute_RGB8_channel_difference=float(abs(delta).mean()),RMS_RGB8_channel_difference=float(np.sqrt((delta**2).mean()))))
        rotation_audit.append(dict(asset_id=sid,pairs=pairs,all_required_rotations_exactly_distinct=True))
    audit=dict(status='PASS_FIXED_PRE_OUTCOME_BALANCE',model_calls=0,sample_selection_uses_answers=False,
        natural_video_count=600,unique_backgrounds=len(used_bg),unique_objects=len(used_obj),
        by_group=dict(Counter(r['group'] for r in rows)),by_task_group={task:dict(Counter(r['group'] for r in rows if r['task']==task)) for task in ('T4','T5')},
        feature_gate=.5,family_count_deviation_gate=2,allocation_searches=audits,
        background_video_frequencies=dict(Counter(r['background_id'] for r in rows)),object_video_frequencies=dict(Counter(r['object_id'] for r in rows)),
        matched='T4 same background/object in LR pair; every coherence has6 seeds/source combination; T5 every timing-position block shares assets across4 occurrence classes x2 masks',
        T4_cross_coherence_assets='may differ: common seed controls geometry only; not same-scene psychometric curves',
        T5_source_onset='each onset has2 of4 source combinations; each source has6timing-position blocks and both positions3each; no claim that every onset contains all4 sources',
        measured_features=dict(background='local mean/std/gradient-energy/orientation at two spatial regions from existing verified preparation',object='rendered alpha area, alpha-weighted luminance mean/SD, log silhouette elongation, compactness'),
        T5_rotation_distinctness=rotation_audit,
        implementation_sha256=sha(__file__),asset_catalog_sha256=sha(p/'asset-catalog.json'))
    write(p/'natural-manifest.json',dict(schema='t45-natural-fixed-manifest-v1',protocol_id=base.NAME,items=rows,allocation_audit=audit))
    write(out/'balance-audit.json',audit)
    fixed=read(p/'allocation.json');assert {r['sample_id'] for r in fixed['items'] if r['domain']=='natural'}=={r['sample_id'] for r in rows}
    fixed['status']='FIXED1200_ASSETS_BALANCE_PASS';fixed['natural_manifest_sha256']=sha(p/'natural-manifest.json');write(p/'allocation-with-assets.json',fixed)
    print(json.dumps({'status':audit['status'],'natural600':len(rows),'BG':len(used_bg),'OBJ':len(used_obj)}),flush=True)


def variant_paths(root):
    p,d,out=ORIGINAL_PATHS(root);return p/V2,d/V2,out/V2


def support_audit(root,rows):
    """Conservative support of positive-alpha output pixel *cells*, not centers.

    For a nonintegral translation, bilinear interpolation makes samples at q
    and q+1 nonzero. Pixel cells add another half pixel. Hull+[ -.5,1.5 ]-frac
    thus bounds every occupied cell. No claim that every envelope point is
    occupied, or that the earlier envelope violation was an actual overlap.
    """
    from scipy.spatial import ConvexHull
    p,d,out=base.paths(root);hulls={};records=[]
    for row in rows:
        if row['task']!='T4':continue
        sid=row['object_id']
        if sid not in hulls:
            z=np.load(d/'prepared'/(sid+'-T4.npy'));yy,xx=np.where(z[:,:,3]>0);xy=np.c_[xx,yy]-27.5
            hulls[sid]=xy[ConvexHull(xy).vertices]
        hull=hulls[sid];start=np.array(row['positions_xy_px']);v=np.array(row['velocity_xy_px_s'])
        pos=start[None]+np.arange(180)[:,None,None]*v[None]/60;delta=np.mod(pos.reshape(-1,2)-28,1)
        low=hull[None]-.5-delta[:,None];high=hull[None]+1.5-delta[:,None]
        ext=np.maximum(abs(low),abs(high));radius=float(np.sqrt((ext**2).sum(2)).max())
        # A uniform bound includes every fractional phase between frames.
        uniform=float(np.sqrt(((abs(hull)+1.5)**2).sum(1)).max())
        xy_extent=abs(hull).max(0)+1.5
        endpoints=np.stack([start,start+v*(179/60)])
        edge=float(min((endpoints-xy_extent).min(),(np.array([768,512])-endpoints-xy_extent).min()))
        gap=row['source_parameters']['minimum_continuous_disc_gap_px']+56-2*uniform
        assert radius<=28 and uniform<=28 and edge>=8-1e-7 and gap>=4-1e-6,(sid,radius,uniform,edge,gap)
        records.append(dict(sample_id=row['sample_id'],asset_id=sid,maximum_rendered_cell_envelope_radius_px=radius,
            uniform_all_fractional_phases_cell_envelope_radius_px=uniform,minimum_conservative_continuous_gap_px=gap,
            minimum_conservative_edge_margin_px=edge,frames=180))
    audit=dict(status='PASS_STRICT56_CIRCLE_ALL_PHASES',T4_sources=len(records),
        maximum_actual_phase_pixel_cell_envelope_radius_px=max(x['maximum_rendered_cell_envelope_radius_px'] for x in records),
        maximum_uniform_pixel_cell_envelope_radius_px=max(x['uniform_all_fractional_phases_cell_envelope_radius_px'] for x in records),
        minimum_conservative_continuous_gap_px=min(x['minimum_conservative_continuous_gap_px'] for x in records),
        minimum_conservative_edge_margin_px=min(x['minimum_conservative_edge_margin_px'] for x in records),
        support_definition='positive-alpha destination pixel cells, including half-pixel corners; hull+interpolation envelope may include unoccupied points',
        classic_comparison='Classic analytic area-coverage disks have exact continuous radius28; boundary pixel cells extend beyond that analytic circle. Natural check deliberately uses stricter enclosing entire positive-alpha cells.',
        prior_v1='28.5985976366 was a conservative envelope of positive-alpha pixel-cell corners at actual frame phases, not center radius and not evidence of actual overlap. Center distances remained>=60px. V2 enforces the requested56px support conservatively.',
        records=records,model_calls=0,implementation_sha256=sha(__file__))
    write(out/'support-audit.json',audit);return audit


def prepare_v2(root,workers):
    root=Path(root);oldp,oldd,oldout=ORIGINAL_PATHS(root);p,d,out=variant_paths(root)
    for folder in (p/'inputs',d/'prepared',d/'references',d/'videos',d/'qa',d/'inputs',out):folder.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','asset-catalog.json'):
        target=p/name
        if not target.exists():shutil.copy2(oldp/name,target)
        else:assert sha(target)==sha(oldp/name)
    for src in (oldp/'inputs').iterdir():
        dst=p/'inputs'/src.name
        if not dst.exists():shutil.copy2(src,dst)
        else:assert sha(dst)==sha(src)
    library=d/'inputs/library'
    if not library.exists():library.symlink_to(oldd/'inputs/library',target_is_directory=True)
    cat=read(p/'asset-catalog.json');base.paths=variant_paths
    for a in cat['backgrounds']:
        src=oldd/'prepared'/(a['id']+'.npy');assert src.exists()
        dst=d/'prepared'/src.name
        if not dst.exists():dst.symlink_to(src)
        else:assert sha(dst)==sha(src)
    with ProcessPoolExecutor(workers) as pool:
        objects=list(pool.map(prepare_object,[(str(root),a,25) for a in cat['objects']]))
    features={r['asset_id']:r for r in objects};old=read(oldp/'natural-manifest.json');rows=[];mapping=[]
    for previous in old['items']:
        row=dict(previous)
        if row['task']=='T4':
            prep=features[row['object_id']]['T4'];row['parent_v1_source_id']=row['sample_id']
            row['sample_id']='n45v2_'+digest([V2,row['sample_id'],prep['prepared_sha256']])[:24]
            row['source_parameters']=dict(row['source_parameters'],object_preparation=prep,geometry_revision=V2)
            mapping.append(dict(parent_v1_source_id=previous['sample_id'],parent_classic_id=row['parent_classic_id'],sample_id=row['sample_id'],task='T4',card=row['card']))
        else:
            # Exactly preserve all already rendered T5 bytes and QA; absent
            # Radon sources use the unchanged scheduler/canonical/reference.
            qsrc=oldd/'qa'/(row['sample_id']+'.json');qdest=d/'qa'/qsrc.name
            if qsrc.exists() and not qdest.exists():qdest.symlink_to(qsrc)
            ref=d/'references'/(row['object_id']+'.png');oldref=oldd/'references'/ref.name
            assert sha(ref)==sha(oldref),('T5 reference changed',row['object_id'])
        rows.append(row)
    assert len(mapping)==408 and len(rows)==600
    audit=support_audit(root,rows)
    write(p/'object-features.json',objects)
    write(p/'replacement-map.json',dict(schema='t45-natural-T4-fit56-v2-replacements-v1',items=mapping,
        unchanged_T5_ids=[r['sample_id'] for r in rows if r['task']=='T5'],model_calls=0,
        reason='Uniform extra source margin for whole positive-alpha pixel cells under fractional bilinear motion; no actual-overlap claim'))
    write(p/'natural-manifest.json',dict(schema='t45-natural-fit56-v2-manifest-v1',protocol_id=base.NAME,
        source_revision=V2,items=rows,parent_manifest_sha256=sha(oldp/'natural-manifest.json'),
        prior_balance_audit_sha256=sha(oldout/'balance-audit.json'),support_audit_sha256=sha(out/'support-audit.json')))
    write(p/'current1200-manifest.json',dict(schema='t45-current1200-fit56-v2-v1',items=read(oldp/'manifest.json')['items']+rows,
        counts=dict(classic600=600,naturalT4_v2=408,naturalT5_unchanged=192,total=1200),
        original_identity_plan_sha256=sha(oldp/'allocation.json'),replacement_map_sha256=sha(p/'replacement-map.json')))
    print(json.dumps({k:v for k,v in audit.items() if k!='records'}),flush=True)


def classic_variant_paths(root):
    p,d,out=ORIGINAL_PATHS(root);return p/CLASSIC_V2,d/CLASSIC_V2,out/CLASSIC_V2


def corrected_classic_t4_frames(root,row):
    _,science=ORIGINAL_T4_FRAMES(root,row)
    positions=np.array(row['positions_xy_px']);velocity=np.array(row['velocity_xy_px_s'])
    def frame_at(f):
        result=np.full((512,768,3),int(base.rgb8(np.array(base.BG))),np.uint8)
        for cx,cy in positions+velocity*f/60:
            x,y,alpha=base.disc_patch(cx,cy);codes=base.rgb8(base.BG+(base.FG-base.BG)*alpha)
            roi=result[y:y+alpha.shape[0],x:x+alpha.shape[1]]
            # Empty parts of a disk's bounding box may intersect another disk.
            # Only covered pixels are assigned. Analytic4px circle gaps ensure
            # no pixel cell has positive coverage from two disks.
            active=alpha>0;roi[active]=codes[active,None]
        return result
    science.update(pixel_rule='Only positive analytic disk coverage writes pixels; empty bounding-box background never overwrites another disk',
        pixel_revision=CLASSIC_V2,pixel_revision_implementation_sha256=sha(__file__))
    return frame_at,science


def prepare_classic_v2(root):
    root=Path(root);oldp,oldd,oldout=ORIGINAL_PATHS(root);p,d,out=classic_variant_paths(root)
    for folder in (p/'inputs',d/'videos',d/'qa',out):folder.mkdir(parents=True,exist_ok=True)
    for src in (oldp/'inputs').iterdir():
        dst=p/'inputs'/src.name
        if not dst.exists():shutil.copy2(src,dst)
        else:assert sha(dst)==sha(src)
    old=read(oldp/'manifest.json');rows=[]
    for previous in old['items']:
        if previous['task']!='T4':continue
        row=dict(previous);row['parent_v1_source_id']=row['sample_id'];row['sample_id']='c45v2_'+digest([CLASSIC_V2,row['sample_id']])[:24]
        row['source_parameters']=dict(row['source_parameters'],pixel_revision=CLASSIC_V2,pixel_revision_implementation_sha256=sha(__file__))
        rows.append(row)
    write(p/'manifest.json',dict(schema='t45-classic-disk-union-v2-candidates-v1',items=rows,model_calls=0,
        old_manifest_sha256=sha(oldp/'manifest.json'),implementation_sha256=sha(__file__),
        revision='Correct empty bounding-box overwrites only; all positions, velocities, seeds, prompts, GT, luminance and analytic disk-area algorithm retained'))
    print(json.dumps({'prepared_classic_T4_candidates':len(rows),'T5_changes':0}),flush=True)


def finish_classic_v2(root,host):
    root=Path(root);oldp,oldd,oldout=ORIGINAL_PATHS(root);p,d,out=classic_variant_paths(root)
    oldcat=read(oldp/(host+'-sources.json'))['items'];newcat=read(p/(host+'-sources.json'))['items']
    original=[x for x in oldcat.values() if x['row']['source_parameters']['domain']=='classic']
    candidates={q['row']['parent_v1_source_id']:q for q in (read(x['source_qa_path']) for x in newcat.values())}
    current={};records=[];changed=[];unchanged=[]
    for item in original:
        oldid=item['sample_id']
        if item['task']=='T5':current[oldid]=item;records.append(dict(parent_v1_source_id=oldid,current_sample_id=oldid,task='T5',changed=False,changed_frames=0));continue
        newq=candidates[oldid];oldq=read(item['source_qa_path']);candidate=newcat[newq['sample_id']]
        for field in ('truth','prompt','choices','accuracy_eligible','positions_xy_px','velocity_xy_px_s','seed','coherence','direction'):
            assert oldq['row'][field]==newq['row'][field],field
        changes=[i for i,(a,b) in enumerate(zip(oldq['decoded_RGB8_frame_hashes'],newq['decoded_RGB8_frame_hashes'])) if a!=b]
        changed_stream=oldq['stream_sha256']!=newq['stream_sha256'];assert changed_stream==bool(changes)
        selected=candidate if changed_stream else item;current[selected['sample_id']]=selected
        record=dict(parent_v1_source_id=oldid,current_sample_id=selected['sample_id'],candidate_sample_id=candidate['sample_id'],
            task='T4',seed=oldq['row']['seed'],coherence=oldq['row']['coherence'],direction=oldq['row']['direction'],changed=changed_stream,
            changed_frames=len(changes),first_changed_frame=changes[0] if changes else None,changed_frame_indices=changes,
            old_video_sha256=oldq['video_sha256'],candidate_video_sha256=newq['video_sha256'],
            old_stream_sha256=oldq['stream_sha256'],candidate_stream_sha256=newq['stream_sha256'],prompt_GT_unchanged=True)
        records.append(record);(changed if changed_stream else unchanged).append(oldid)
    assert len(current)==(464 if host=='fourier' else 136)
    write(p/(host+'-current-sources.json'),dict(schema='t45-classic-current-source-catalog-v1',items=current))
    audit=dict(status='PASS_PER_ID_OLD_NEW_RGB_COMPARISON',host=host,classic_sources=len(current),changed_T4=len(changed),unchanged_T4=len(unchanged),unchanged_T5=sum(x['task']=='T5' for x in original),
        changed_parent_ids=changed,unchanged_parent_ids=unchanged,records=records,model_calls=0,
        current_catalog_sha256=sha(p/(host+'-current-sources.json')),candidate_catalog_sha256=sha(p/(host+'-sources.json')),
        rule='Reuse prior input/answer only when every decoded RGB frame hash and prompt/GT are identical; replacement decisions never read model answers')
    write(out/(host+'-replacement-audit.json'),audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('records','changed_parent_ids','unchanged_parent_ids')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','allocate','prepare-v2','render-v2','prepare-classic-v2','render-classic-v2','finish-classic-v2'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    host='fourier' if str(args.root).startswith('/home/yang') else 'radon'
    if args.action=='prepare':prepare(args.root,args.workers)
    elif args.action=='allocate':allocate(args.root)
    elif args.action=='prepare-v2':prepare_v2(args.root,args.workers)
    elif args.action=='render-v2':
        base.paths=variant_paths
        base.render(args.root,host,args.workers,'natural','all',None,0)
    elif args.action=='prepare-classic-v2':prepare_classic_v2(args.root)
    elif args.action=='finish-classic-v2':finish_classic_v2(args.root,host)
    else:
        base.paths=classic_variant_paths;base.t4_frames=corrected_classic_t4_frames
        base.render(args.root,host,args.workers,'classic','T4',None,0)
