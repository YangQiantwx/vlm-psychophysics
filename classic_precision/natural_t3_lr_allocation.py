"""Outcome-independent T3 left/right blocks with maximal exact source reuse.

Select one accepted video per physical classic LR pair, then render its opposite
side with identical assets. Scientific outputs are never read by this allocator.
"""
from __future__ import annotations
import argparse, copy, json, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, hstack
from .natural_t13_stimuli import ANGLES, read, write, sha, digest
from .dataset_migration_preflight import asset_alias

NAME = 'natural-t3-lr-paired-20261102'
PRIOR = 'natural-t3-classic-paired-20261102'

def paths(root):
    root=Path(root)
    return root/'plans'/NAME,root/'datasets'/NAME,root/'runs/stimulus-preparation'/NAME

def key(row):
    return (float(row['angle_deg']),row['physical_tilt_sign'],row['onset_index'])

class Design:
    def __init__(self,n):
        self.n=n;self.rr=[];self.cc=[];self.vv=[];self.lo=[];self.hi=[];self.names=[]
    def add(self,indices,low,high,name,weights=None):
        indices=list(indices)
        if not indices:
            assert low<=0<=high,(name,low,high)
            return
        j=len(self.lo);self.rr.extend([j]*len(indices));self.cc.extend(indices)
        self.vv.extend([1.]*len(indices) if weights is None else list(weights))
        self.lo.append(low);self.hi.append(high);self.names.append(name)
    def solve(self,cost,previous=None):
        matrix=coo_matrix((self.vv,(self.rr,self.cc)),shape=(len(self.lo),self.n)).tocsc()
        started=time.monotonic()
        # First enforce discrete coverage/paired-control counts. Visual feature
        # means are optimized using a continuous minimax slack, not made into
        # fictitious guarantees of identical natural-image appearance.
        hard=np.array([x[0] not in ('feature','family') for x in self.names])
        if previous is None:
            result=milp(np.zeros(self.n),integrality=np.ones(self.n),bounds=Bounds(0,1),
                        constraints=LinearConstraint(matrix[hard],np.array(self.lo)[hard],np.array(self.hi)[hard]),
                        options={'time_limit':60,'presolve':True,'threads':12})
            assert result.x is not None,(result.message,len(self.lo))
            x=np.rint(result.x);status=result.message
        else:x=np.asarray(previous,float);status='Existing count-feasible allocation independently rechecked'
        observed=matrix@x
        assert np.all(observed[hard]>=np.array(self.lo)[hard]-1e-7) and np.all(observed[hard]<=np.array(self.hi)[hard]+1e-7)
        baseline=x.copy();baseline_reuse=-float(np.array(cost)@x)
        # A bounded second solve improves image-covariate balance while retaining
        # a feasible count solution if the optional optimization hits its limit.
        feature=np.array([x[0]=='feature' for x in self.names]);f=matrix[feature]
        counts=np.array([2*self.hi[i] for i in np.flatnonzero(feature)])
        baseline_shift=float(np.max(abs(f@x)/counts));limit=baseline_shift+1e-8
        augmented=hstack([matrix[hard],np.zeros((int(hard.sum()),1))]).tocsc()
        fm=hstack([f,-counts[:,None]]).tocsc();negative=hstack([-f,-counts[:,None]]).tocsc()
        objective=np.r_[np.asarray(cost),.001] if previous is not None else np.r_[np.asarray(cost)*.00001,1.]
        reuse_constraint=LinearConstraint(np.r_[cost,0][None,:],-np.inf,-baseline_reuse)
        optimized=milp(objective,integrality=np.r_[np.ones(self.n),0],
            bounds=Bounds(np.zeros(self.n+1),np.r_[np.ones(self.n),limit]),
            constraints=[LinearConstraint(augmented,np.array(self.lo)[hard],np.array(self.hi)[hard]),
                         LinearConstraint(fm,-np.inf,0),LinearConstraint(negative,-np.inf,0)]+([reuse_constraint] if previous is not None else []),
            options={'time_limit':60 if previous is not None else 90,'mip_rel_gap':.02,'presolve':True,'threads':12})
        if optimized.x is not None:
            candidate=np.rint(optimized.x[:self.n]);values=matrix@candidate
            assert np.all(values[hard]>=np.array(self.lo)[hard]-1e-7) and np.all(values[hard]<=np.array(self.hi)[hard]+1e-7)
            if float(np.max(abs(f@candidate)/counts))<=baseline_shift+1e-7:x=candidate;observed=values
        return x.astype(bool),dict(status=status,seconds=time.monotonic()-started,
            binary_variables=self.n,hard_constraints=int(hard.sum()),all_hard_constraints_verified=True,
            outcome_independent_seed=202611021,baseline_reuse=int(baseline_reuse),
            visual_covariate_optimization_status=optimized.message,baseline_max_mean_shift_SD=baseline_shift,
            final_max_mean_shift_SD=float(np.max(abs(f@x)/counts)))

def audit(rows,classic):
    expected={(float(r['angle_deg']),r['physical_tilt_sign'],r['onset_index'],r['truth']):r['classic_sample_id'] for r in classic}
    actual={(float(r['angle_deg']),r['physical_tilt_sign'],r['onset_index'],r['truth']):r['classic_sample_id'] for r in rows}
    assert actual==expected and len(rows)==len(actual)==2736
    assert len({r['sample_id'] for r in rows})==2736
    blocks=defaultdict(list)
    for row in rows:blocks[key(row)].append(row)
    assert len(blocks)==1368
    equal=['background_id','object_id','background_source_sha256','background_prepared_sha256',
           'object_source_sha256','render_theta_deg','onset_frame','offset_frame','group','prompt']
    for pair in blocks.values():
        assert len(pair)==2 and {r['truth'] for r in pair}=={'LEFT','RIGHT'}
        assert all(pair[0][k]==pair[1][k] for k in equal)
    chosen=[v[0] for v in blocks.values()]
    assert len({(r['background_id'],r['object_id']) for r in chosen})==1368
    assert set(Counter(r['group'] for r in rows).values())=={684}
    for angle in ANGLES:
        cell=[r for r in rows if r['angle_deg']==angle]
        assert len(cell)==(48 if angle==90 else 96)
        assert set(Counter(r['group'] for r in cell).values())=={12 if angle==90 else 24}
        for g in set(r['group'] for r in cell):
            subset=[r for r in cell if r['group']==g]
            assert Counter(r['truth'] for r in subset)['LEFT']==len(subset)//2
            if angle!=90:assert set(Counter(r['physical_tilt_sign'] for r in subset).values())=={12}
    objects=sorted({r['object_id'] for r in rows});backgrounds=sorted({r['background_id'] for r in rows})
    assert len(objects)==40 and len(backgrounds)==200
    for ident in objects:
        sample=[r for r in chosen if r['object_id']==ident]
        assert 33<=len(sample)<=35
        assert len({r['angle_deg'] for r in sample if r['angle_deg']!=90})==28
        assert len({r['onset_index'] for r in sample})==24
    for ident in backgrounds:
        sample=[r for r in chosen if r['background_id']==ident]
        assert 6<=len(sample)<=7
        assert len({r['onset_index']//6 for r in sample})==4
    return dict(status='PASS_STRICT_IDENTICAL_ASSET_LR_BLOCKS',classic_configurations=2736,
        natural_videos=2736,strict_LR_pairs=1368,missing=0,duplicates=0,
        source_group_counts=dict(Counter(r['group'] for r in rows)),
        object_identities=40,background_identities=200,distinct_BG_OBJ_pairs=1368,
        individual_asset_LR_difference_max=0,every_object_all_28_non90_angles=True,
        every_object_all_24_onsets=True,objects_at_90=len({r['object_id'] for r in chosen if r['angle_deg']==90}),
        coverage_limit='90 degrees has 24 LR pairs, so at most 24 distinct objects; full asset Cartesian product is not claimed.',
        model_outputs_used=False,model_calls=0)

def audit_features(rows,assets,prepared):
    """Measured image covariates; family quantization is in whole LR pairs."""
    result={};om={x['asset_id']:x for x in prepared['objects']}
    for kind,field in [('object','object_id'),('background','background_id')]:
        max_sd=0.;max_family=0.
        for source in ['photo','gpt']:
            pool=[x for x in assets[kind+'s'] if x['id'].startswith(source+'_')]
            feature=np.array([om[x['id']]['features'] if kind=='object' else x['features'] for x in pool])
            z=(feature-feature.mean(0))/np.maximum(feature.std(0),1e-10)
            values={x['id']:v for x,v in zip(pool,z)}
            families={x['id']:x.get('family','kitchen' if x.get('pair_id')=='obj_051' else 'workbench') for x in pool}
            for angle in ANGLES:
                for group in ['photo_photo','photo_gpt','gpt_photo','gpt_gpt']:
                    cell=[r for r in rows if r['angle_deg']==angle and r['group']==group and r[field] in values]
                    if not cell:continue
                    max_sd=max(max_sd,float(abs(np.array([values[r[field]] for r in cell]).mean(0)).max()))
                    for family in set(families.values()):
                        max_family=max(max_family,abs(sum(families[r[field]]==family for r in cell)-len(cell)*sum(v==family for v in families.values())/len(pool)))
        assert np.isfinite(max_sd) and np.isfinite(max_family)
        result[kind]=dict(max_cell_mean_shift_SD=max_sd,max_family_count_deviation_videos=max_family,
                         interpretation='Measured visual covariates; hard guarantees concern paired controls and declared discrete coverage, not equal natural-image appearance.')
    return result

def allocate(root):
    root=Path(root);p,d,r=paths(root);old=root/'plans'/PRIOR
    p.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    for name in ['assets.json','asset-preparation.json','classic-configurations.json']:
        target=p/name;data=(old/name).read_bytes()
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
    prior=read(old/'allocation.json');oldrows=prior['rows'];assets=read(p/'assets.json');prepared=read(p/'asset-preparation.json')
    original_blocks=defaultdict(list)
    for row in oldrows:original_blocks[key(row)].append(row)
    bm={x['id']:x for x in assets['backgrounds']};ometa={x['id']:x for x in assets['objects']}
    rows=[];cost=[]
    for k,pair in original_blocks.items():
        for group in ['photo_photo','photo_gpt','gpt_photo','gpt_gpt']:
            bs,osource=group.split('_');base=copy.deepcopy(pair[0])
            bg=bm[bs+'_'+base['background_id'].split('_',1)[1]]
            obj=ometa[osource+'_'+base['object_id'].split('_',1)[1]]
            base.update(group=group,background_source=bs,object_source=osource,
                background_id=bg['id'],object_id=obj['id'],background_source_sha256=bg['sha256'],
                background_prepared_path=bg['prepared_path'],background_prepared_sha256=bg['prepared_sha256'],object_source_sha256=obj['sha256'])
            exact=next((x for x in pair if x['group']==group),None)
            rows.append(copy.deepcopy(exact) if exact else base);cost.append(-int(exact is not None))
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[key(row)].append(i)
    assert len(groups)==1368 and all(len(ix)==4 for ix in groups.values())
    model=Design(len(rows))
    def ix(**conditions):
        return [i for i,row in enumerate(rows) if all(row[k]==v for k,v in conditions.items())]
    for k,indices in groups.items():model.add(indices,1,1,('one_existing_video',k))
    # For non-horizontal conditions the two physical tilt directions use the
    # complementary provenance pair. This preserves the accepted semantic asset
    # coverage at every angle/onset, and prevents a BG source clustering there.
    for angle in ANGLES[:-1]:
        for onset in range(24):
            for field in ['background_source','object_source']:
                model.add([i for i in ix(angle_deg=angle,onset_index=onset) if rows[i][field]=='photo'],1,1,
                          ('complementary_sources',angle,onset,field))
    for angle in ANGLES:
        for group in ['photo_photo','photo_gpt','gpt_photo','gpt_gpt']:
            indices=ix(angle_deg=angle,group=group);count=6 if angle==90 else 12
            model.add(indices,count,count,('group_by_angle',angle,group))
            if angle!=90:
                for sign in [-1,1]:model.add(ix(angle_deg=angle,group=group,physical_tilt_sign=sign),6,6,('group_sign',angle,group,sign))
    for onset in range(24):
        for group in ['photo_photo','photo_gpt','gpt_photo','gpt_gpt']:
            model.add(ix(onset_index=onset,group=group),14,15,('group_onset',onset,group))
    for obj in assets['objects']:
        ident=obj['id'];indices=ix(object_id=ident)
        model.add(indices,33,35,('object_frequency',ident))
        for angle in ANGLES:
            model.add(ix(object_id=ident,angle_deg=angle),0 if angle==90 else 1,1 if angle==90 else 2,('object_angle',ident,angle))
        for onset in range(24):model.add(ix(object_id=ident,onset_index=onset),1,2,('object_onset',ident,onset))
        for block in range(4):
            subset=[i for i in indices if rows[i]['angle_deg'] in ANGLES[block*7:block*7+7]]
            model.add(subset,8,9,('object_angle_band',ident,block))
        model.add(indices,-1,1,('object_sign',ident),[rows[i]['physical_tilt_sign'] for i in indices])
    for bg in assets['backgrounds']:
        ident=bg['id'];indices=ix(background_id=ident);model.add(indices,6,7,('background_frequency',ident))
        for angle in ANGLES:model.add(ix(background_id=ident,angle_deg=angle),0,1,('background_angle',ident,angle))
        for block in range(4):model.add([i for i in indices if rows[i]['onset_index']//6==block],1,2,('background_onset_band',ident,block))
        model.add(indices,-3,3,('background_sign',ident),[rows[i]['physical_tilt_sign'] for i in indices])
    pairs=defaultdict(list)
    for i,row in enumerate(rows):pairs[(row['background_id'],row['object_id'])].append(i)
    for k,indices in pairs.items():
        if len(indices)>1:model.add(indices,0,1,('unique_asset_pair',k))
    om={x['asset_id']:x for x in prepared['objects']}
    for kind,field in [('object','object_id'),('background','background_id')]:
        for source in ['photo','gpt']:
            pool=[x for x in assets[kind+'s'] if x['id'].startswith(source+'_')]
            features=np.array([om[x['id']]['features'] if kind=='object' else x['features'] for x in pool])
            z=(features-features.mean(0))/np.maximum(features.std(0),1e-10)
            zmap={x['id']:v for x,v in zip(pool,z)}
            families={x['id']:x.get('family','kitchen' if x.get('pair_id')=='obj_051' else 'workbench') for x in pool}
            for angle in ANGLES:
                for group in ['photo_photo','photo_gpt','gpt_photo','gpt_gpt']:
                    indices=[i for i in ix(angle_deg=angle,group=group) if rows[i][field] in zmap]
                    if not indices:continue
                    count=6 if angle==90 else 12
                    for col in range(features.shape[1]):
                        model.add(indices,-.5*count,.5*count,('feature',kind,source,angle,group,col),[zmap[rows[i][field]][col] for i in indices])
                    for family in set(families.values()):
                        expected=count*sum(x==family for x in families.values())/len(pool)
                        model.add(indices,expected-2,expected+2,('family',kind,source,angle,group,family),[int(families[rows[i][field]]==family) for i in indices])
    previous=None
    if (p/'allocation.json').exists():
        previous_allocation=read(p/'allocation.json');pg={key(x):x['group'] for x in previous_allocation['rows']}
        previous=[pg[key(x)]==x['group'] for x in rows]
        saved=r/'preceding-draft-allocation.json'
        if not saved.exists():saved.write_bytes((p/'allocation.json').read_bytes())
    selected,diagnostic=model.solve(cost,previous);new=[];reused=[];replacements=[];pair_table=[]
    for k,indices in groups.items():
        chosen=next(i for i in indices if selected[i]);base=rows[chosen];built=[]
        for original in original_blocks[k]:
            if original['group']==base['group']:
                item=copy.deepcopy(original);reused.append(item['sample_id'])
            else:
                item=copy.deepcopy(base)
                item.update(truth=original['truth'],classic_sample_id=original['classic_sample_id'],classic_video_sha256=original['classic_video_sha256'],schema='natural-T3-identical-asset-LR-paired-v1')
                for field in ['sample_id','path','representative']:item.pop(field,None)
                item['sample_id']='n'+digest(item)[:24]
                direction='horizontal' if item['physical_tilt_sign']==0 else ('tiltLeft' if item['physical_tilt_sign']>0 else 'tiltRight')
                item['path']=f'T3/T3_vertical{item["truth"].title()}_{direction}_{item["angle_deg"]:g}deg_bg-{asset_alias(item["background_id"])}_obj-{asset_alias(item["object_id"])}_trial{item["onset_index"]+1:02d}.mp4'
                item['representative']=original.get('representative',False)
                replacements.append(dict(new_sample_id=item['sample_id'],retired_sample_id=original['sample_id'],parent_is_alias=False))
            built.append(item)
        new.extend(built)
        pair_table.append(dict(angle_deg=k[0],physical_tilt_sign=k[1],onset_index=k[2],
            background_id=base['background_id'],object_id=base['object_id'],group=base['group'],
            left=next(x['sample_id'] for x in built if x['truth']=='LEFT'),right=next(x['sample_id'] for x in built if x['truth']=='RIGHT')))
    by_classic={x['classic_sample_id']:x for x in new};new=[by_classic[x['classic_sample_id']] for x in oldrows]
    report=audit(new,read(p/'classic-configurations.json')['rows']);report['measured_covariate_balance']=audit_features(new,assets,prepared)
    allocation=dict(schema='natural-T3-identical-asset-LR-allocation-v1',rows=new,
        prior_allocation=dict(path=str(old/'allocation.json'),sha256=sha(old/'allocation.json')),
        reused_source_ids=reused,replacements=replacements,LR_pairs=pair_table,
        reuse_upper_bound_within_prior_current_source=1368,reuse_optimization='Maximize exact retained videos subject to strict LR pairing and all declared allocation constraints; solver gap is reported, no global optimum claimed without proof.',
        diagnostics=diagnostic,implementation_sha256=sha(__file__),assets_sha256=sha(p/'assets.json'),
        asset_preparation_sha256=sha(p/'asset-preparation.json'),classic_manifest_sha256=sha(p/'classic-configurations.json'),
        model_outputs_used=False,model_calls=0)
    write(p/'allocation.json',allocation);write(p/'allocation-audit.json',{**report,'allocation_sha256':sha(p/'allocation.json'),'solver':diagnostic})
    print(json.dumps({**report,'solver':diagnostic,'reused':len(reused),'render_new':2736-len(reused)}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    allocate(parser.parse_args().root)
if __name__=='__main__':main()
