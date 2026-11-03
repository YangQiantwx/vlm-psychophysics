"""Deterministic, outcome-independent T3 allocation and source publication audit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,linear_sum_assignment
from scipy.sparse import coo_matrix
from .natural_t13_stimuli import ANGLES,GROUPS,read,write,sha,digest
from .natural_t3_classic_paired import NAME,paths
from .dataset_migration_preflight import asset_alias

class IntegerDesign:
    def __init__(self,n):self.n=n;self.rr=[];self.cc=[];self.vv=[];self.lo=[];self.hi=[]
    def add(self,a,l,h):
        a=np.asarray(a).ravel();ix=np.flatnonzero(a);i=len(self.lo)
        self.rr.extend([i]*len(ix));self.cc.extend(ix.tolist());self.vv.extend(a[ix].tolist());self.lo.append(l);self.hi.append(h)
    def solve(self,seed=20261102,upper=None):
        m=coo_matrix((self.vv,(self.rr,self.cc)),shape=(len(self.lo),self.n)).tocsc()
        result=milp(np.random.default_rng(seed).random(self.n),integrality=np.ones(self.n),
            bounds=Bounds(0,np.ones(self.n) if upper is None else upper),
            constraints=LinearConstraint(m,self.lo,self.hi),options={'time_limit':60,'mip_rel_gap':.2})
        assert result.x is not None,result.message
        x=np.rint(result.x);assert np.all(m@x>=np.array(self.lo)-1e-7) and np.all(m@x<=np.array(self.hi)+1e-7)
        return x

def objects(features,cats):
    # Order semantic identities before making the cyclic four-item extras;
    # alphabetic object IDs cluster families and are not a random order.
    z=(features-features.mean(0))/np.maximum(features.std(0),1e-10)
    for order_seed in range(100000):
        order=np.random.default_rng(order_seed).permutation(20)
        windows=np.array([order[(np.arange(4)+s)%20] for s in range(20)])
        if abs(z[windows].sum(1)/24).max()<=.5 and abs(cats[windows].sum(1)-4*cats.mean(0)).max()<=2:break
    else:raise AssertionError('No object order balances every cyclic extra window')
    features=features[order];cats=cats[order]
    model=IntegerDesign(560)
    for a in range(28):
        v=np.zeros((28,20));v[a]=1;model.add(v,1,1)
    for s in range(20):
        v=np.zeros((28,20));v[:,s]=1;model.add(v,1,2)
    for o in range(20):
        extra=((o-np.arange(20))%20<4).astype(float)
        for q in range(4):
            v=np.zeros((28,20));v[q*7:q*7+7]=extra;model.add(v,1,2)
        model.add(np.tile(extra,28),5,6)
        v=np.array([(-1.)**((o-np.arange(20)+a)%2)*(1+extra) for a in range(28)])
        model.add(v,-1,1)
    shifts=np.argmax(model.solve().reshape(28,20),1)
    arr=(np.arange(24)[None,:]+shifts[:,None])%20
    counts=np.bincount(arr.ravel(),minlength=20)
    delta=np.array([sum((-1)**((k+a)%2) for a in range(28) for k in range(24) if arr[a,k]==o) for o in range(20)])
    timing=np.array([[sum(arr[:,k]==o) for k in range(24)] for o in range(20)])
    # The 90-degree row has one photo and one generated object at each onset.
    # Preserve all 20 objects, improve global frequency, and make individual LR totals exact or differ by one.
    z=(features-features.mean(0))/np.maximum(features.std(0),1e-10)
    model=IntegerDesign(480)
    for k in range(24):
        v=np.zeros((24,20));v[k]=1;model.add(v,1,1)
    for o in range(20):
        v=np.zeros((24,20));v[:,o]=1;model.add(v,1,2 if counts[o]==33 else 1)
        v[:,o]=np.array([1,-1,-1,1]*6);model.add(v,-1-2*delta[o],1-2*delta[o])
    for parity in (0,1):
        for column in z.T:
            v=np.zeros((24,20));v[parity::2]=column;model.add(v,-6,6)
        for column in cats.T:
            v=np.zeros((24,20));v[parity::2]=column;expected=12*column.mean();model.add(v,expected-2,expected+2)
    upper=(timing.T<2).ravel().astype(float)
    row=np.argmax(model.solve(20261103,upper).reshape(24,20),1)
    return order[np.vstack([arr,row])],dict(object_order_seed=order_seed,object_order=order.tolist(),shifts=shifts.tolist(),non90_counts=counts.tolist(),non90_side_slot_difference=delta.tolist())

def backgrounds(features,families,obj):
    z=(features-features.mean(0))/np.maximum(features.std(0),1e-10)
    cats=np.array([[f==v for v in sorted(set(families))] for f in families],float)
    for seed in range(100000):
        base=np.resize(np.random.default_rng(seed).permutation(100),696).reshape(29,24)
        if abs(z[base].mean(1)).max()<=.5 and abs(cats[base].sum(1)-24*cats.mean(0)).max()<=2:break
    else:raise AssertionError('No feature-balanced cyclic background allocation')
    arr=base[:28].copy();rng=np.random.default_rng(20261104)
    # Within-angle swaps preserve every identity-by-angle count and feature balance.
    # Balance broad time bins, physical sign, and avoid repeating BG/OBJ semantic pairings.
    bins=np.zeros((100,4),int);sign=np.zeros(100,int);pairs=np.zeros((100,20),int)
    weight=np.array([[2,0,-2,0][(a+k)%4] for a in range(28) for k in range(24)]).reshape(28,24)
    for a in range(28):
        for k,b in enumerate(arr[a]):bins[b,k//6]+=1;sign[b]+=weight[a,k];pairs[b,obj[a,k]]+=1
    def cost(b):
        return (np.maximum(1-bins[b],0)**2+np.maximum(bins[b]-2,0)**2).sum()*30+max(abs(sign[b])-2,0)**2*3+(np.maximum(pairs[b]-1,0)**2).sum()
    costs=np.array([cost(b) for b in range(100)],float);start=time.monotonic()
    for step in range(800000):
        a=int(rng.integers(28));k,l=rng.choice(24,2,replace=False);x,y=arr[a,k],arr[a,l]
        old=costs[x]+costs[y];bk,bl=k//6,l//6;ok,ol=obj[a,k],obj[a,l];dw=weight[a,l]-weight[a,k]
        bins[x,bk]-=1;bins[x,bl]+=1;bins[y,bl]-=1;bins[y,bk]+=1;sign[x]+=dw;sign[y]-=dw
        pairs[x,ok]-=1;pairs[x,ol]+=1;pairs[y,ol]-=1;pairs[y,ok]+=1
        nx,ny=cost(x),cost(y);temperature=max(.03,2*(1-step/800000))
        if nx+ny<=old or rng.random()<math.exp(-min(700,(nx+ny-old)/temperature)):
            arr[a,k],arr[a,l]=y,x;costs[x]=nx;costs[y]=ny
        else:
            bins[x,bk]+=1;bins[x,bl]-=1;bins[y,bl]+=1;bins[y,bk]-=1;sign[x]-=dw;sign[y]+=dw
            pairs[x,ok]+=1;pairs[x,ol]-=1;pairs[y,ol]+=1;pairs[y,ok]-=1
        if step%2000==0 and bins.min()>=1 and bins.max()<=2 and abs(sign).max()<=2 and pairs.max()<=1:break
    assert bins.min()>=1 and bins.max()<=2 and abs(sign).max()<=2,('background marginal balance',step,bins.min(),bins.max(),abs(sign).max())
    tail=base[28];allowed=np.array([[bins[b,k//6]<2 for k in range(24)] for b in tail])
    for attempt in range(20000):
        c=rng.random((24,24))+1e6*(~allowed)
        ix,jx=linear_sum_assignment(c);row=np.empty(24,int);row[jx]=tail[ix]
        assert allowed[ix,jx].all()
        if all(abs(z[row[parity::2]].mean(0)).max()<=.5 and abs(cats[row[parity::2]].sum(0)-12*cats.mean(0)).max()<=2 for parity in (0,1)):break
    else:raise AssertionError('90-degree background group balance')
    return np.vstack([arr,row]),dict(cyclic_seed=seed,swap_steps=step+1,seconds=time.monotonic()-start,
        non90_onset_bin_count_range=[int(bins.min()),int(bins.max())],non90_physical_sign_max_difference=int(abs(sign).max()),
        non90_max_semantic_BG_OBJ_pair_reuse=int(pairs.max()),horizontal_assignment_attempts=attempt+1)

def audit_rows(rows,classic):
    expected={(float(r['angle_deg']),r['physical_tilt_sign'],r['onset_frame'],r['truth']):r['classic_sample_id'] for r in classic}
    actual={(float(r['angle_deg']),r['physical_tilt_sign'],r['onset_frame'],r['truth']):r['classic_sample_id'] for r in rows}
    assert len(rows)==len(expected)==len(actual)==2736 and actual==expected
    assert len(set(r['sample_id'] for r in rows))==2736
    groups=Counter(r['group'] for r in rows);assert set(groups.values())=={684}
    for angle in ANGLES:
        cell=[r for r in rows if r['angle_deg']==angle];assert len(cell)==(48 if angle==90 else 96)
        assert len(set(r['onset_frame'] for r in cell))==24
        assert set(Counter(r['group'] for r in cell).values())=={12 if angle==90 else 24}
        for g in groups:
            v=[r for r in cell if r['group']==g];assert Counter(r['truth'] for r in v)['LEFT']==len(v)//2
            if angle!=90:
                assert len(set(r['object_id'] for r in v))==20
                assert len(set(r['onset_frame'] for r in v))==24
                assert set(Counter((r['physical_tilt_sign'],r['truth']) for r in v).values())=={6}
    for kind in ['object_id','background_id']:
        for ident in set(r[kind] for r in rows):
            v=[r for r in rows if r[kind]==ident];c=Counter(r['truth'] for r in v)
            assert abs(c['LEFT']-c['RIGHT'])<=1,(ident,c)
            if kind=='object_id':
                assert len(set(r['angle_deg'] for r in v))==29
                assert len(set(r['onset_frame'] for r in v))==24
    return dict(status='PASS',classic_configurations=2736,natural_videos=2736,missing=0,duplicates=0,
        source_group_counts=dict(groups),object_identities=40,background_identities=200,
        every_object_all_29_angles=True,every_object_all_24_onsets=True,individual_asset_LR_difference_max=1,
        physical_sign_convention='positive means upper endpoint left; natural renderer theta=-classic physical angle; horizontal sign0',model_outputs_used=False)

def audit_features(rows,assets,prepared):
    result={};om={x['asset_id']:x for x in prepared['objects']}
    for kind,key in [('object','object_id'),('background','background_id')]:
        maximum_sd=0.;maximum_family=0.
        for source in ['photo','gpt']:
            pool=[x for x in assets[kind+'s'] if x['id'].startswith(source+'_')]
            feature=np.array([om[x['id']]['features'] if kind=='object' else x['features'] for x in pool])
            mu=feature.mean(0);sd=np.maximum(feature.std(0),1e-10)
            fm={x['id']:x.get('family','kitchen' if x.get('pair_id')=='obj_051' else 'workbench') for x in pool}
            values={x['id']:f for x,f in zip(pool,feature)}
            for angle in ANGLES:
                for bgsource,objsource in GROUPS:
                    group=bgsource+'_'+objsource
                    cell=[x for x in rows if x['angle_deg']==angle and x['group']==group and x[key].startswith(source+'_')]
                    if not cell:continue
                    maximum_sd=max(maximum_sd,float(abs((np.array([values[x[key]] for x in cell]).mean(0)-mu)/sd).max()))
                    for family in set(fm.values()):
                        maximum_family=max(maximum_family,abs(sum(fm[x[key]]==family for x in cell)-len(cell)*sum(v==family for v in fm.values())/len(pool)))
        assert maximum_sd<=.5 and maximum_family<=2,(kind,maximum_sd,maximum_family)
        result[kind]=dict(max_cell_mean_shift_SD=maximum_sd,max_family_count_deviation=maximum_family)
    return result

def allocate(root):
    p,d,r=paths(root);assets=read(p/'assets.json');prep=read(p/'asset-preparation.json')
    classic=read(p/'classic-configurations.json')['rows'];objids=assets['eligible_pairs'];bgids=sorted(set(x['pair_id'] for x in assets['backgrounds']))
    prompt_path=Path(root)/'plans/natural-t13-20261029/T3-prompt.txt'
    prompt=prompt_path.read_text();assert all(x['prompt'].strip()==prompt.strip() for x in classic)
    om={x['asset_id']:x for x in prep['objects']};am={x['id']:x for x in assets['objects']+assets['backgrounds']}
    of=np.array([om['photo_'+x]['features']+om['gpt_'+x]['features'] for x in objids])
    bf=np.array([am['photo_'+x]['features']+am['gpt_'+x]['features'] for x in bgids])
    families=[[am[source+'_'+x].get('family','kitchen' if x=='obj_051' else 'workbench') for x in objids] for source in ['photo','gpt']]
    cats=np.concatenate([np.array([[f==v for v in sorted(set(fs))] for f in fs],float) for fs in families],axis=1)
    obj,odiag=objects(of,cats)
    bg,bdiag=backgrounds(bf,[am['gpt_'+x]['family'] for x in bgids],obj)
    lookup={(float(x['angle_deg']),x['physical_tilt_sign'],x['onset_index'],x['truth']):x for x in classic}
    rows=[]
    for ai,angle in enumerate(ANGLES):
        for k in range(24):
            combinations=[(1,'LEFT'),(1,'RIGHT'),(-1,'LEFT'),(-1,'RIGHT')] if angle!=90 else [(0,'LEFT'),(0,'RIGHT')]
            for j,(sign,side) in enumerate(combinations):
                g=(k+j+ai)%4 if angle!=90 else [[0,3],[1,2],[3,0],[2,1]][k%4][j]
                bs,os=GROUPS[g];bid=bs+'_'+bgids[bg[ai,k]];oid=os+'_'+objids[obj[ai,k]]
                source=lookup[(float(angle),sign,k,side)]
                row=dict(task='T3',angle_deg=angle,physical_tilt_sign=sign,render_theta_deg=90 if angle==90 else -sign*angle,
                    onset_index=k,onset_frame=18+5*k,offset_frame=48+5*k,truth=side,background_id=bid,object_id=oid,
                    background_source=bs,object_source=os,group=bs+'_'+os,classic_sample_id=source['classic_sample_id'],
                    classic_video_sha256=source['classic_video_sha256'],background_source_sha256=am[bid]['sha256'],
                    background_prepared_path=am[bid]['prepared_path'],background_prepared_sha256=am[bid]['prepared_sha256'],
                    object_source_sha256=am[oid]['sha256'],prompt=prompt,schema='natural-T3-classic-paired-v1')
                row['sample_id']='n'+digest(row)[:24]
                direction='horizontal' if sign==0 else ('tiltLeft' if sign>0 else 'tiltRight')
                row['path']=f'T3/T3_vertical{side.title()}_{direction}_{angle:g}deg_bg-{asset_alias(bid)}_obj-{asset_alias(oid)}_trial{k+1:02d}.mp4'
                row['representative']=bool(ai in [0,3,10,20,24,27,28] and k==0)
                rows.append(row)
    audit=audit_rows(rows,classic)
    audit['measured_covariate_balance']=audit_features(rows,assets,prep)
    # Full count tables remain available for checking every asset, not just aggregate claims.
    matrices={}
    for kind in ['object_id','background_id']:
        matrices[kind]={ident:dict(total=sum(x[kind]==ident for x in rows),
            angles=dict(Counter(f'{x["angle_deg"]:g}' for x in rows if x[kind]==ident)),
            onsets=dict(Counter(str(x['onset_index']) for x in rows if x[kind]==ident)),
            sign=dict(Counter(str(x['physical_tilt_sign']) for x in rows if x[kind]==ident)),
            side=dict(Counter(x['truth'] for x in rows if x[kind]==ident))) for ident in sorted(set(x[kind] for x in rows))}
    data=dict(schema='natural-T3-classic-paired-allocation-v1',rows=rows,diagnostics={'objects':odiag,'backgrounds':bdiag},
        classic_manifest_sha256=sha(p/'classic-configurations.json'),assets_sha256=sha(p/'assets.json'),asset_preparation_sha256=sha(p/'asset-preparation.json'),
        implementation_sha256=sha(__file__),prompt_sha256=sha(prompt_path),model_outputs_used=False,model_calls=0)
    write(p/'allocation.json',data);write(p/'allocation-audit.json',{**audit,'allocation_sha256':sha(p/'allocation.json'),'count_tables':matrices})
    print(json.dumps({**audit,'diagnostics':data['diagnostics']},ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();allocate(args.root)
if __name__=='__main__':main()
