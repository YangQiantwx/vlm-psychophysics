"""Independent T3 pixel-area reference and image-only axis estimators.

Pixel squares are [x,x+1]×[y,y+1]. Positive angle tilts a vertical axis right.
The reference integrates horizontal strips with positive-part antiderivatives;
it neither imports nor duplicates Sutherland-Hodgman polygon clipping.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, sys, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.optimize import linprog, least_squares, minimize_scalar, brentq
import shapely
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/t3_precision_validation_20261020_v30/mathematics'
ANGLES=[0.,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,1.,1.5,2.,2.5,3.,4.,8.,12.,16.,24.,32.,45.,90.]
OFFSETS=[0.,.25,.5,.75]
BG,FG=.1,.82

def angle_error(a,b):
    return abs((a-b+90)%180-90)

def vertices(theta,cx=128.,cy=128.,width=20.,height=190.):
    t=math.radians(theta);c,s=math.cos(t),math.sin(t)
    return np.array([[cx+c*u+s*v,cy-s*u+c*v] for u,v in
        [(-width/2,-height/2),(width/2,-height/2),(width/2,height/2),(-width/2,height/2)]])

def _hinge_integral(m,b,ya,yb):
    """Integral of max(m*y+b,0), vectorized in b without cancellation."""
    p=m*ya+b;q=m*yb+b;d=yb-ya
    if abs(m)<1e-15:
        return np.maximum(p,0)*d
    both=(p>=0)&(q>=0)
    cross=(p*q<0)
    out=np.where(both,(p+q)*d/2,0.)
    out=np.where(cross,np.maximum(p,q)**2/(2*abs(m)),out)
    return out

def strip_coverage(theta,dx=0.,dy=0.,shape=(256,256),center=(128.,128.),width=20.,height=190.,pixel_indices=None):
    """Exact box coverage by piecewise-affine horizontal boundary integration.

    Each interval between rectangle-vertex y coordinates has exactly two active
    edges. The overlap length with [x,x+1] equals a signed sum of four hinges.
    This is an independent integration algorithm, not polygon/pixel clipping.
    """
    p=vertices(theta,center[0]+dx,center[1]+dy,width,height)
    if pixel_indices is None:
        lo=np.maximum(0,np.floor(p.min(0)).astype(int));hi=np.minimum(shape[::-1],np.ceil(p.max(0)).astype(int))
        yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]; yy=yy.ravel();xx=xx.ravel()
    else:
        yy,xx=np.asarray(pixel_indices).T
    result=np.zeros(len(xx),np.float64)
    for ya,yb in zip(sorted(set(p[:,1]))[:-1],sorted(set(p[:,1]))[1:]):
        if yb-ya<1e-13:continue
        mid=(ya+yb)/2;edges=[]
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            if min(a[1],b[1])<mid<max(a[1],b[1]):
                m=(b[0]-a[0])/(b[1]-a[1]);k=a[0]-m*a[1];edges.append((m,k))
        edges.sort(key=lambda v:v[0]*mid+v[1])
        assert len(edges)==2,(theta,ya,yb,edges)
        ids=np.flatnonzero((yy<yb)&(yy+1>ya))
        l=np.maximum(yy[ids],ya);h=np.minimum(yy[ids]+1,yb)
        (ml,bl),(mr,br)=edges
        x=xx[ids]
        result[ids]+=_hinge_integral(mr,br-x,l,h)-_hinge_integral(ml,bl-x,l,h)-_hinge_integral(mr,br-x-1,l,h)+_hinge_integral(ml,bl-x-1,l,h)
    result=np.clip(result,0,1)
    if pixel_indices is not None:return result
    out=np.zeros(shape,np.float64);out[yy,xx]=result
    return out

def geos_coverage(theta,dx=0.,dy=0.,shape=(256,256),center=(128.,128.),width=20.,height=190.):
    p=vertices(theta,center[0]+dx,center[1]+dy,width,height);poly=Polygon(p)
    lo=np.maximum(0,np.floor(p.min(0)).astype(int));hi=np.minimum(shape[::-1],np.ceil(p.max(0)).astype(int))
    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
    cells=shapely.box(xx,yy,xx+1,yy+1)
    # GEOS robust overlay, double coordinate precision and no precision snapping.
    a=shapely.area(shapely.intersection(cells,poly,grid_size=0))
    out=np.zeros(shape,np.float64);out[yy,xx]=a
    return out

def srgb(y):return np.where(y<=.0031308,12.92*y,1.055*np.maximum(y,0)**(1/2.4)-.055)
def linear(q):return np.where(q<=.04045,q/12.92,((q+.055)/1.055)**2.4)
def encode(a,bits=8):return np.rint(srgb(BG+(FG-BG)*a)*((1<<bits)-1)).astype(np.uint16 if bits>8 else np.uint8)
def decode(q,bits=8):return np.clip((linear(np.asarray(q,dtype=float)/((1<<bits)-1))-BG)/(FG-BG),0,1)
def code_intervals(q,bits=8):
    n=(1<<bits)-1
    return (np.clip((linear(np.maximum(0,np.asarray(q,dtype=float)-.5)/n)-BG)/(FG-BG),0,1),
            np.clip((linear(np.minimum(n,np.asarray(q,dtype=float)+.5)/n)-BG)/(FG-BG),0,1))

def pca_axis(a):
    yy,xx=np.indices(a.shape,dtype=float);xx+=.5;yy+=.5
    z=a.sum();cx=(a*xx).sum()/z;cy=(a*yy).sum()/z
    x=xx-cx;y=yy-cy
    v=np.linalg.eigh([[np.sum(a*x*x)/z,np.sum(a*x*y)/z],[np.sum(a*x*y)/z,np.sum(a*y*y)/z]])[1][:,-1]
    if v[1]<0:v=-v
    return math.degrees(math.atan2(v[0],v[1]))

def legacy_pca_codes(q,bits=8):
    y=linear(np.asarray(q,dtype=float)/((1<<bits)-1))
    z=linear(encode(np.array([0.,1.]),bits).astype(float)/((1<<bits)-1))
    return pca_axis(np.maximum(0,y-z[0])/(z[1]-z[0]))

def edge_axis(a,lower=None,upper=None):
    """Image-only straight-edge fit; never accepts nominal angle or translation.

    PCA chooses whether to transpose only. Rows from the central 60% of the
    nonzero support avoid both short caps for this elongated 20×190 stimulus.
    Each long-edge coordinate is an integral of local row coverage. Fit the two
    edge intercepts and their common slope. Optional code-derived intervals give
    LP bounds under the explicitly checked straight-edge/box-filter model.
    """
    init=pca_axis(a);transpose=abs(init)>45
    if transpose:
        a=a.T
        if lower is not None:lower,upper=lower.T,upper.T
    support=np.flatnonzero(np.any(a>1e-12,axis=1))
    if len(support)<10:return {'status':'insufficient_rows'}
    lo,hi=support[0],support[-1]+1;low=lo+int(.2*(hi-lo));high=hi-int(.2*(hi-lo))
    positions=[];design=[];intervals=[];invalid=0
    for y in range(low,high):
        row=a[y];occupied=np.flatnonzero(row>1e-12);full=np.flatnonzero(row>.98)
        if len(full)<5 or len(occupied)<5:invalid+=1;continue
        # Three-pixel guards contain the <=1-pixel-per-row long-edge motion.
        j=max(0,occupied[0]-2);k=min(full[0]+2,int(np.mean(full)))
        j2=max(full[-1]-1,int(np.mean(full)));k2=min(len(row),occupied[-1]+3)
        for side,start,stop in [(0,j,k),(1,j2,k2)]:
            if stop<=start:invalid+=1;continue
            if side==0:
                position=stop-row[start:stop].sum()
                interval=None if lower is None else (stop-upper[y,start:stop].sum(),stop-lower[y,start:stop].sum())
            else:
                position=start+row[start:stop].sum()
                interval=None if lower is None else (start+lower[y,start:stop].sum(),start+upper[y,start:stop].sum())
            positions.append(position);design.append([y+.5-(low+high)/2,1. if side==0 else 0.,1. if side==1 else 0.])
            if interval is not None:intervals.append(interval)
    X=np.asarray(design);z=np.asarray(positions)
    if len(z)<10:return {'status':'insufficient_edges'}
    coef=np.linalg.lstsq(X,z,rcond=None)[0];m=coef[0]
    def axis(m):return ((math.degrees(math.atan2(1,m)) if transpose else math.degrees(math.atan(m)))+90)%180-90
    result={'status':'ok','axis_deg':axis(m),'initial_image_pca_deg':init,'transpose':transpose,
            'edge_rows':len(z)//2,'edge_fit_rms_px':float(np.sqrt(np.mean((X@coef-z)**2))),
            'edge_fit_max_abs_px':float(np.max(np.abs(X@coef-z))),'skipped_rows':invalid}
    if intervals:
        L,U=np.asarray(intervals).T
        A=np.vstack([X,-X]);b=np.r_[U,-L]
        fits=[linprog(c,A_ub=A,b_ub=b,bounds=[(-1.01,1.01),(None,None),(None,None)],method='highs') for c in ([1.,0.,0.],[-1.,0.,0.])]
        feasible=all(x.success for x in fits)
        result['quantization_interval_feasible']=feasible
        if feasible:
            bounds=sorted([axis(x.x[0]) for x in fits])
            # Axis wraps at ±90; express local offsets around the image estimate.
            offsets=sorted([((v-result['axis_deg']+90)%180-90) for v in bounds])
            result.update({'axis_interval_low_deg':result['axis_deg']+offsets[0],
                'axis_interval_high_deg':result['axis_deg']+offsets[1],
                'axis_interval_width_deg':offsets[1]-offsets[0]})
        else:result['interval_failure']=[x.message for x in fits]
    return result

def quantized_edge_axis(q,bits=8):
    """Image-only nonlinear half-plane fit plus exact code-bin feasibility.

    Each selected pixel constrains one straight edge. At fixed slope the box
    half-plane integral is analytically invertible, giving an interval for the
    intercept. Intersect all those intervals separately for the two long edges.
    The final angular feasible interval is conditional on this declared model;
    neither true theta nor true subpixel center is accepted by this function.
    """
    a=decode(q,bits);L,U=code_intervals(q,bits);initial=edge_axis(a,L,U)
    if initial.get('status')!='ok':return initial
    transpose=initial['transpose']
    if transpose:a,L,U,q=a.T,L.T,U.T,q.T
    support=np.flatnonzero(np.any(a>1e-12,axis=1));lo,hi=support[0],support[-1]+1
    low=lo+int(.2*(hi-lo));high=hi-int(.2*(hi-lo));yc=(low+high)/2
    xs=[];ys=[];sides=[];obs=[];lower=[];upper=[]
    for y in range(low,high):
        occupied=np.flatnonzero(a[y]>1e-12);full=np.flatnonzero(a[y]>.98)
        if len(full)<5:continue
        windows=[(0,max(0,occupied[0]-2),full[0]+2),(1,full[-1]-1,min(a.shape[1],occupied[-1]+3))]
        for side,start,stop in windows:
            for x in range(start,stop):
                xs.append(x);ys.append(y+.5-yc);sides.append(side);obs.append(q[y,x]/((1<<bits)-1));lower.append(L[y,x]);upper.append(U[y,x])
    x=np.asarray(xs);y=np.asarray(ys);side=np.asarray(sides);obs=np.asarray(obs);pl=np.asarray(lower);pu=np.asarray(upper)
    def coverage(par):
        m,b0,b1=par;z=m*y+np.where(side==0,b0,b1)
        # Symmetric row integration of one half plane, evaluated with local
        # coordinates, avoids loss of precision from absolute image positions.
        t=np.where(side==0,x+1-z,z-x);h=abs(m)/2
        if h<1e-9:return np.clip(t,0,1)
        v=2*h;small=min(1.,v);large=max(1.,v)
        f=np.where(t<=-h,0,np.where(t<small-h,(t+h)**2/(2*v),
            np.where(t<=large-h,(t+(large-1)/2)/large,
            np.where(t<1+h,1-(1+h-t)**2/(2*v),1))))
        return np.clip(f,0,1)
    ma=math.tan(math.radians(initial['axis_deg'])) if not transpose else 1/math.tan(math.radians(initial['axis_deg']))
    # Read both intercepts from the observed edge-coordinate fit, without GT.
    zs=[];X=[]
    for sid in (0,1):
        ids=np.flatnonzero((side==sid)&(decode(np.rint(obs*((1<<bits)-1)).astype(np.uint16),bits)>.01)&(decode(np.rint(obs*((1<<bits)-1)).astype(np.uint16),bits)<.99))
        aa=decode(np.rint(obs*((1<<bits)-1)).astype(np.uint16),bits)
        pos=np.where(sid==0,x+1-aa,x+aa)
        zs.append(float(np.median(pos[ids]-ma*y[ids])) if len(ids) else float(np.median(pos[side==sid]-ma*y[side==sid])))
    fit=least_squares(lambda p:srgb(BG+(FG-BG)*coverage(p))-obs,[ma,*zs],bounds=([-1.01,-np.inf,-np.inf],[1.01,np.inf,np.inf]),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=100)
    def axis(m):return ((math.degrees(math.atan2(1,m)) if transpose else math.degrees(math.atan(m)))+90)%180-90
    def inverse_area(p,m):
        v=abs(m)
        if v<1e-12:return p.copy()
        small=min(1.,v);large=max(1.,v);tail=small/(2*large)
        return np.where(p<tail,np.sqrt(np.maximum(0,2*v*p))-v/2,
            np.where(p<=1-tail,large*p+(1-large)/2,
            1+v/2-np.sqrt(np.maximum(0,2*v*(1-p)))))
    def slack(m):
        invlo=inverse_area(pl,m);invhi=inverse_area(pu,m)
        bl=np.where(side==0,x+1-invhi,x+invlo)-m*y
        bu=np.where(side==0,x+1-invlo,x+invhi)-m*y
        bl=np.where(np.where(side==0,pu>=1,pl<=0),-np.inf,bl)
        bu=np.where(np.where(side==0,pl<=0,pu>=1),np.inf,bu)
        return max(float(bl[side==s].max()-bu[side==s].min()) for s in (0,1))
    m=float(fit.x[0]);result={'status':'ok','axis_deg':axis(m),'initial_image_edge_deg':initial['axis_deg'],
        'transpose':transpose,'pixels':len(x),'edge_rows':initial['edge_rows'],
        'sRGB_code_fit_rms':float(np.sqrt(np.mean((srgb(BG+(FG-BG)*coverage(fit.x))-obs)**2))*((1<<bits)-1)),
        'sRGB_code_fit_max_abs':float(np.max(abs(srgb(BG+(FG-BG)*coverage(fit.x))-obs))*((1<<bits)-1)),
        'optimizer_success':bool(fit.success),'quantization_interval_feasible':False}
    # Outer bounds are the earlier conservative row-integral LP, independent of
    # this nonlinear refinement. At most ±.1 slope is searched if LP mismatches.
    if initial.get('quantization_interval_feasible'):
        aa=[initial['axis_interval_low_deg'],initial['axis_interval_high_deg']]
        mm=sorted([math.tan(math.radians(v)) if not transpose else 1/math.tan(math.radians(v)) for v in aa])
        ml,mh=max(-1.01,mm[0]),min(1.01,mm[1])
    else:ml,mh=max(-1.01,m-.1),min(1.01,m+.1)
    # Search a slope delta around the image-only fit. Bounded minimization on
    # an absolute slope near .5 otherwise stops at sqrt(machine_eps)*.5 even
    # when xatol is tighter; 16-bit feasible intervals can be narrower than it.
    opt=minimize_scalar(lambda d:slack(m+d),bounds=(ml-m,mh-m),method='bounded',options={'xatol':1e-15,'maxiter':200})
    mid=m+float(opt.x)
    if slack(m)<slack(mid):mid=m
    result['minimum_intercept_interval_gap_px']=float(slack(mid))
    if slack(mid)<=1e-9:
        tol=1e-10
        left=ml if slack(ml)<=tol else brentq(lambda z:slack(z)-tol,ml,mid,xtol=1e-15)
        right=mh if slack(mh)<=tol else brentq(lambda z:slack(z)-tol,mid,mh,xtol=1e-15)
        off=sorted([((axis(v)-result['axis_deg']+90)%180-90) for v in (left,right)])
        result.update({'quantization_interval_feasible':True,'axis_interval_low_deg':result['axis_deg']+off[0],
            'axis_interval_high_deg':result['axis_deg']+off[1],'axis_interval_width_deg':off[1]-off[0],
            'feasible_midpoint_axis_deg':axis((left+right)/2)})
    return result

def _json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def _csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def freeze(out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        retained=json.loads((out/'manifest.json').read_text())
        if retained.get('configuration_count')!=800:raise ValueError('Retained manifest is not the registered 800-configuration version; choose a new --output.')
        return retained
    manifest={'version':'v30-math-1','frozen_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'nominal_abs_degrees':ANGLES,'signs_nonzero_except_axis_duplicate_90':[-1,1],'control_and_90_signs':[1],
        'signed_angle_degrees':[a*s for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1])],
        'subpixel_dx_px':OFFSETS,'subpixel_dy_px':OFFSETS,'configuration_count':(2*(len(ANGLES)-2)+2)*16,
        'fixed_primary':{'source_frame_wh':[768,512],'analysis_patch_wh':[256,256],'bar_wh_px':[20,190],'linear_BG':BG,'linear_FG':FG},
        'reference_methods':['historical_float64_polygon_clip_readonly','independent_GE0S_overlay_double','independent_analytic_horizontal_strips','mpmath80_boundary_samples'],
        'image_estimators':['historical_inverse_sRGB_weighted_pixel_center_PCA','image_only_shared_slope_long_edge_fit','code_bin_interval_linear_program','image_only_exact_box_halfplane_sRGB_fit','pixelwise_code_bin_intercept_feasibility'],
        'bit_depths':[8,10,16,'linear_float64'],'primary_tolerance_deg':.001,
        'estimator_prohibition':'No nominal theta, true center, or renderer polygon is passed to edge_axis; only image coverage or decoded codes with fixed luminance calibration.',
        'interval_scope':'Conditional feasible range under straight long-edge + exact box pixel filter + known linear BG/FG + nearest quantization model. Not a distribution-free confidence interval.',
        'resize_primary':{'from_wh':[768,512],'official_Nemo_to_wh':[608,416],'sx':608/768,'sy':416/512,'ratio_sx_sy':38/39,'axis_formula':'atan2((38/39)*sin(theta),cos(theta))'},
        'native_candidates':[{'canvas_wh':[512,512],'bar_wh_px':[20,190],'source_aspect_exact':True,'change':'source canvas/aspect and relative field of view; bar remains 20x190 source pixels','actual_processor_check_required':True}, {'canvas_wh':[768,768],'bar_wh_px':[20,190],'source_aspect_exact':True,'change':'source canvas/aspect and relative field of view; bar remains 20x190 source pixels','actual_processor_check_required':True}],
        'runtime':{'python':sys.executable,'numpy':np.__version__,'shapely':shapely.__version__,'GEOS':shapely.geos_version_string,'platform':platform.platform()},
        'sources':[{'title':'Catmull 1978 area filtering original paper','url':'https://people.csail.mit.edu/ericchan/bib/pdf/p6-catmull.pdf','use':'pixel area box filtering mathematical reference'},
        {'title':'Shapely intersection official documentation','url':'https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html','use':'independent GEOS overlay, grid_size=0 double precision'},
        {'title':'Shapely area official documentation','url':'https://shapely.readthedocs.io/en/stable/reference/shapely.area.html','use':'polygon area measurement'},
        {'title':'SciPy linprog official documentation','url':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html','use':'conditional code-interval feasible slope bounds'}],
        'finite_scope':'Finite representative exact-area, GEOS, SS/filter, bit-depth, and resize comparison; no claim of exhaustive internet search.'}
    _json(out/'manifest.json',manifest)
    return manifest

def mp_pixel(theta,dx,dy,y,x,dps=80):
    """80-digit horizontal-strip integral from decimal theta/offset inputs."""
    import mpmath as mp
    with mp.workdps(dps):
        t=mp.mpf(str(theta))*mp.pi/180;c=mp.cos(t);s=mp.sin(t)
        cx=mp.mpf(128)+mp.mpf(str(dx));cy=mp.mpf(128)+mp.mpf(str(dy))
        p=[(cx+c*u+s*v,cy-s*u+c*v) for u,v in [(-10,-95),(10,-95),(10,95),(-10,95)]]
        vy=sorted(set(v[1] for v in p));total=mp.mpf(0)
        def H(m,b,a,z):
            q=m*a+b;r=m*z+b
            if abs(m)<mp.mpf('1e-70'):return max(q,0)*(z-a)
            if q>=0 and r>=0:return (q+r)*(z-a)/2
            if q*r<0:return max(q,r)**2/(2*abs(m))
            return mp.mpf(0)
        for ya,yb in zip(vy[:-1],vy[1:]):
            a=max(ya,mp.mpf(y));b=min(yb,mp.mpf(y+1))
            if b<=a:continue
            mid=(ya+yb)/2;edges=[]
            for v,w in zip(p,p[1:]+p[:1]):
                if min(v[1],w[1])<mid<max(v[1],w[1]):
                    m=(w[0]-v[0])/(w[1]-v[1]);k=v[0]-m*v[1];edges.append((m,k))
            edges.sort(key=lambda v:v[0]*mid+v[1])
            (ml,bl),(mr,br)=edges
            total+=H(mr,br-x,a,b)-H(ml,bl-x,a,b)-H(mr,br-x-1,a,b)+H(ml,bl-x-1,a,b)
        return mp.nstr(total,70)

def run_validation(out=OUT):
    if (out/'summary.json').exists():raise FileExistsError('Retained CPU validation exists; use --output for a new run.')
    out.mkdir(parents=True,exist_ok=True);manifest=freeze(out);started=time.monotonic()
    configurations=[(a*s,dx,dy) for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1]) for dx in OFFSETS for dy in OFFSETS]
    assert len(configurations)==len(set(configurations))==manifest['configuration_count']==800
    _csv(out/'configurations.csv',[{'configuration_id':i,'theta_deg':a,'dx_px':dx,'dy_px':dy,'nonzero_scored':a!=0} for i,(a,dx,dy) in enumerate(configurations)])
    oldpath=ROOT/'src/stimuli/validate_t3_renderer.py'
    spec=importlib.util.spec_from_file_location('historical_v27_readonly',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    geometry=[];measurements=[];highprec=[];saved={}
    for i,(theta,dx,dy) in enumerate(configurations):
        tick=time.monotonic();a=strip_coverage(theta,dx,dy);g=geos_coverage(theta,dx,dy);h=old.area_coverage(theta,dx,dy)
        row={'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,'area_strip_px2':float(a.sum()),
            'strip_area_error_px2':float(a.sum()-3800),'GEOS_area_error_px2':float(g.sum()-3800),
            'historical_clip_area_error_px2':float(h.sum()-3800),'max_abs_strip_vs_GEOS':float(np.max(abs(a-g))),
            'max_abs_strip_vs_historical_clip':float(np.max(abs(a-h))),
            'strip_float_PCA_deg':pca_axis(a),'GEOS_float_PCA_deg':pca_axis(g),'historical_float_PCA_deg':pca_axis(h)}
        for key in ['strip_float_PCA','GEOS_float_PCA','historical_float_PCA']:
            row[key+'_error_deg']=angle_error(row[key+'_deg'],theta)
        geometry.append(row)
        boundary=np.argwhere((g>1e-10)&(g<1-1e-10))
        if len(boundary):
            values=g[boundary[:,0],boundary[:,1]]
            inds=list(dict.fromkeys([int(np.argmin(values)),int(np.argmax(values)),int(np.argmin(abs(values-.5)))]))
            for index in inds:
                yy,xx=map(int,boundary[index]);v=mp_pixel(theta,dx,dy,yy,xx)
                highprec.append({'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,'y':yy,'x':xx,
                    'mpmath80_coverage':v,'GEOS_coverage':g[yy,xx],'strip_coverage':a[yy,xx],
                    'abs_GEOS_vs_mp80':abs(g[yy,xx]-float(v)),'abs_strip_vs_mp80':abs(a[yy,xx]-float(v))})
        for bits in [8,10,16]:
            q=encode(a,bits);dec=decode(q,bits);low,upper=code_intervals(q,bits)
            edge=edge_axis(dec,low,upper);qe=quantized_edge_axis(q,bits)
            r={'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,'encoding':f'sRGB{bits}',
                'legacy_PCA_deg':legacy_pca_codes(q,bits),'calibrated_PCA_deg':pca_axis(dec),
                'codes_sha256':hashlib.sha256(q.tobytes()).hexdigest()}
            for key in ['legacy_PCA','calibrated_PCA']:r[key+'_error_deg']=angle_error(r[key+'_deg'],theta)
            for prefix,e in [('row_edge',edge),('pixel_halfplane',qe)]:
                for key,v in e.items():r[prefix+'_'+key]=v
                if e.get('status')=='ok':r[prefix+'_error_deg']=angle_error(e['axis_deg'],theta)
                if e.get('quantization_interval_feasible'):
                    tt=e['axis_deg']+((theta-e['axis_deg']+90)%180-90)
                    r[prefix+'_interval_contains_nominal']=e['axis_interval_low_deg']-1e-8<=tt<=e['axis_interval_high_deg']+1e-8
                    if 'feasible_midpoint_axis_deg' in e:r[prefix+'_feasible_midpoint_error_deg']=angle_error(e['feasible_midpoint_axis_deg'],theta)
            measurements.append(r)
        edge=edge_axis(a)
        measurements.append({'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,'encoding':'linear_float64',
            'legacy_PCA_deg':pca_axis(a),'legacy_PCA_error_deg':angle_error(pca_axis(a),theta),
            **{'row_edge_'+k:v for k,v in edge.items()},'row_edge_error_deg':angle_error(edge['axis_deg'],theta)})
        if dx==dy==0 and theta in [0,.05,.25,.75,3,32,45,90]:
            saved[f'alpha_strip_{theta:g}']=a;saved[f'RGB8_{theta:g}']=encode(a)
        if (i+1)%32==0 or i+1==len(configurations):
            _json(out/'progress.json',{'complete_configurations':i+1,'total_configurations':len(configurations),'elapsed_seconds':time.monotonic()-started,'last_configuration_seconds':time.monotonic()-tick})
            print(json.dumps({'complete_configurations':i+1,'total':len(configurations),'elapsed_s':round(time.monotonic()-started,2)}),flush=True)
    _csv(out/'geometry_comparison.csv',geometry);_csv(out/'image_measurements.csv',measurements);_csv(out/'mpmath80_pixel_samples.csv',highprec)
    np.savez_compressed(out/'representative_arrays.npz',**saved)
    summaries=[]
    for encoding in ['sRGB8','sRGB10','sRGB16','linear_float64']:
        for angle in ANGLES:
            rr=[r for r in measurements if r['encoding']==encoding and abs(r['theta_deg'])==angle]
            summary={'encoding':encoding,'abs_theta_deg':angle,'configurations':len(rr)}
            for key in ['legacy_PCA_error_deg','calibrated_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','pixel_halfplane_feasible_midpoint_error_deg','pixel_halfplane_axis_interval_width_deg']:
                valid=[r for r in rr if key in r]
                if valid:
                    worst=max(valid,key=lambda r:r[key]);summary[key+'_max']=worst[key]
                    summary[key+'_worst_configuration_id']=worst['configuration_id']
            summary['pixel_halfplane_intervals_infeasible']=sum(r.get('pixel_halfplane_quantization_interval_feasible') is False for r in rr)
            summary['pixel_halfplane_intervals_excluding_nominal']=sum(r.get('pixel_halfplane_interval_contains_nominal') is False for r in rr)
            summaries.append(summary)
    _csv(out/'per_angle_worst_offset.csv',summaries)
    resize=[]
    for theta in [a*s for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1])]:
        t=math.radians(theta);native=math.degrees(math.atan2((38/39)*math.sin(t),math.cos(t)))
        resize.append({'nominal_source_theta_deg':theta,'official_768x512_to_608x416_continuous_axis_deg':native,
            'anisotropic_axis_error_deg':angle_error(native,theta),'sx':608/768,'sy':416/512,'sx_over_sy':38/39})
    _csv(out/'official_resize_analytic_distortion.csv',resize)
    report={'status':'complete','configuration_count':len(configurations),'image_measurement_rows':len(measurements),'mpmath80_pixel_samples':len(highprec),
        'max_abs_strip_vs_GEOS':max(r['max_abs_strip_vs_GEOS'] for r in geometry),'max_abs_strip_vs_historical_clip':max(r['max_abs_strip_vs_historical_clip'] for r in geometry),
        'max_abs_strip_area_error_px2':max(abs(r['strip_area_error_px2']) for r in geometry),'max_abs_GEOS_vs_mp80':max(r['abs_GEOS_vs_mp80'] for r in highprec),
        'historical_v27_sha256':hashlib.sha256(oldpath.read_bytes()).hexdigest(),'reference_module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started,
        'separation':'Continuous rectangle/box coverage agreement validates generation geometry. Pixel-center PCA has sampling bias; RGB code quantization adds error. Image-only edge fits and conditional feasible intervals are reported separately and never replace old PCA.',
        'limits':'This sub-run does not execute a VLM or GPU. Official native-processor tensor/resize evidence belongs to GPU sub-run. Higher-bit encodings are numerical outputs, not an assertion that RGB8 video/model input retains them.'}
    _json(out/'summary.json',report);print(json.dumps(report),flush=True)

def run_supplement(out=OUT):
    """Separately registered finite size and post-filter interventions."""
    if (out/'secondary_summary.json').exists():raise FileExistsError('Retained secondary validation exists; use --output for a new run.')
    from scipy.ndimage import gaussian_filter
    configurations=[(a*s,dx,dy) for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1]) for dx in [0.,.5] for dy in [0.,.5]]
    _json(out/'secondary_manifest.json',{'kind':'secondary changes to source stimulus, not a correction of primary data',
        'primary_manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
        'source_canvas_wh':[768,512],'luminance_BG_FG':[BG,FG],
        'size_interventions':[{'width_px':24,'height_px':228,'scale':1.2,'analysis_patch_wh':[256,256]}, {'width_px':40,'height_px':380,'scale':2.,'analysis_patch_wh':[512,512]}],
        'configurations_per_size':len(configurations),'angles_and_signs':'same 50 unique signed orientations as primary',
        'subpixel_dx_dy_cartesian':[0.,.5],'encoding_bits':[8,10,16,'linear_float64'],
        'source_placement':'isolated bar, source center (384,256); a translated analysis patch is used without clipping the geometry; no paired-display/model claim',
        'filter_interventions':[{'sigma_native_px':0.},{'sigma_native_px':.5},{'sigma_native_px':1.}],
        'filter_scope':'all 800 primary geometries; Gaussian post-filter changes the spatial intensity profile. Float and FINAL RGB8 PCA are measured separately. No dither or code tuning.'})
    started=time.monotonic();rows=[]
    for width,height,shape,center in [(24,228,(256,256),(128.,128.)),(40,380,(512,512),(256.,256.))]:
        for i,(theta,dx,dy) in enumerate(configurations):
            a=strip_coverage(theta,dx,dy,shape=shape,center=center,width=width,height=height)
            for bits in [8,10,16]:
                q=encode(a,bits);qe=quantized_edge_axis(q,bits);pca=legacy_pca_codes(q,bits)
                r={'width_px':width,'height_px':height,'scale':width/20,'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,'encoding':f'sRGB{bits}',
                    'legacy_PCA_deg':pca,'legacy_PCA_error_deg':angle_error(pca,theta),**{'pixel_halfplane_'+k:v for k,v in qe.items()}}
                if qe.get('status')=='ok':r['pixel_halfplane_error_deg']=angle_error(qe['axis_deg'],theta)
                if qe.get('quantization_interval_feasible'):
                    tt=qe['axis_deg']+((theta-qe['axis_deg']+90)%180-90)
                    r['pixel_halfplane_interval_contains_nominal']=qe['axis_interval_low_deg']-1e-8<=tt<=qe['axis_interval_high_deg']+1e-8
                    r['pixel_halfplane_feasible_midpoint_error_deg']=angle_error(qe['feasible_midpoint_axis_deg'],theta)
                rows.append(r)
            e=edge_axis(a);rows.append({'width_px':width,'height_px':height,'scale':width/20,'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,
                'encoding':'linear_float64','legacy_PCA_deg':pca_axis(a),'legacy_PCA_error_deg':angle_error(pca_axis(a),theta),
                'row_edge_deg':e['axis_deg'],'row_edge_error_deg':angle_error(e['axis_deg'],theta)})
        print(json.dumps({'secondary_size_complete':[width,height],'configurations':len(configurations),'elapsed_s':round(time.monotonic()-started,2)}),flush=True)
    _csv(out/'secondary_size_measurements.csv',rows)
    size_summary=[]
    for width in [24,40]:
        for encoding in ['sRGB8','sRGB10','sRGB16','linear_float64']:
            rr=[r for r in rows if r['width_px']==width and r['encoding']==encoding];r={'width_px':width,'height_px':width*9.5,'encoding':encoding,'configurations':len(rr)}
            for key in ['legacy_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','pixel_halfplane_feasible_midpoint_error_deg','pixel_halfplane_axis_interval_width_deg']:
                valid=[x for x in rr if key in x]
                if valid:
                    w=max(valid,key=lambda x:x[key]);r[key+'_max']=w[key];r[key+'_worst_theta']=w['theta_deg'];r[key+'_worst_dx']=w['dx_px'];r[key+'_worst_dy']=w['dy_px']
            r['intervals_infeasible']=sum(x.get('pixel_halfplane_quantization_interval_feasible') is False for x in rr)
            r['intervals_excluding_nominal']=sum(x.get('pixel_halfplane_interval_contains_nominal') is False for x in rr)
            size_summary.append(r)
    _csv(out/'secondary_size_summary.csv',size_summary)
    filters=[]
    primary=[(a*s,dx,dy) for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1]) for dx in OFFSETS for dy in OFFSETS]
    for i,(theta,dx,dy) in enumerate(primary):
        a=strip_coverage(theta,dx,dy)
        for sigma in [0.,.5,1.]:
            f=a if sigma==0 else gaussian_filter(a,sigma=sigma,mode='constant',cval=0,truncate=4.)
            pa=pca_axis(f);pr=legacy_pca_codes(encode(f))
            filters.append({'configuration_id':i,'theta_deg':theta,'dx_px':dx,'dy_px':dy,'gaussian_sigma_native_px':sigma,
                'float_PCA_deg':pa,'float_PCA_error_deg':angle_error(pa,theta),'FINAL_RGB8_PCA_deg':pr,'FINAL_RGB8_PCA_error_deg':angle_error(pr,theta),
                'area_error_px2':float(f.sum()-3800),'spatial_profile_changed':sigma!=0})
    _csv(out/'filter_measurements.csv',filters)
    fsum=[]
    for sigma in [0.,.5,1.]:
        rr=[r for r in filters if r['gaussian_sigma_native_px']==sigma];r={'sigma_native_px':sigma,'configurations':len(rr)}
        for key in ['float_PCA_error_deg','FINAL_RGB8_PCA_error_deg']:
            w=max(rr,key=lambda x:x[key]);r[key+'_max']=w[key];r[key+'_worst_theta']=w['theta_deg'];r[key+'_worst_dx']=w['dx_px'];r[key+'_worst_dy']=w['dy_px'];r[key+'_above_001']=sum(x[key]>.001 for x in rr)
        fsum.append(r)
    _csv(out/'filter_summary.csv',fsum)
    resize=[]
    for theta in [a*s for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1])]:
        t=math.radians(theta);sx,sy=608/768,416/512
        major=np.array([math.sin(t),math.cos(t)]);minor=np.array([math.cos(t),-math.sin(t)])
        cov=(190**2*np.outer(major,major)+20**2*np.outer(minor,minor))/12
        aff=np.diag([sx,sy]);v=np.linalg.eigh(aff@cov@aff)[1][:,-1]
        if v[1]<0:v=-v
        edge=math.degrees(math.atan2(sx*major[0],sy*major[1]));pc=math.degrees(math.atan2(v[0],v[1]))
        resize.append({'nominal_source_theta_deg':theta,'long_edge_after_768x512_to_608x416_deg':edge,
            'continuous_covariance_axis_after_affine_deg':pc,'long_edge_error_from_source_label_deg':angle_error(edge,theta),
            'covariance_axis_error_from_source_label_deg':angle_error(pc,theta),'covariance_vs_long_edge_difference_deg':angle_error(pc,edge)})
    _csv(out/'official_resize_distinct_axis_targets.csv',resize)
    _json(out/'secondary_summary.json',{'status':'complete','sizes':size_summary,'filters':fsum,'elapsed_seconds':time.monotonic()-started,
        'reference_module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'Secondary bar-size or filtering effects are interventions. Primary 20x190 RGB8 results remain unchanged. No actual processor/native equivalence is inferred from these calculations.'})
    print(json.dumps({'secondary_status':'complete','elapsed_seconds':time.monotonic()-started,'sizes':size_summary,'filters':fsum}),flush=True)

def repair_boundary(out=OUT):
    """Re-evaluate ±45° and rows with a recorded numerical interval failure."""
    if (out/'estimator_boundary_repair.json').exists() and json.loads((out/'estimator_boundary_repair.json').read_text()).get('previous_failed_rows'):
        raise FileExistsError('Retained numerical-repair evidence exists; do not overwrite it.')
    evidence={'reason':'Generalize the box half-plane CDF/inverse across |slope|=1, and recenter bounded feasible-slope optimization to an image-derived slope delta to avoid its relative sqrt(machine_eps)*absolute_slope stopping limit. Four original secondary16bit failures at ±32/±24 degrees are numerical optimizer precision failures, not image/model inconsistencies. No stimulus pixels or nominal labels change.',
        'previous_primary_module_sha256':json.loads((out/'summary.json').read_text())['reference_module_sha256'],
        'new_module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'repaired_rows':0,'previous_failed_rows':[]}
    for filename,secondary in [('image_measurements.csv',False),('secondary_size_measurements.csv',True)]:
        rr=list(csv.DictReader((out/filename).open()))
        for r in rr:
            if (abs(float(r['theta_deg']))!=45 and r.get('pixel_halfplane_quantization_interval_feasible')!='False') or r['encoding']=='linear_float64':continue
            if r.get('pixel_halfplane_quantization_interval_feasible')=='False':evidence['previous_failed_rows'].append(dict(r))
            theta=float(r['theta_deg']);dx=float(r['dx_px']);dy=float(r['dy_px']);bits=int(r['encoding'][4:])
            width=int(r['width_px']) if secondary else 20;height=width*9.5;n=512 if width==40 else 256
            a=strip_coverage(theta,dx,dy,shape=(n,n),center=(n/2,n/2),width=width,height=height);qe=quantized_edge_axis(encode(a,bits),bits)
            for key in list(r):
                if key.startswith('pixel_halfplane_'):r.pop(key)
            for key,v in qe.items():r['pixel_halfplane_'+key]=v
            r['pixel_halfplane_error_deg']=angle_error(qe['axis_deg'],theta)
            if qe.get('quantization_interval_feasible'):
                tt=qe['axis_deg']+((theta-qe['axis_deg']+90)%180-90)
                r['pixel_halfplane_interval_contains_nominal']=qe['axis_interval_low_deg']-1e-8<=tt<=qe['axis_interval_high_deg']+1e-8
                r['pixel_halfplane_feasible_midpoint_error_deg']=angle_error(qe['feasible_midpoint_axis_deg'],theta)
            evidence['repaired_rows']+=1
        _csv(out/filename,rr)
        summaries=[]
        for width in ([24,40] if secondary else [20]):
            for encoding in ['sRGB8','sRGB10','sRGB16','linear_float64']:
                for angle in ([None] if secondary else ANGLES):
                    ss=[r for r in rr if r['encoding']==encoding and (not secondary or int(r['width_px'])==width) and (angle is None or abs(float(r['theta_deg']))==angle)]
                    s={'encoding':encoding,'configurations':len(ss)}
                    if secondary:s.update({'width_px':width,'height_px':width*9.5})
                    else:s['abs_theta_deg']=angle
                    for key in ['legacy_PCA_error_deg','calibrated_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','pixel_halfplane_feasible_midpoint_error_deg','pixel_halfplane_axis_interval_width_deg']:
                        valid=[r for r in ss if str(r.get(key,'')) not in ('','None')]
                        if valid:
                            worst=max(valid,key=lambda r:float(r[key]));s[key+'_max']=float(worst[key]);s[key+'_worst_configuration_id']=int(worst['configuration_id'])
                            if secondary:
                                for k in ['theta_deg','dx_px','dy_px']:s[key+'_worst_'+k]=float(worst[k])
                    s['pixel_halfplane_intervals_infeasible']=sum(str(r.get('pixel_halfplane_quantization_interval_feasible'))=='False' for r in ss)
                    s['pixel_halfplane_intervals_excluding_nominal']=sum(str(r.get('pixel_halfplane_interval_contains_nominal'))=='False' for r in ss)
                    summaries.append(s)
        _csv(out/('secondary_size_summary.csv' if secondary else 'per_angle_worst_offset.csv'),summaries)
    _json(out/'estimator_boundary_repair.json',evidence)
    for name in ['summary.json','secondary_summary.json']:
        data=json.loads((out/name).read_text());data['estimator_boundary_repair']='estimator_boundary_repair.json';data['final_estimator_module_sha256']=evidence['new_module_sha256']
        if name=='secondary_summary.json':data['sizes']=list(csv.DictReader((out/'secondary_size_summary.csv').open()))
        _json(out/name,data)
    print(json.dumps({'status':'boundary_repair_complete','repaired_rows':evidence['repaired_rows'],'preserved_initial_failed_rows':len(evidence['previous_failed_rows'])}),flush=True)

def cairo_coverage(theta):
    import cairocffi as cairo
    surface=cairo.ImageSurface(cairo.FORMAT_A8,256,256);ctx=cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST);ctx.set_operator(cairo.OPERATOR_SOURCE);ctx.set_source_rgba(1,1,1,1)
    p=vertices(theta);ctx.move_to(*p[0])
    for point in p[1:]:ctx.line_to(*point)
    ctx.close_path();ctx.fill();surface.flush()
    alpha=np.frombuffer(surface.get_data(),dtype=np.uint8).reshape(256,surface.get_stride())[:,:256].copy()
    return alpha.astype(np.float64)/255

def agg_coverage(theta):
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.path import Path as AggPath
    from matplotlib.transforms import Affine2D
    renderer=RendererAgg(256,256,72);renderer.clear();gc=renderer.new_gc()
    gc.set_foreground((1,1,1,1),isRGBA=True);gc.set_antialiased(True);gc.set_snap(False);gc.set_linewidth(0)
    p=vertices(theta);path=AggPath(np.vstack([p,p[0]]),[AggPath.MOVETO,AggPath.LINETO,AggPath.LINETO,AggPath.LINETO,AggPath.CLOSEPOLY])
    renderer.draw_path(gc,path,Affine2D().scale(1,-1).translate(0,256),rgbFace=(1,1,1,1))
    return np.asarray(renderer.buffer_rgba())[:,:,3].copy().astype(np.float64)/255

def run_vectors(out=OUT):
    if (out/'vector_summary.json').exists():raise FileExistsError('Retained vector validation exists; use --output for a new run.')
    import cairocffi as cairo,matplotlib
    configs=[a*s for a in ANGLES for s in ([1] if a in (0.,90.) else [-1,1])]
    _json(out/'vector_manifest.json',{'primary_manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
        'configurations_per_method':50,'signed_angles':configs,'offset_dx_dy':[0,0],'bar_wh_px':[20,190],
        'methods':{'cairo_A8_best':{'cairocffi_version':cairo.__version__,'libcairo_version':cairo.cairo_version_string(),'surface':'FORMAT_A8','antialias':'ANTIALIAS_BEST','operator':'SOURCE','coverage_bits':8},
        'matplotlib_Agg_path':{'matplotlib_version':matplotlib.__version__,'renderer':'RendererAgg','fill_path':True,'antialiased':True,'snap':False,'linewidth':0,'coverage_bits':8,'canvas_transform':'x unchanged, y downward→Agg upward via scale(1,-1).translate(0,256)'}},
        'composition':'Native transparent 8-bit alpha→linear BG .1/FG .82→sRGB8. No RGB/gamma blending inside either rasterizer.',
        'interpretation':'Native AA implementations do not promise the exact box area filter. Pixel differences and box-model infeasibility are measured filter/model differences, not by themselves an implementation error.',
        'sources':[{'url':'https://www.cairographics.org/manual/cairo-cairo-t.html#cairo-antialias-t','use':'Cairo antialias mode quality setting'},{'url':'https://matplotlib.org/stable/api/backend_agg_api.html','use':'Agg renderer'},{'url':'https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.GraphicsContextBase.set_snap','use':'disable pixel snapping'}]})
    rows=[];arrays={}
    for method,fun in [('cairo_A8_best',cairo_coverage),('matplotlib_Agg_path',agg_coverage)]:
        masks={}
        for theta in configs:
            a=fun(theta);q=encode(a);pca=legacy_pca_codes(q);e=edge_axis(decode(q));qe=quantized_edge_axis(q);ref=strip_coverage(theta);masks[theta]=a
            r={'method':method,'theta_deg':theta,'nominal_abs_deg':abs(theta),'dx_px':0,'dy_px':0,'native_alpha_bits':8,
                'native_alpha_PCA_deg':pca_axis(a),'native_alpha_PCA_error_deg':angle_error(pca_axis(a),theta),
                'native_alpha_area_error_px2':float(a.sum()-3800),'native_alpha_max_abs_vs_exact_box':float(np.max(abs(a-ref))),
                'FINAL_RGB8_PCA_deg':pca,'FINAL_RGB8_PCA_error_deg':angle_error(pca,theta),
                'row_edge_deg':e.get('axis_deg'),'row_edge_error_deg':None if e.get('status')!='ok' else angle_error(e['axis_deg'],theta),
                **{'pixel_halfplane_'+k:v for k,v in qe.items()},'RGB8_sha256':hashlib.sha256(q.tobytes()).hexdigest()}
            if qe.get('status')=='ok':r['pixel_halfplane_error_deg']=angle_error(qe['axis_deg'],theta)
            if theta in [0,.05,.75,3,32,45,90]:arrays[f'{method}_alpha_{theta:g}']=a;arrays[f'{method}_RGB8_{theta:g}']=q
            rows.append(r)
        for r in rows:
            if r['method']!=method:continue
            t=r['theta_deg']
            if -t in masks:r['sign_mirror_max_abs_alpha']=float(np.max(abs(masks[t]-np.fliplr(masks[-t]))))
    _csv(out/'vector_measurements.csv',rows);np.savez_compressed(out/'vector_representative_arrays.npz',**arrays)
    summaries=[]
    for method in ['cairo_A8_best','matplotlib_Agg_path']:
        rr=[r for r in rows if r['method']==method];s={'method':method,'configurations':len(rr)}
        for key in ['native_alpha_PCA_error_deg','FINAL_RGB8_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','sign_mirror_max_abs_alpha']:
            valid=[r for r in rr if r.get(key) is not None]
            if valid:
                w=max(valid,key=lambda r:r[key]);s[key+'_max']=w[key];s[key+'_worst_theta_deg']=w['theta_deg']
        s['native_alpha_area_error_abs_max_px2']=max(abs(r['native_alpha_area_error_px2']) for r in rr)
        s['exact_box_model_interval_infeasible']=sum(r.get('pixel_halfplane_quantization_interval_feasible') is False for r in rr)
        summaries.append(s)
    _csv(out/'vector_summary.csv',summaries);_json(out/'vector_summary.json',{'status':'complete','summaries':summaries,'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(summaries),flush=True)

def write_reports(out=OUT):
    """Small method×angle worst-offset tables and final CPU qualification."""
    tables=[('secondary_size_measurements.csv','secondary_per_angle_worst_offset.csv',['width_px','height_px','encoding'],
             ['legacy_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','pixel_halfplane_feasible_midpoint_error_deg','pixel_halfplane_axis_interval_width_deg']),
            ('filter_measurements.csv','filter_per_angle_worst_offset.csv',['gaussian_sigma_native_px'],['float_PCA_error_deg','FINAL_RGB8_PCA_error_deg']),
            ('vector_measurements.csv','vector_per_angle.csv',['method'],['native_alpha_PCA_error_deg','FINAL_RGB8_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','native_alpha_area_error_px2'])]
    for source,target,groups,metrics in tables:
        rr=list(csv.DictReader((out/source).open()));keys=sorted(set(tuple(r[k] for k in groups)+(abs(float(r['theta_deg'])),) for r in rr))
        sums=[]
        for key in keys:
            ss=[r for r in rr if tuple(r[k] for k in groups)==key[:-1] and abs(float(r['theta_deg']))==key[-1]]
            r={**dict(zip(groups,key[:-1])),'abs_theta_deg':key[-1],'configurations':len(ss)}
            for metric in metrics:
                valid=[s for s in ss if s.get(metric)]
                if not valid:continue
                w=max(valid,key=lambda s:abs(float(s[metric])));r[metric+'_max_abs']=abs(float(w[metric]))
                r[metric+'_worst_theta_deg']=w['theta_deg'];r[metric+'_worst_dx_px']=w['dx_px'];r[metric+'_worst_dy_px']=w['dy_px']
            sums.append(r)
        _csv(out/target,sums)
    rows=list(csv.DictReader((out/'image_measurements.csv').open()));summary=[]
    for enc in ['sRGB8','sRGB10','sRGB16','linear_float64']:
        ss=[r for r in rows if r['encoding']==enc];r={'encoding':enc,'configurations':len(ss)}
        for key in ['legacy_PCA_error_deg','calibrated_PCA_error_deg','row_edge_error_deg','pixel_halfplane_error_deg','pixel_halfplane_feasible_midpoint_error_deg','pixel_halfplane_axis_interval_width_deg']:
            vv=[x for x in ss if x.get(key)]
            if vv:
                w=max(vv,key=lambda x:float(x[key]));r[key+'_max']=float(w[key]);r[key+'_above_001']=sum(float(x[key])>.001 for x in vv)
                r[key+'_worst_configuration']={k:w[k] for k in ['configuration_id','theta_deg','dx_px','dy_px']}
        r['intervals_infeasible']=sum(s.get('pixel_halfplane_quantization_interval_feasible')=='False' for s in ss)
        r['intervals_excluding_nominal']=sum(s.get('pixel_halfplane_interval_contains_nominal')=='False' for s in ss)
        summary.append(r)
    _json(out/'cpu_qualification.json',{'status':'complete','primary':summary,
        'execution_host':'Fourier /home/yang/cvpr2027_benchmark','python':sys.executable,
        'final_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frozen_manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
        'new_packages_in_existing_qy':[{'name':'Shapely','version':shapely.__version__,'installation':'--no-deps; existing NumPy untouched'},{'name':'cairocffi','version':__import__('cairocffi').__version__,'installation':'--no-deps; reused existing cffi/libcairo'}],
        'important_limits':['Native processor evidence is a separate GPU sub-run; analytic resize targets do not substitute for real tensor measurement.',
            'Higher-bit numerical outputs are not claims that a subsequent RGB8 video or model pipeline retains their precision.',
            'Long-edge feasible intervals use selected central rows, not the complete rectangle. Wide intervals do not prove whole-image non-identifiability.',
            'The independent pixel-halfplane point estimator does not change the old PCA results or improve generation geometry.',
            '24x228/40x380 bars and Gaussian post-filters are independent stimulus interventions; original 20x190 remains primary.',
            'Cairo/Agg native AA filters are not asserted to be exact box filters; model infeasibility measures that mismatch.'],
        'outputs_on_mac':'small CSV/JSON metadata and method summaries only; representative numerical NPZ arrays remain on Fourier; no model inference or GPU activity by this branch.'})
    print(json.dumps({'status':'reports_complete','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--supplement',action='store_true');p.add_argument('--repair-boundary',action='store_true');p.add_argument('--vectors',action='store_true');p.add_argument('--report',action='store_true');args=p.parse_args()
    if args.run:run_validation(args.output)
    elif args.supplement:run_supplement(args.output)
    elif args.repair_boundary:repair_boundary(args.output)
    elif args.vectors:run_vectors(args.output)
    elif args.report:write_reports(args.output)
    else:freeze(args.output);print(args.output/'manifest.json')
