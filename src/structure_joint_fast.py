"""B3-only decision acceleration. Same complete-mixture predicate, own threshold.
The positive component lower bound skips only proven-positive sign cells.
Cycle support compilation exactly enumerates public minimal cuts in O(m^2).
"""
import time,math
from fractions import Fraction
import numpy as np
from .structure_joint import StructureJoint,calibration
from .certifier import Certifier
from .numerics import I,SCALE


def is_cycle(g):
    if g.n<3 or g.m!=g.n or any(len(a)!=2 for a in g.adj):return False
    seen={0};stack=[0]
    while stack:
        u=stack.pop()
        for v,e in g.adj[u]:
            if v not in seen:seen.add(v);stack.append(v)
    return len(seen)==g.n

def cycle_family(g,qs):
    if not is_cycle(g):raise ValueError('exact cycle specialization requires a connected simple cycle')
    cuts=set();paths=set();byquery=[]
    for q in qs:
        if g.topological(q)>=0:byquery.append(([],[]));continue
        ps=[tuple(sorted(p)) for p in g.paths(q)];assert len(ps)==2
        cs=sorted({tuple(sorted((a,b))) for a in ps[0] for b in ps[1]});cuts.update(cs);paths.update(ps);byquery.append((cs,ps))
    families=[{(e,) for e in range(g.m)},cuts,paths,{tuple(range(g.m))}];families=[f-{()} for f in families if f-{()}];weights={}
    for fam in families:
        for h in fam:weights[h]=weights.get(h,Fraction(0))+Fraction(1,len(families)*len(fam))
    assert sum(weights.values())==1
    return weights,byquery

class FastStructureJoint(StructureJoint):
    def __init__(self,g,qs,eps,delta,mode='mixture'):
        if g.m==0:
            Certifier.__init__(self,g,qs,eps,delta);self.mode=mode;self.supports=[];self.mass=[];self.matrix=np.empty((0,0),dtype=np.int64);self.cpu_structure=self.cpu_calibration=0.;return
        if not is_cycle(g):
            super().__init__(g,qs,eps,delta,mode);return
        Certifier.__init__(self,g,qs,eps,delta);self.mode=mode;tick=time.process_time()
        weights,self.byquery=cycle_family(g,qs)
        self.supports=sorted(weights);self.mass=[weights[h] for h in self.supports];self.matrix=np.zeros((len(weights),g.m),dtype=np.int64)
        for j,h in enumerate(self.supports):self.matrix[j,list(h)]=1
        self.cpu_structure=time.process_time()-tick;tick=time.process_time()
        if mode=='hierarchical':self.thresholds=np.array([calibration(len(h),p.numerator,p.denominator,str(delta)) for h,p in zip(self.supports,self.mass)],dtype=np.int64)
        elif mode=='cardinality':self.thresholds=np.array([calibration(k,1,g.m*math.comb(g.m,k),str(delta)) for k in range(1,g.m+1)],dtype=np.int64)
        elif mode=='mixture':
            self.logmass=np.array([(I(p.numerator)/p.denominator).log().floor_scaled() for p in self.mass],dtype=np.int64);self.threshold=(1/I(delta)).log().ceil_scaled()
        else:raise ValueError(mode)
        self.cpu_calibration=time.process_time()-tick

    def _scores(self,base,opposite,masks,verified):
        if self.mode!='mixture' or not verified:return super()._scores(base,opposite,masks,verified)
        h=base[None,:]+masks*(opposite-base)[None,:];values=h@self.matrix.T+self.logmass;out=[];exp_cache={}
        for row in values:
            mx=int(max(row))
            # Two integer ticks ensure the original outward sum/log gate is
            # also positive. No tolerance and no EviFlow threshold are used.
            if mx>=self.threshold+2:
                out.append(mx-self.threshold-1);continue
            s=I(0)
            for v in row:
                d=int(v)-mx
                if d not in exp_cache:exp_cache[d]=(I(d)/SCALE).exp()
                s=s+exp_cache[d]
            out.append((I(mx)/SCALE+s.log()).floor_scaled()-self.threshold)
        return np.array(out,dtype=np.int64)
