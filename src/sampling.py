"""Deployable planners: no simulator truth is accepted by any API."""
import math
import numpy as np
from scipy.optimize import linprog
from .numerics import rr,kl


def empirical(c,epsilon):
    a,b=rr(epsilon)
    ph=np.clip(np.divide(c.S,c.N,out=np.full(len(c.N),.5),where=c.N>0),a,b)
    return ph,(ph-a)/(b-a)

def information_rows(g,qs,c,epsilon):
    ph,theta=empirical(c,epsilon);a,b=rr(epsilon);rows=[]
    for q in qs:
        if g.topological(q)>=0:continue
        r=a+(b-a)*q.tau;d=kl(ph,r)
        # Ridge only stabilizes the planner; never used by a certifier.
        d=np.maximum(d,1e-8)
        for alt in g.alternatives(ph>=r,q):
            row=np.zeros(g.m);row[list(alt)]=d[list(alt)];rows.append(row)
    return np.unique(np.asarray(rows).reshape((-1,g.m)),axis=0)

def fractional_target(rows,m):
    if len(rows)==0:return np.ones(m)/m,{'objective':0.,'status':'topology'}
    sol=linprog(np.ones(m),A_ub=-rows,b_ub=-np.ones(len(rows)),bounds=(0,None),method='highs',options={'threads':1})
    if not sol.success:raise RuntimeError(sol.message)
    x=np.maximum(sol.x,0);return x/x.sum(),{'objective':float(x.sum()),'status':'optimal'}

def integral_target(g,qs,c,epsilon):
    """Exact shared complete certificate via a small binary/continuous MILP.

    x_e is a shared sample requirement. Selecting one full witness for q
    imposes x_e>=1/d(e,q) for each of its edges. x is shared across all q.
    """
    from scipy.optimize import milp,Bounds,LinearConstraint
    ph,_=empirical(c,epsilon);a,b=rr(epsilon)
    fam=[];ds=[]
    for q in qs:
        if g.topological(q)>=0:continue
        r=a+(b-a)*q.tau;hat=ph>=r
        fam.append(g.complete_certificates(hat,q));ds.append(np.maximum(kl(ph,r),1e-8))
    if not fam:return np.ones(g.m)/g.m,{'objective':0.,'gap':0.}
    # Equal thresholds permit exact binary subset search, no solver tolerance.
    if len({q.tau for q in qs})==1 and g.m<=15:
        d=ds[0];costs=1/d;best=float('inf');chosen=None
        for mask in range(1,1<<g.m):
            x=np.array([bool(mask>>e&1) for e in range(g.m)])
            cost=float(costs@x)
            if cost>=best:continue
            if all(any(all(x[e] for e in w) for w in ws) for ws in fam):best=cost;chosen=x
        if chosen is None:raise RuntimeError('No complete witness')
        target=costs*chosen
        return target/target.sum(),{'objective':best,'gap':0.,'status':'exact_enumeration'}
    count=sum(map(len,fam));size=g.m+count;obj=np.r_[np.ones(g.m),np.zeros(count)]
    constraints=[];lower=[];offset=g.m
    for ws,d in zip(fam,ds):
        row=np.zeros(size);row[offset:offset+len(ws)]=1;constraints.append(row);lower.append(1.)
        for j,w in enumerate(ws):
            for e in w:
                row=np.zeros(size);row[e]=1;row[offset+j]=-1/d[e]
                constraints.append(row);lower.append(0.)
        offset+=len(ws)
    result=milp(obj,integrality=np.r_[np.zeros(g.m),np.ones(count)],bounds=Bounds(np.zeros(size),np.r_[np.full(g.m,np.inf),np.ones(count)]),constraints=LinearConstraint(np.array(constraints),lower,np.inf),options={'time_limit':10.,'mip_rel_gap':0.,'threads':1})
    if result.x is None:raise RuntimeError('Shared certificate MILP failed')
    x=result.x[:g.m];return x/x.sum(),{'objective':float(result.fun),'gap':float(result.mip_gap),'status':str(result.message)}

class Sampler:
    def __init__(self,g,qs,epsilon,method):
        self.g=g;self.qs=qs;self.eps=epsilon;self.method=method
        self.x=np.ones(g.m)/g.m;self.target_total=np.zeros(g.m);self.k=0;self.meta={}
    def allocate(self,c,batch):
        self.k+=1;m=self.g.m
        if self.method=='uniform':x=np.ones(m)/m
        elif self.method=='integral':x,self.meta=integral_target(self.g,self.qs,c,self.eps)
        else:
            rows=information_rows(self.g,self.qs,c,self.eps)
            if self.method=='fractional':x,self.meta=fractional_target(rows,m)
            elif self.method in ('fws','fws_native'):
                # Batch-clock adaptation retains R2's r-subgradient game,
                # 1/t averaging, square-block forced exploration, ratio tracker.
                t=m+self.k
                forced=math.isqrt(t//m)**2==t//m
                if forced or len(rows)==0:z=np.ones(m)/m
                else:
                    values=rows@self.x;active=rows[values<values.min()+t**(-.9)]
                    rhs=-(active@self.x)
                    # max_z min_h <z-x,h>: -h.z + v <= -h.x
                    sol=linprog(np.r_[np.zeros(m),-1.],A_ub=np.c_[-active,np.ones(len(active))],b_ub=rhs,A_eq=np.array([np.r_[np.ones(m),0.]]),b_eq=[1.],bounds=[(0,None)]*m+[(None,None)],method='highs',options={'threads':1})
                    if not sol.success:raise RuntimeError(sol.message)
                    z=sol.x[:m]
                self.x=(t-1)/t*self.x+z/t;x=self.x.copy()
                self.meta={'round':t,'forced':forced,'r':t**(-.9)}
            else:raise ValueError(self.method)
        x=np.maximum(0,x);x/=x.sum()
        # Same decaying exploration for LP/integral; FWS has native square blocks.
        if self.method in ('fractional','integral'):
            eta=min(1.,1/math.sqrt(self.k));x=(1-eta)*x+eta/m
        allocation=np.zeros(m,dtype=np.int64)
        if self.method in ('fws','fws_native'):
            for _ in range(batch):
                e=int(np.argmax(x/np.maximum(c.N+allocation,1e-30)))
                allocation[e]+=1
        else:
            self.target_total+=batch*x
            for _ in range(batch):
                e=int(np.argmax(self.target_total-c.N-allocation));allocation[e]+=1
        return allocation
