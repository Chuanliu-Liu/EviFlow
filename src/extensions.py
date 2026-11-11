"""Explicitly labelled baseline extensions; shared core certifier unchanged."""
import math
import numpy as np
from scipy.optimize import brentq
from .numerics import rr,fast_stats
from .sampling import Sampler

class BLUCBThreshold(Sampler):
    def __init__(self,g,qs,epsilon,delta):
        super().__init__(g,qs,epsilon,'blucb');self.delta=delta;self.cursor=0
    def allocate(self,c,batch):
        self.k+=1;m=self.g.m;a,b=rr(self.eps);ph,lh,lq=fast_stats(c.N,c.S,self.eps)
        lo=np.full(m,a);hi=np.full(m,b);threshold=math.log(m/self.delta)
        for e,(n,s) in enumerate(zip(c.N,c.S)):
            if n==0:continue
            def f(v):return lq[e]-s*math.log(v)-(n-s)*math.log1p(-v)-threshold
            if f(a)>0 and ph[e]>a:lo[e]=brentq(f,a,ph[e])
            if f(b)>0 and ph[e]<b:hi[e]=brentq(f,ph[e],b)
        choices=[]
        for offset in range(len(self.qs)):
            j=(self.cursor+offset)%len(self.qs);q=self.qs[j]
            if self.g.topological(q)>=0:continue
            paths=self.g.paths(q)
            if not paths:continue
            candidate=max(paths,key=lambda p:min(lo[e] for e in p))
            # Native challenger excludes all supersets of the candidate.
            alternatives=[p for p in paths if not set(candidate).issubset(p)]
            optimistic=max(paths,key=lambda p:min(hi[e] for e in p))
            r=a+(b-a)*q.tau
            if min(lo[e] for e in candidate)>=r or min(hi[e] for e in optimistic)<r:continue
            challenger=max(alternatives,key=lambda p:min(hi[e] for e in p)) if alternatives else candidate
            ce=min(candidate,key=lambda e:lo[e]);de=min(challenger,key=lambda e:lo[e])
            picked=max((ce,de),key=lambda e:hi[e]-lo[e]);choices.append((j,picked))
        alloc=np.zeros(m,dtype=np.int64)
        if choices:
            j,e=choices[0];self.cursor=(j+1)%len(self.qs);alloc[e]=batch
        else:
            # If planning intervals suffice but the common joint rule does not,
            # continue with the same uncertainty-driven bottleneck logic.
            e=int(np.argmax(hi-lo));alloc[e]=batch
        self.meta={'candidate_challenger':True,'intervals':'KL-mixture','query_round_robin':True}
        return alloc

class FWSReportClock(Sampler):
    """R2 update on the report clock; frozen means within each batch.

    Compute one FW direction at the batch boundary, average it over every
    nonforced report index and the uniform direction over square-block indices.
    Batch=1 with sequential observations reproduces R2's sampling update.
    """
    def __init__(self,g,qs,epsilon,method='fws_report'):
        super().__init__(g,qs,epsilon,method)
    def allocate(self,c,batch):
        from scipy.optimize import linprog
        from .sampling import information_rows,empirical
        m=self.g.m;t0=int(c.N.sum());self.k+=1
        rows=information_rows(self.g,self.qs,c,self.eps)
        ph,_=empirical(c,self.eps);a,b=rr(self.eps)
        undefined_instance=any(np.any(ph==a+(b-a)*q.tau) for q in self.qs)
        t=max(m+1,t0+1);r=t**(-.9)
        need_direction=(not undefined_instance) and any(report>m and math.isqrt(report//m)**2!=report//m for report in range(t0+1,t0+batch+1))
        if len(rows) and need_direction:
            values=rows@self.x;active=rows[values<values.min()+r]
            sol=linprog(np.r_[np.zeros(m),-1.],A_ub=np.c_[-active,np.ones(len(active))],b_ub=-(active@self.x),A_eq=np.array([np.r_[np.ones(m),0.]]),b_eq=[1.],bounds=[(0,None)]*m+[(None,None)],method='highs',options={'threads':1})
            if not sol.success:raise RuntimeError(sol.message)
            z=sol.x[:m]
        else:z=np.ones(m)/m
        alloc=np.zeros(m,dtype=np.int64);forced_count=0
        for offset in range(batch):
            report=t0+offset+1
            if report<=m:
                e=report-1;alloc[e]+=1;continue
            forced=undefined_instance or math.isqrt(report//m)**2==report//m
            direction=np.ones(m)/m if forced else z
            self.x=(report-1)/report*self.x+direction/report
            forced_count+=int(forced)
            e=int(np.argmax(self.x/np.maximum(c.N+alloc,1)))
            alloc[e]+=1
        self.meta={'report_clock':t0+batch,'forced_reports_in_batch':forced_count,'r':r,'batch_frozen_gradient':True}
        return alloc

from .certifier import Certifier
class TopologyPriorCertifier(Certifier):
    """One mechanism repair: public connectivity sets activation density."""
    def __init__(self,g,qs,eps,delta):
        from fractions import Fraction
        ranks=[g.oracle([1]*g.m,q,True)[0] for q in qs if g.topological(q)<0]
        if ranks and g.m>1:
            rho=min(Fraction(3,4),Fraction(min(ranks),g.m))
        else:rho=Fraction(1,g.m+1) if g.m else Fraction(1,2)
        super().__init__(g,qs,eps,delta,(rho.numerator,rho.denominator))
