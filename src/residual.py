"""One residual-evidence acquisition intervention; predictions never certify.

LP linearizes the calibrated log-mixture score at current counts, rather than
tracking a historical KL share. It covers every alternative of every query
still awaiting formal delivery. Only counts and public workload enter planning.
"""
import math
import numpy as np
from scipy.optimize import linprog
from .numerics import fast_stats,rr
from .certifier import Counts


class ResidualSampler:
    def __init__(self,g,qs,epsilon,delta):
        self.g=g;self.qs=qs;self.eps=epsilon;self.delta=delta
        self.active=np.array([g.topological(q)<0 for q in qs]);self.meta={}

    def delivered(self,proposed):
        # Actual unchanged exact certifier outputs only. No prediction here.
        self.active &= np.asarray(proposed)<0

    def _terms(self,c,tau):
        ph,lh,lq=fast_stats(c.N,c.S,self.eps);a,b=rr(self.eps)
        r=a+(b-a)*tau;rho=1/(self.g.m+1)
        base=np.logaddexp(math.log1p(-rho),math.log(rho)+lq-lh)
        lt=c.S*np.log(r)+(c.N-c.S)*np.log1p(-r)
        w=np.logaddexp(math.log1p(-rho),math.log(rho)+lq-lt)-base
        return ph,base,np.maximum(w,0),ph>=r

    def allocate(self,c,batch):
        m=self.g.m;total=int(c.N.sum());rows=[];rhs=[];families=set()
        reason='pilot' if total==0 else 'delivered' if not self.active.any() else 'residual'
        if total and self.active.any():
            ph=fast_stats(c.N,c.S,self.eps)[0]
            predicted=Counts(c.N.astype(float)+batch,c.S.astype(float)+batch*ph)
            for j,q in enumerate(self.qs):
                if not self.active[j]:continue
                _,base,w,hat=self._terms(c,q.tau)
                _,base1,w1,_=self._terms(predicted,q.tau)
                for alt in self.g.alternatives(hat,q):
                    key=(q.tau,alt)
                    if key in families:continue
                    families.add(key)
                    row=(base1-base)/batch
                    row=row.copy();row[list(alt)]+=(w1-w)[list(alt)]/batch
                    deficit=math.log(1/self.delta)+1e-6-base.sum()-w[list(alt)].sum()
                    rows.append(row);rhs.append(deficit)
        x=np.ones(m)/m;lp_reports=None
        if rows and max(rhs)>0:
            sol=linprog(np.ones(m),A_ub=-np.array(rows),b_ub=-np.array(rhs),bounds=(0,None),method='highs',options={'threads':1})
            if sol.success and sol.x.sum()>0:
                x=np.maximum(sol.x,0)/sol.x.sum();lp_reports=float(sol.x.sum())
            else:reason='uniform_infeasible_prediction'
        elif total and self.active.any():reason='uniform_exact_rule_pending'
        # Fixed decaying exploration; no fitted rho, values, or query selection.
        eta=min(1.,math.sqrt(m/(total+batch)));x=(1-eta)*x+eta/m
        alloc=np.zeros(m,dtype=np.int64);target=batch*x
        for _ in range(batch):alloc[int(np.argmax(target-alloc))]+=1
        self.meta=dict(reason=reason,unresolved_queries=int(self.active.sum()),alternatives=len(rows),predicted_residual_reports=lp_reports,exploration=eta,predictions_certify=False,fresh_target=True)
        return alloc
