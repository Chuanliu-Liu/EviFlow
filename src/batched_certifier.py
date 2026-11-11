"""Same certificate with bounded, within-call reuse of verified edge vectors.

One cached threshold only: memory stays O(m) even for diverse thresholds.
The cache is discarded after every evaluate call, including exceptions.
"""
import numpy as np
from .certifier import Certifier
from .numerics import I,rr_i,loglik_i

class BatchedCertifier(Certifier):
    def evaluate(self,c,method='selective',active=None):
        self._batch_key=None;self._batch_payload=None
        try:return super().evaluate(c,method,active)
        finally:self._batch_key=None;self._batch_payload=None

    def _payload(self,c,stats,tau,method):
        a,b=rr_i(str(self.eps));r=a+(b-a)*I(tau)
        hat=[];lhs=[];lqs=[];lts=[];likelihood_cache={}
        for n,s,(v,lh,lq,_,_) in zip(c.N,c.S,stats):
            if v.lo>=r.hi:hat.append(True)
            elif v.hi<r.lo:hat.append(False)
            elif int(n)==0:hat.append(tau<=.5)
            elif tau==.5:hat.append(int(s)*2>=int(n))
            else:return None
            key=(int(n),int(s))
            if key not in likelihood_cache:likelihood_cache[key]=loglik_i(*key,r)
            lhs.append(lh);lqs.append(lq);lts.append(likelihood_cache[key])
        hat=np.array(hat,dtype=bool)
        if method=='edgewise':
            threshold=(I(self.g.m)/I(self.delta)).log().ceil_scaled()
            ok=np.array([(u-v).floor_scaled()>threshold for u,v in zip(lqs,lts)])
            return hat,ok,threshold
        weights=[];correction=I(0)
        rho=I(1)/(self.g.m+1) if self.rho_ratio is None else I(self.rho_ratio[0])/self.rho_ratio[1]
        cache={}
        for lh,lq,lt in zip(lhs,lqs,lts):
            if method=='selective':
                key=(lh.lo,lh.hi,lq.lo,lq.hi,lt.lo,lt.hi)
                if key not in cache:
                    base=(1-rho+rho*(lq-lh).exp()).log()
                    at=(1-rho+rho*(lq-lt).exp()).log()
                    cache[key]=(max(0,(at-base).floor_scaled()),base)
                weight,base=cache[key];weights.append(weight);correction=correction-base
            else:
                weights.append(max(0,(lh-lt).floor_scaled()));correction=correction+lh-lq
        threshold=((I(1)/I(self.delta)).log()+correction).ceil_scaled()
        return hat,weights,threshold

    def _verify(self,c,stats,j,ans,method):
        # Structure already reuses its common projected-label score array.
        if method=='structure':return super()._verify(c,stats,j,ans,method)
        q=self.qs[j];key=(method,q.tau)
        if getattr(self,'_batch_key',None)!=key:
            self._batch_payload=self._payload(c,stats,q.tau,method);self._batch_key=key
        if self._batch_payload is None:return False,0
        hat,vector,threshold=self._batch_payload
        if int(self.g.reachable(hat,q))!=ans:return False,0
        if method=='edgewise':
            valid=self.g.reachable(vector&hat,q) if ans else not self.g.reachable(~(vector&~hat),q)
            return valid,int(valid)
        weights=[w if bool(h)==bool(ans) else 0 for w,h in zip(vector,hat)]
        z,_=self.g.oracle(weights,q,bool(ans))
        return z>threshold,int(z-threshold)
