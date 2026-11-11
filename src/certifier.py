import math
from dataclasses import dataclass
import numpy as np
from .numerics import I, rr, rr_i, fast_stats, verified_stats, loglik_i, SCALE

METHODS=('selective','edgewise','nonselective','structure')

@dataclass
class Counts:
    N: np.ndarray
    S: np.ndarray
    @classmethod
    def zeros(cls,m):return cls(np.zeros(m,dtype=np.int64),np.zeros(m,dtype=np.int64))
    def update(self,n,s):
        self.N+=n;self.S+=s
        if np.any(self.S<0) or np.any(self.S>self.N):raise ValueError('Invalid counts')

class Certifier:
    def __init__(self,graph,queries,epsilon,delta,rho_ratio=None):
        if epsilon<0 or not 0<delta<1:raise ValueError('Invalid privacy/risk')
        self.g=graph;self.qs=queries;self.eps=epsilon;self.delta=delta
        self.rho_ratio=rho_ratio
        if rho_ratio is not None and not 0<rho_ratio[0]<rho_ratio[1]:raise ValueError("rho must be interior")
        self.top=np.array([graph.topological(q) for q in queries],dtype=int)
        self._labels=None
        self.last_meta={}

    def structure_setup(self):
        if self._labels is None:
            self._labels,self._answers=self.g.label_table(self.qs)
            fam=self.g.structure_family(self.qs)
            self._support=np.zeros((len(fam),self.g.m),dtype=np.int64)
            for j,a in enumerate(fam):self._support[j,list(a)]=1
        return self._labels,self._answers,self._support

    def evaluate(self,c,method='selective',active=None):
        if method not in METHODS:raise ValueError(method)
        out=self.top.copy()
        if self.eps==0 or self.g.m==0:return out
        if active is None:active=np.ones(len(self.qs),dtype=bool)
        ph,lh,lq=fast_stats(c.N,c.S,self.eps)
        a,b=rr(self.eps);rho=1/(self.g.m+1) if self.rho_ratio is None else self.rho_ratio[0]/self.rho_ratio[1];R=lh-lq
        h0=np.logaddexp(math.log1p(-rho),math.log(rho)-R)
        candidates=[];structure_screen_scores={}
        self._structure_verified={}
        for j,q in enumerate(self.qs):
            if out[j]>=0 or not active[j]:continue
            r=a+(b-a)*q.tau
            lt=c.S*np.log(r)+(c.N-c.S)*np.log1p(-r)
            hat=ph>=r;ans=int(self.g.reachable(hat,q))
            gain=np.maximum(0,lh-lt)
            if method=='edgewise':
                ok=lq-lt>math.log(self.g.m/self.delta)
                val=(self.g.reachable(ok&hat,q) if ans else not self.g.reachable(~(ok&~hat),q))
            elif method=='structure':
                labels,answers,support=self.structure_setup()
                if q.tau not in structure_screen_scores:
                    h=np.where(labels==hat,(lq-lh)[None,:],(lq-lt)[None,:])
                    structure_screen_scores[q.tau]=np.max(h@support.T,axis=1)
                score=structure_screen_scores[q.tau]
                val=np.min(score[answers[:,j]!=ans])>math.log(len(support)/self.delta)
            else:
                if method=='selective':
                    weights=np.maximum(0,np.logaddexp(math.log1p(-rho),math.log(rho)+gain-R)-h0)
                    beta=math.log(1/self.delta)-sum(h0)
                else:weights=gain;beta=math.log(1/self.delta)+sum(R)
                weights=weights*np.where(ans,hat,~hat)
                z,_=self.g.oracle(weights,q,bool(ans));val=z>beta
            if val:candidates.append((j,ans))
        if not candidates:return out
        # The fast path can only nominate, never authorize, an output.
        stats=verified_stats(c.N,c.S,self.eps)
        self.last_meta={'integral_log_width_max':max(x[3] for x in stats),'verified_queries':0,'integer_scale':SCALE}
        for j,ans in candidates:
            valid,margin=self._verify(c,stats,j,ans,method)
            if valid:out[j]=ans;self.last_meta['verified_queries']+=1
            self.last_meta['margin_integer_last']=margin
        return out

    def _verify(self,c,stats,j,ans,method):
        q=self.qs[j];a,b=rr_i(str(self.eps));r=a+(b-a)*I(q.tau)
        hat=[];lhs=[];lqs=[];lts=[];likelihood_cache={}
        for n,s,(v,lh,lq,_,_) in zip(c.N,c.S,stats):
            if v.lo>=r.hi:hat.append(True)
            elif v.hi<r.lo:hat.append(False)
            elif int(n)==0:hat.append(q.tau<=.5)
            else:
                # Exact comparison at tau=.5 has rational r=.5 by symmetry.
                if q.tau==.5:hat.append(int(s)*2>=int(n))
                else:return False,0
            key=(int(n),int(s))
            if key not in likelihood_cache:likelihood_cache[key]=loglik_i(*key,r)
            lhs.append(lh);lqs.append(lq);lts.append(likelihood_cache[key])
        hat=np.array(hat,dtype=bool)
        if int(self.g.reachable(hat,q))!=ans:return False,0
        if method=='edgewise':
            threshold=(I(self.g.m)/I(self.delta)).log().ceil_scaled()
            ok=np.array([(u-v).floor_scaled()>threshold for u,v in zip(lqs,lts)])
            valid=self.g.reachable(ok&hat,q) if ans else not self.g.reachable(~(ok&~hat),q)
            return valid,int(valid)
        if method=='structure':
            labels,answers,support=self.structure_setup()
            if q.tau not in self._structure_verified:
                h0=np.array([(u-v).floor_scaled() for u,v in zip(lqs,lhs)],dtype=np.int64)
                h1=np.array([(u-v).floor_scaled() for u,v in zip(lqs,lts)],dtype=np.int64)
                h=np.where(labels==hat,h0[None,:],h1[None,:])
                score=np.max(h@support.T,axis=1)
                threshold=(I(len(support))/I(self.delta)).log().ceil_scaled()
                self._structure_verified[q.tau]=(score,threshold)
            score,threshold=self._structure_verified[q.tau]
            z=int(np.min(score[answers[:,j]!=ans]))
            return z>threshold,z-threshold
        weights=[];correction=I(0);rho=I(1)/(self.g.m+1) if self.rho_ratio is None else I(self.rho_ratio[0])/self.rho_ratio[1]
        weight_cache={}
        for lh,lq,lt in zip(lhs,lqs,lts):
            if method=='selective':
                key=(lh.lo,lh.hi,lq.lo,lq.hi,lt.lo,lt.hi)
                if key not in weight_cache:
                    base=(1-rho+rho*(lq-lh).exp()).log()
                    at=(1-rho+rho*(lq-lt).exp()).log()
                    weight_cache[key]=(max(0,(at-base).floor_scaled()),base)
                weight,base=weight_cache[key]
                weights.append(weight)
                correction=correction-base
            else:
                weights.append(max(0,(lh-lt).floor_scaled()))
                correction=correction+lh-lq
        weights=[w if bool(h)==bool(ans) else 0 for w,h in zip(weights,hat)]
        z,_=self.g.oracle(weights,q,bool(ans))
        threshold=((I(1)/I(self.delta)).log()+correction).ceil_scaled()
        return z>threshold,int(z-threshold)
