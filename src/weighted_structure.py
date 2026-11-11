"""Public typed-support mixture calibration, no per-query risk split.

Equal prior mass on singleton, minimal-cut, path, and full-domain families;
duplicated supports combine their rational masses. This avoids charging a
minimal cut for the number of redundant nonminimal public cut supports.
"""
from fractions import Fraction
import math
import numpy as np
from .certifier import Certifier
from .numerics import I,rr,rr_i,fast_stats,verified_stats,loglik_i

class WeightedStructure(Certifier):
    def __init__(self,g,qs,eps,delta):
        super().__init__(g,qs,eps,delta)
        self.ready=False

    def _setup(self):
        g,qs=self.g,self.qs
        cuts=set();paths=set()
        for q in qs:
            if g.topological(q)>=0:continue
            minimal=[]
            for c in sorted(g.cuts(q),key=len):
                cs=set(c)
                if not any(set(z).issubset(cs) for z in minimal):minimal.append(c)
            cuts.update(minimal);paths.update(g.paths(q))
        families=[{(e,) for e in range(g.m)},cuts,paths,{tuple(range(g.m))}]
        families=[f-{()} for f in families if f-{()}];weights={}
        for fam in families:
            for h in fam:weights[h]=weights.get(h,Fraction(0))+Fraction(1,len(families)*len(fam))
        if weights:assert sum(weights.values())==1
        self._labels,self._answers=g.label_table(qs)
        self.supports=sorted(weights);self._support=np.zeros((len(weights),g.m),dtype=np.int64)
        for j,h in enumerate(self.supports):self._support[j,list(h)]=1
        self.mass=[weights[h] for h in self.supports]
        self.logp=np.array([math.log(float(v)) for v in self.mass])
        self.logp_lower=np.array([(I(v.numerator)/v.denominator).log().floor_scaled() for v in self.mass],dtype=np.int64)
        self.threshold=(I(1)/I(self.delta)).log().ceil_scaled()
        self.ready=True

    def evaluate(self,c,method='structure_weighted',active=None):
        out=self.top.copy()
        if self.eps==0 or self.g.m==0:return out
        if not self.ready:self._setup()
        if active is None:active=np.ones(len(self.qs),dtype=bool)
        ph,lh,lq=fast_stats(c.N,c.S,self.eps);a,b=rr(self.eps);cache={};candidates=[]
        for j,q in enumerate(self.qs):
            if out[j]>=0 or not active[j]:continue
            if q.tau not in cache:
                r=a+(b-a)*q.tau;lt=c.S*np.log(r)+(c.N-c.S)*np.log1p(-r);hat=ph>=r
                h=np.where(self._labels==hat,(lq-lh)[None,:],(lq-lt)[None,:]);score=np.max(h@self._support.T+self.logp,axis=1);cache[q.tau]=(hat,score)
            hat,score=cache[q.tau];ans=int(self.g.reachable(hat,q))
            if np.min(score[self._answers[:,j]!=ans])>math.log(1/self.delta):candidates.append((j,ans))
        if not candidates:return out
        stats=verified_stats(c.N,c.S,self.eps);verified={};a,b=rr_i(str(self.eps))
        self.last_meta={'integral_log_width_max':max(v[3] for v in stats),'verified_queries':0,'supports':len(self.supports),'prior':'equal mass over four public support types'}
        for j,ans in candidates:
            q=self.qs[j]
            if q.tau not in verified:
                r=a+(b-a)*I(q.tau);hat=[];h0=[];h1=[];uncertain=False
                for n,s,(v,ll,qq,_,_) in zip(c.N,c.S,stats):
                    if v.lo>=r.hi:hat.append(True)
                    elif v.hi<r.lo:hat.append(False)
                    elif q.tau==.5:hat.append(int(s)*2>=int(n))
                    elif int(n)==0:hat.append(q.tau<=.5)
                    else:uncertain=True;break
                    h0.append((qq-ll).floor_scaled());h1.append((qq-loglik_i(int(n),int(s),r)).floor_scaled())
                if uncertain:verified[q.tau]=None
                else:
                    hat=np.array(hat);h=np.where(self._labels==hat,np.array(h0)[None,:],np.array(h1)[None,:]);score=np.max(h@self._support.T+self.logp_lower,axis=1);verified[q.tau]=(hat,score)
            value=verified[q.tau]
            if value is None:continue
            hat,score=value
            if int(self.g.reachable(hat,q))!=ans:continue
            margin=int(np.min(score[self._answers[:,j]!=ans]))-self.threshold
            if margin>0:out[j]=ans;self.last_meta['verified_queries']+=1
            self.last_meta['margin_integer_last']=margin
        return out
