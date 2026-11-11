"""R002 topology-only product calibration; original R001 methods are untouched."""
from fractions import Fraction
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.special import expit, xlogy
from .certifier import Certifier
from .numerics import I, rr, rr_i, fast_stats, verified_stats, loglik_i


def public_library(g, qs):
    """Exactly R001's typed tuple/mass convention for m<=10, including duplicates.

    Above 10 edges the declared finite library has BFS paths and endpoint cuts.
    Neither API receives means, observations, privacy, risk, or simulation seeds.
    """
    cuts, paths = set(), set()
    for q in qs:
        if g.topological(q) >= 0:
            continue
        if g.m <= 10:
            minimal = []
            for c in sorted(g.cuts(q), key=len):
                if not any(set(z).issubset(c) for z in minimal):
                    minimal.append(c)
            cuts.update(minimal)
            paths.update(g.paths(q))
        else:
            from collections import deque
            parent = {q.s: None}
            queue = deque([q.s])
            while queue and q.t not in parent:
                u = queue.popleft()
                for v, e in g.adj[u]:
                    if v not in parent:
                        parent[v] = (u, e)
                        queue.append(v)
            if q.t in parent:
                v = q.t; path = []
                while v != q.s:
                    v, e = parent[v]; path.append(e)
                paths.add(tuple(reversed(path)))
            cuts.update(tuple(sorted(e for _, e in g.adj[v])) for v in (q.s, q.t))
    families = [{(e,) for e in range(g.m)}, cuts, paths, {tuple(range(g.m))}]
    families = [f - {()} for f in families if f - {()}]
    weights = {}
    for family in families:
        for h in family:
            weights[h] = weights.get(h, Fraction()) + Fraction(1, len(families)*len(family))
    supports = sorted(weights)
    return supports, [weights[h] for h in supports]


def optimize_product(m, supports, masses, tolerance=1e-7):
    """Convex epigraph in logits + independent entropy-dual lower bound.

    Returned rational vector is the deployed prior; numerical gap includes
    rounding and clipping. No successful optimizer status is an inference gate.
    """
    t0 = time.process_time()
    if not supports:
        return dict(rho=[], rho_ratios=[], primal=0., dual=0., gap=0., cpu=0., supports=[])
    B = np.zeros((len(supports), m))
    for j, h in enumerate(supports): B[j, list(h)] = 1
    logp = np.log([float(p) for p in masses])
    def losses(x): return np.logaddexp(0., x).sum() - B@x + logp
    x0 = np.zeros(m)
    start = np.r_[x0, max(losses(x0))]
    def constraint(z): return z[-1] - losses(z[:-1])
    def jac(z): return np.c_[B-expit(z[:-1]), np.ones(len(B))]
    sol = minimize(lambda z:z[-1], start, jac=lambda z:np.r_[np.zeros(m),1.],
                   method='SLSQP', bounds=[(-18.420680733952366,18.420680733952366)]*m+[(None,None)],
                   constraints=[dict(type='ineq',fun=constraint,jac=jac)],
                   options=dict(ftol=1e-12,maxiter=2000))
    rho = np.clip(expit(sol.x[:-1]),1e-8,1-1e-8)
    ratios = [Fraction(str(round(float(v),12))) for v in rho]
    rho = np.array([float(v) for v in ratios])
    loss = -B@np.log(rho)-(1-B)@np.log1p(-rho)+logp
    # A dual feasible alpha is a distribution on supports. Marginal equality
    # is only an LP search aid; re-evaluate actual alpha marginals below.
    lp = linprog(-logp,A_eq=np.r_[np.ones((1,len(B))),B.T],b_eq=np.r_[1.,rho],
                 bounds=(0,None),method='highs',options={'threads':1})
    alpha = np.maximum(lp.x,0) if lp.success else np.ones(len(B))/len(B)
    alpha /= alpha.sum()
    u = np.clip(alpha@B,0,1)
    dual = float((-xlogy(u,u)-xlogy(1-u,1-u)).sum()+alpha@logp)
    primal = float(max(loss)); gap = primal-dual
    return dict(rho=rho.tolist(),rho_ratios=[[p.numerator,p.denominator] for p in ratios],
                primal=primal,dual=dual,gap=gap,converged=bool(sol.success and gap<tolerance),
                optimizer_message=str(sol.message),dual_lp_message=lp.message,
                alpha=alpha.tolist(),supports=[list(h) for h in supports],
                masses=[[p.numerator,p.denominator] for p in masses],
                cpu=time.process_time()-t0,scope='full typed library' if m<=10 else 'finite BFS/endpoint library')


def topology_prior(g,qs):
    return optimize_product(g.m,*public_library(g,qs))


class VectorCertifier(Certifier):
    def __init__(self,g,qs,eps,delta,rho_ratios):
        super().__init__(g,qs,eps,delta)
        if len(rho_ratios)!=g.m: raise ValueError('one fixed rho per edge required')
        if any(not 0<a<b for a,b in rho_ratios): raise ValueError('interior rho required')
        self.ratios=tuple(tuple(map(int,r)) for r in rho_ratios)
        self.rho=np.array([a/b for a,b in self.ratios]); self.rhoi=[I(a)/b for a,b in self.ratios]

    def payload(self,c,stats,tau):
        a,b=rr_i(str(self.eps));r=a+(b-a)*I(tau)
        hat=[];weights=[];correction=I(0);cache={};llcache={}
        for e,(n,s,(v,lh,lq,_,_)) in enumerate(zip(c.N,c.S,stats)):
            if v.lo>=r.hi: hat.append(True)
            elif v.hi<r.lo: hat.append(False)
            elif int(n)==0: hat.append(tau<=.5)
            elif tau==.5: hat.append(int(s)*2>=int(n))
            else: return None
            ns=(int(n),int(s))
            if ns not in llcache: llcache[ns]=loglik_i(*ns,r)
            lt=llcache[ns];rho=self.rhoi[e]
            key=(*ns,self.ratios[e])
            if key not in cache:
                base=(1-rho+rho*(lq-lh).exp()).log()
                at=(1-rho+rho*(lq-lt).exp()).log()
                cache[key]=(max(0,(at-base).floor_scaled()),base)
            w,base=cache[key];weights.append(w);correction=correction-base
        threshold=((I(1)/I(self.delta)).log()+correction).ceil_scaled()
        return np.array(hat),weights,threshold

    def evaluate(self,c,method='vector',active=None):
        if method!='vector': return super().evaluate(c,method,active)
        out=self.top.copy()
        if self.eps==0 or self.g.m==0:return out
        if active is None:active=np.ones(len(self.qs),dtype=bool)
        ph,lh,lq=fast_stats(c.N,c.S,self.eps);a,b=rr(self.eps)
        logr=np.log(self.rho);log1=np.log1p(-self.rho);R=lh-lq
        h0=np.logaddexp(log1,logr-R);A=-float(sum(h0));beta=math.log(1/self.delta)+A
        candidates=[];fast={}
        for j,q in enumerate(self.qs):
            if out[j]>=0 or not active[j]:continue
            if q.tau not in fast:
                r=a+(b-a)*q.tau;lt=c.S*np.log(r)+(c.N-c.S)*np.log1p(-r)
                gain=np.maximum(0,lh-lt);hat=ph>=r
                w=np.maximum(0,np.logaddexp(log1,logr+gain-R)-h0)
                fast[q.tau]=(hat,w,gain)
            hat,w,gain=fast[q.tau];ans=int(self.g.reachable(hat,q))
            score,H=self.g.oracle(w*np.where(ans,hat,~hat),q,bool(ans))
            if score>beta:candidates.append((j,ans,H,score))
        if not candidates:return out
        stats=verified_stats(c.N,c.S,self.eps);payloads={};self.last_meta={'verified_queries':0,'A':A}
        decomp=[]
        for j,ans,H,z in candidates:
            q=self.qs[j]
            if q.tau not in payloads:payloads[q.tau]=self.payload(c,stats,q.tau)
            payload=payloads[q.tau]
            if payload is None:continue
            hat,w,threshold=payload
            if int(self.g.reachable(hat,q))!=ans:continue
            score,_=self.g.oracle([v if bool(h)==bool(ans) else 0 for v,h in zip(w,hat)],q,bool(ans))
            if score>threshold:
                out[j]=ans;self.last_meta['verified_queries']+=1
                H=[e for e in H if bool(hat[e])==bool(ans)]
                mask=np.zeros(self.g.m,dtype=bool);mask[H]=True
                _,_,gain=fast[q.tau]
                activation=-float(np.where(mask,logr,log1).sum())
                signal=float((gain-R)[mask].sum())
                decomp.append(dict(query=j,H=H,activation=activation,mixture_regret=float(R[mask].sum()),
                                   signal=signal,projected_log_e=float(z-A),remainder=float(z-A-signal+activation)))
            self.last_meta['margin_integer_last']=int(score-threshold)
        self.last_meta['decomposition']=decomp
        return out
