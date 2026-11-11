"""R005 source-faithful hierarchical subset GLR and complete subset mixture.

Primary: Kaufmann--Koolen JMLR 22(246), Thm 7, Sec 6, Eq (26), Sec 6.2.3.
No EviFlow threshold/compiler. All final acceptance uses directed intervals.
"""
from fractions import Fraction
from functools import lru_cache
import math
import time
import numpy as np
from scipy.special import logsumexp
from .certifier import Certifier
from .numerics import I, rr_i, fast_stats, verified_stats, loglik_i, SCALE

ZETA2 = I('1.644934066848226436472415166646025189218949901206798437735',
          '1.644934066848226436472415166646025189218949901206798437736')


def hinv(x):
    """Enclose inverse of h(u)=u-ln(u) on [1,infinity)."""
    low=I(1).lo;high=(2*x+4).hi
    for _ in range(150):
        mid=(I(low)+I(high))/2; hm=mid-mid.log()
        if hm.hi < x.lo:low=mid.lo
        elif hm.lo > x.hi:high=mid.hi
        else:break
    return I(low,high)


def cexp(x):
    """Directed enclosure of source Eq (10), two-sided, no approximation."""
    z=I(3)/2; u=(hinv(1+x)+(2*ZETA2).log())/2
    boundary=1/z.log();hb=boundary-boundary.log()
    linear=z*(u-z.log().log())
    if u.hi < hb.lo:return 2*linear
    y=hinv(u); curved=y*(1/y).exp()
    if u.lo > hb.hi:return 2*curved
    return I(min((2*linear).lo,(2*curved).lo),max((2*linear).hi,(2*curved).hi))


@lru_cache(maxsize=50000)
def calibration(k,numerator,denominator,delta):
    x=(I(denominator)/(I(numerator)*I(delta))).log()/k
    return (k*cexp(x)).ceil_scaled()


def public_family(g,qs):
    if g.n>20:raise ValueError('exact structure boundary n<=20; use explicit relaxation')
    cuts=set();paths=set();byquery=[]
    for q in qs:
        if g.topological(q)>=0:byquery.append(([],[]));continue
        cs=[]
        for c in sorted(g.cuts(q),key=lambda c:(len(c),c)):
            if not any(set(z).issubset(c) for z in cs):cs.append(c)
        ps=g.paths(q)
        if len(ps)>100000:raise ValueError('exact path boundary 100000')
        cuts.update(cs);paths.update(tuple(sorted(p)) for p in ps)
        byquery.append((cs,[tuple(sorted(p)) for p in ps]))
    families=[{(e,) for e in range(g.m)},cuts,paths,{tuple(range(g.m))}]
    families=[f-{()} for f in families if f-{()}];weights={}
    for fam in families:
        for h in fam:weights[h]=weights.get(h,Fraction(0))+Fraction(1,len(families)*len(fam))
    assert sum(weights.values())==1
    return weights,byquery


class StructureJoint(Certifier):
    """Exact alternative-answer projection via public minimal cuts/simple paths.

    mode=mixture: sum all typed components, dominates legacy max-component B3.
    mode=hierarchical: Thm7 positive-part subset region with same typed masses.
    mode=cardinality: Thm7 all subsets with source uniform cardinality prior.
    """
    def __init__(self,g,qs,eps,delta,mode='mixture'):
        super().__init__(g,qs,eps,delta)
        if mode not in ('mixture','hierarchical','cardinality'):raise ValueError(mode)
        self.mode=mode;tick=time.process_time()
        weights,self.byquery=public_family(g,qs)
        self.supports=sorted(weights);self.mass=[weights[h] for h in self.supports]
        self.matrix=np.zeros((len(weights),g.m),dtype=np.int64)
        for j,h in enumerate(self.supports):self.matrix[j,list(h)]=1
        self.cpu_structure=time.process_time()-tick;tick=time.process_time()
        if mode=='cardinality':
            self.thresholds=np.array([calibration(k,1,g.m*math.comb(g.m,k),str(delta)) for k in range(1,g.m+1)],dtype=np.int64)
        elif mode=='hierarchical':
            self.thresholds=np.array([calibration(len(h),p.numerator,p.denominator,str(delta)) for h,p in zip(self.supports,self.mass)],dtype=np.int64)
        else:
            self.logmass=np.array([(I(p.numerator)/p.denominator).log().floor_scaled() for p in self.mass],dtype=np.int64)
            self.threshold=(1/I(delta)).log().ceil_scaled()
        self.cpu_calibration=time.process_time()-tick

    def _patterns(self,hat,j,ans):
        family=self.byquery[j][0 if ans else 1]
        changed=sorted(set(tuple(e for e in w if bool(hat[e])==bool(ans)) for w in family))
        masks=np.zeros((len(changed),self.g.m),dtype=np.int64)
        for i,c in enumerate(changed):masks[i,list(c)]=1
        return masks

    def _scores(self,base,opposite,masks,verified):
        h=base[None,:]+masks*(opposite-base)[None,:]
        if self.mode=='cardinality':
            return np.max(np.cumsum(np.sort(h,axis=1)[:,::-1],axis=1)-self.thresholds,axis=1)
        vals=h@self.matrix.T
        if self.mode=='hierarchical':return np.max(vals-self.thresholds,axis=1)
        vals=vals+self.logmass
        if not verified:return logsumexp(vals/SCALE,axis=1)*SCALE-self.threshold
        # No floating log-sum-exp can authorize certification. The integer
        # component maxima are only a sufficient screen, summed intervals decide.
        result=[]
        for row in vals:
            mx=int(max(row));s=I(0)
            for v in row:s=s+(I(int(v)-mx)/SCALE).exp()
            result.append((I(mx)/SCALE+s.log()).floor_scaled()-self.threshold)
        return np.array(result,dtype=np.int64)

    def evaluate(self,c,method=None,active=None):
        out=self.top.copy();self.last_meta={'exact_projection':True,'mode':self.mode,'supports':len(self.supports),'verified_queries':0}
        if self.eps==0 or self.g.m==0:return out
        if active is None:active=np.ones(len(self.qs),dtype=bool)
        ph,lh,lq=fast_stats(c.N,c.S,self.eps)
        a,b=rr_i(str(self.eps));stats=None;cache={}
        for j,q in enumerate(self.qs):
            if out[j]>=0 or not active[j]:continue
            if q.tau not in cache:
                r=a+(b-a)*I(q.tau);rf=float((r.lo+r.hi)/2)
                hat=ph>=rf
                lt=c.S*math.log(rf)+(c.N-c.S)*math.log1p(-rf)
                if self.mode=='mixture':base=lq-lh;opposite=lq-lt
                else:
                    raw=np.divide(c.S,c.N,out=np.full(self.g.m,.5),where=c.N>0)
                    lr=c.S*np.log(np.maximum(raw,1e-300))+(c.N-c.S)*np.log(np.maximum(1-raw,1e-300))
                    corr=3*np.log1p(np.log(np.maximum(c.N,1)))
                    base=np.maximum(0,lr-lh-corr);opposite=np.maximum(0,lr-lt-corr)
                cache[q.tau]=(r,hat,base,opposite)
            r,hat,base,opposite=cache[q.tau];ans=int(self.g.reachable(hat,q));masks=self._patterns(hat,j,ans)
            if not len(masks):continue
            # Screening has no authority; a tiny screening margin can only delay.
            if min(self._scores(base*SCALE,opposite*SCALE,masks,False))<=0:continue
            if stats is None:stats=verified_stats(c.N,c.S,self.eps) if self.mode=='mixture' else [None]*self.g.m
            vhat=[];vbase=[];vopp=[];uncertain=False
            for e,(n,s) in enumerate(zip(c.N,c.S)):
                n=int(n);s=int(s)
                if self.mode=='mixture':v,ll,qq,_,_=stats[e]
                else:
                    raw=I(s)/n if n else I('.5');v=I(max(a.lo,min(b.lo,raw.lo)),max(a.hi,min(b.hi,raw.hi)))
                    ll=loglik_i(n,s,v)
                if q.tau==.5:vhat.append(s*2>=n)
                elif v.lo>=r.hi:vhat.append(True)
                elif v.hi<r.lo:vhat.append(False)
                elif not n:vhat.append(q.tau<=.5)
                else:uncertain=True;break
                lt=loglik_i(n,s,r)
                if self.mode=='mixture':vbase.append((qq-ll).floor_scaled());vopp.append((qq-lt).floor_scaled())
                else:
                    lr=I(0) if not n else (I(0) if s==0 or s==n else loglik_i(n,s,I(s)/n))
                    corr=3*(1+I(max(n,1)).log()).log()
                    vbase.append(max(0,(lr-ll-corr).floor_scaled()));vopp.append(max(0,(lr-lt-corr).floor_scaled()))
            if uncertain:continue
            vhat=np.array(vhat)
            if int(self.g.reachable(vhat,q))!=ans:continue
            masks=self._patterns(vhat,j,ans)
            margin=int(min(self._scores(np.array(vbase,dtype=np.int64),np.array(vopp,dtype=np.int64),masks,True)))
            if margin>0:out[j]=ans;self.last_meta['verified_queries']+=1
            self.last_meta['margin_integer_last']=margin
        return out

class StructureRelaxation(StructureJoint):
    """Explicit conservative componentwise projection, never advertised exact."""
    def __init__(self,g,qs,eps,delta,mode='mixture',same_supports=False):
        if same_supports:
            super().__init__(g,qs,eps,delta,mode)
        else:
            Certifier.__init__(self,g,qs,eps,delta);self.mode=mode;tick=time.process_time()
            # Public deterministic spanning forest, paths and fundamental cuts.
            import networkx as nx
            tree=nx.Graph();tree.add_nodes_from(range(g.n));net=nx.Graph();net.add_nodes_from(range(g.n));net.add_edges_from(g.edges)
            for comp in nx.connected_components(net):
                for u,v in nx.bfs_edges(net,min(comp),sort_neighbors=sorted):tree.add_edge(u,v)
            lookup={frozenset(e):i for i,e in enumerate(g.edges)};paths=set();cuts=set()
            for q in qs:
                if nx.has_path(tree,q.s,q.t):
                    path=nx.shortest_path(tree,q.s,q.t);paths.add(tuple(sorted(lookup[frozenset(e)] for e in zip(path,path[1:]))))
            for u,v in list(tree.edges):
                tree.remove_edge(u,v);left=nx.node_connected_component(tree,u);cuts.add(tuple(e for e,(x,y) in enumerate(g.edges) if (x in left)!=(y in left)));tree.add_edge(u,v)
            families=[{(e,) for e in range(g.m)},paths,cuts,{tuple(range(g.m))}];families=[f-{()} for f in families if f-{()}];weights={}
            for fam in families:
                for h in fam:weights[h]=weights.get(h,Fraction(0))+Fraction(1,len(families)*len(fam))
            self.supports=sorted(weights);self.mass=[weights[h] for h in self.supports];self.matrix=np.zeros((len(weights),g.m),dtype=np.int64)
            for j,h in enumerate(self.supports):self.matrix[j,list(h)]=1
            self.cpu_structure=time.process_time()-tick;tick=time.process_time()
            if mode=='hierarchical':self.thresholds=np.array([calibration(len(h),p.numerator,p.denominator,str(delta)) for h,p in zip(self.supports,self.mass)],dtype=np.int64)
            elif mode=='mixture':
                self.logmass=np.array([(I(p.numerator)/p.denominator).log().floor_scaled() for p in self.mass],dtype=np.int64);self.threshold=(1/I(delta)).log().ceil_scaled()
            else:raise ValueError('relaxation only supports explicit typed families')
            self.cpu_calibration=time.process_time()-tick
        self.same_supports=same_supports

    def evaluate(self,c,method=None,active=None):
        out=self.top.copy();self.last_meta={'exact_projection':False,'conservative_componentwise':True,'supports':len(self.supports),'verified_queries':0,'same_supports':self.same_supports}
        if self.eps==0 or self.g.m==0:return out
        if active is None:active=np.ones(len(self.qs),dtype=bool)
        a,b=rr_i(str(self.eps));stats=verified_stats(c.N,c.S,self.eps);cache={}
        for j,q in enumerate(self.qs):
            if out[j]>=0 or not active[j]:continue
            if q.tau not in cache:
                r=a+(b-a)*I(q.tau);hat=[];base=[];opp=[];bad=False
                for n,s,(v,ll,qq,_,_) in zip(c.N,c.S,stats):
                    n=int(n);s=int(s)
                    if q.tau==.5:hat.append(s*2>=n)
                    elif v.lo>=r.hi:hat.append(True)
                    elif v.hi<r.lo:hat.append(False)
                    elif n==0:hat.append(q.tau<=.5)
                    else:bad=True;break
                    lt=loglik_i(n,s,r)
                    if self.mode=='mixture':base.append((qq-ll).floor_scaled());opp.append((qq-lt).floor_scaled())
                    else:
                        lr=I(0) if not n or s in (0,n) else loglik_i(n,s,I(s)/n);corr=3*(1+I(max(n,1)).log()).log()
                        base.append(max(0,(lr-ll-corr).floor_scaled()));opp.append(max(0,(lr-lt-corr).floor_scaled()))
                cache[q.tau]=None if bad else (np.array(hat),np.array(base,dtype=np.int64),np.array(opp,dtype=np.int64))
            if cache[q.tau] is None:continue
            hat,base,opp=cache[q.tau];ans=int(self.g.reachable(hat,q));values=[]
            for h,p in zip(self.supports,self.mass):
                weights=[max(0,int(opp[e]-base[e])) if e in h and bool(hat[e])==bool(ans) else 0 for e in range(self.g.m)]
                z,_=self.g.oracle(weights,q,bool(ans));values.append(int(sum(base[list(h)]))+int(z))
            if self.mode=='hierarchical':margin=max(int(v)-int(t) for v,t in zip(values,self.thresholds))
            else:
                vals=[int(v)+int(p) for v,p in zip(values,self.logmass)];mx=max(vals);s=I(0)
                for v in vals:s=s+(I(v-mx)/SCALE).exp()
                margin=(I(mx)/SCALE+s.log()).floor_scaled()-self.threshold
            if margin>0:out[j]=ans;self.last_meta['verified_queries']+=1
            self.last_meta['margin_integer_last']=margin
        return out
