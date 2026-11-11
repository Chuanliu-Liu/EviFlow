import itertools,math
from fractions import Fraction as F
import numpy as np
import mpmath as mp
from src.graph import Graph,Query
from src.certifier import Counts,Certifier,METHODS
from src.numerics import I,verified_edge,fast_stats,rr
from src.sampling import information_rows,fractional_target,integral_target

def independent_reachable(n,edges,labels,s,t):
    # Union-find is independent of Graph.reachable's DFS.
    parent=list(range(n))
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    for (u,v),on in zip(edges,labels):
        if on:parent[root(u)]=root(v)
    return root(s)==root(t)

def test_dense_label_enumeration():
    rng=np.random.default_rng(87001);checked=0
    for nv in range(2,6):
        all_e=list(itertools.combinations(range(nv),2))
        for trial in range(12):
            edges=[e for e in all_e if rng.random()<.85]
            g=Graph(nv,edges);q=Query(0,nv-1)
            if g.topological(q)>=0:continue
            weights=rng.integers(0,100,size=g.m);hat=rng.integers(0,2,size=g.m).astype(bool)
            ans=independent_reachable(nv,edges,hat,q.s,q.t)
            brute=min(sum(int(w) for w,x,y in zip(weights,hat,labels) if x!=y) for labels in itertools.product([False,True],repeat=g.m) if independent_reachable(nv,edges,labels,q.s,q.t)!=ans)
            compiled,_=g.oracle(weights*np.where(ans,hat,~hat),q,ans)
            assert compiled==brute
            checked+=1
    assert checked>=30

def test_conditional_expectation_exact():
    # Exact rational arithmetic for arbitrary adaptive history, all selected edges.
    mu=[F(2,5),F(3,5),F(1,2)];vs=[F(1,4),F(1,2),F(3,4)];rho=F(1,4)
    def evalue(N,S):
        p=F(1)
        for n,s,u in zip(N,S,mu):
            q=sum(v**s*(1-v)**(n-s) for v in vs)/3
            p*=1-rho+rho*q/(u**s*(1-u)**(n-s))
        return p
    for N in itertools.product(range(3),repeat=3):
        for S in itertools.product(*(range(n+1) for n in N)):
            for e in range(3):
                nxt=[]
                for y in (0,1):
                    ns=list(N);ss=list(S);ns[e]+=1;ss[e]+=y;nxt.append(evalue(ns,ss))
                assert (1-mu[e])*nxt[0]+mu[e]*nxt[1]==evalue(N,S)

def test_uniform_integral_enclosures():
    mp.mp.dps=75
    for eps in ('1.0','2.0'):
        a=1/(1+mp.exp(eps));b=1-a
        for n,s in [(0,0),(1,0),(1,1),(10,5),(100,80),(1000,650),(10000,9900)]:
            v,lh,lq,width,_=verified_edge(n,s,eps)
            if n==0:assert lq.lo==0;continue
            # Independent incomplete beta quadrature at high precision.
            exact=mp.log(mp.betainc(s+1,n-s+1,a,b,regularized=False)/(b-a))
            assert mp.mpf(str(lq.lo))<=exact
            assert exact-mp.mpf(str(lq.lo))<=width+mp.mpf('1e-12')

def test_boundaries_and_n0():
    g=Graph(3,[(0,1)]);qs=[Query(0,0),Query(0,2),Query(0,1),Query(0,1,0),Query(0,1,1)]
    for eps in (0.,1.,2.):
        cert=Certifier(g,qs,eps,.05)
        for m in METHODS:assert list(cert.evaluate(Counts.zeros(1),m))==[1,0,-1,1,-1]
    c=Counts(np.array([1000]),np.array([100]));cert=Certifier(g,qs,1.,.05)
    for m in METHODS:assert cert.evaluate(c,m)[-1]==0

def test_evidence_can_decrease():
    from src.numerics import fast_stats
    g=Graph(2,[(0,1)]);c1=Counts(np.array([200]),np.array([140]));c2=Counts(np.array([300]),np.array([140]))
    a1=fast_stats(c1.N,c1.S,1.);a2=fast_stats(c2.N,c2.S,1.)
    l1=a1[2][0]-200*math.log(.5);l2=a2[2][0]-300*math.log(.5)
    assert l2<l1

def test_triangle_costs():
    g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(1,2),Query(0,2)]
    c=Counts(np.array([1000]*3),np.array([650]*3))
    rows=information_rows(g,qs,c,1.);p,a=fractional_target(rows,3);z,b=integral_target(g,qs,c,1.)
    assert np.allclose(p,[1/3]*3)
    assert math.isclose(a['objective']/b['objective'],.75,rel_tol=1e-7)

def test_integer_gate_is_strict():
    x=I('1.00000000000000000000000000001')
    assert x.floor_scaled()<x.ceil_scaled()
    assert not x.floor_scaled()>x.ceil_scaled()
