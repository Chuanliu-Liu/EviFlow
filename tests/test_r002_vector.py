from fractions import Fraction
from itertools import combinations,product
import math
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts,Certifier
from src.vector_prior import public_library,optimize_product,VectorCertifier
from src.weighted_structure import WeightedStructure
from src.numerics import fast_stats,rr


def test_library_matches_unchanged_typed():
    g=Graph(4,[(0,1),(1,2),(2,3),(0,3),(0,2)])
    qs=[Query(0,3),Query(1,2)]
    typed=WeightedStructure(g,qs,1.,.05);typed._setup()
    supports,masses=public_library(g,qs)
    assert supports==typed.supports and masses==typed.mass


def test_constant_vector_decisions_identical():
    g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(0,2),Query(1,2)]
    for ratio in [(1,4),(2,3)]:
        ref=Certifier(g,qs,1.,.05,rho_ratio=ratio)
        vec=VectorCertifier(g,qs,1.,.05,[ratio]*3)
        for n in [0,16,64,256,1024]:
            for fractions in product([.2,.5,.8],repeat=3):
                c=Counts(np.full(3,n),np.array([int(n*v) for v in fractions]))
                assert np.array_equal(ref.evaluate(c),vec.evaluate(c))


def test_exhaustive_vector_projection_and_decomposition():
    g=Graph(3,[(0,1),(1,2),(0,2)]);q=Query(0,2)
    rho=np.array([.13,.47,.79]);labels,answers=g.label_table([q])
    for ss in product([5,30,55],repeat=3):
        c=Counts(np.full(3,60),np.array(ss));hatp,lh,lq=fast_stats(c.N,c.S,1.)
        lt=-c.N*math.log(2);hat=hatp>=.5;R=lh-lq;gain=lh-lt
        base=np.logaddexp(np.log1p(-rho),np.log(rho)-R)
        at=np.logaddexp(np.log1p(-rho),np.log(rho)+gain-R)
        ans=g.reachable(hat,q);weights=(at-base)*np.where(ans,hat,~hat)
        z,_=g.oracle(weights,q,ans)
        projected=np.where(labels==hat,base,at).sum(axis=1)
        assert np.isclose(z+sum(base),min(projected[answers[:,0]!=ans]))
        for lab in labels:
            H=lab!=hat
            logpi=np.where(H,np.log(rho),np.log1p(-rho)).sum()
            rem=np.where(H,np.logaddexp(0,np.log1p(-rho)-np.log(rho)+R-gain),
                         np.logaddexp(0,np.log(rho)-np.log1p(-rho)-R)).sum()
            assert np.isclose(np.where(H,at,base).sum(),sum((gain-R)[H])+logpi+rem)


def test_optimizer_family_exact_and_grid_oracle():
    for m,k in [(3,2),(4,2),(5,1)]:
        fam=list(combinations(range(m),k));p=[Fraction(1,len(fam))]*len(fam)
        sol=optimize_product(m,fam,p)
        exact=m*(-(k/m)*math.log(k/m)-(1-k/m)*math.log(1-k/m))-math.log(math.comb(m,k))
        assert abs(sol['primal']-exact)<1e-7 and abs(sol['dual']-exact)<1e-7
    fam=[(0,1),(2,3),(4,5)];sol=optimize_product(6,fam,[Fraction(1,3)]*3)
    exact=math.log(3)+4*math.log(1.5)
    assert abs(sol['primal']-exact)<1e-7
    fam=[(0,),(1,),(0,1)];p=[Fraction(1,4),Fraction(1,4),Fraction(1,2)]
    sol=optimize_product(2,fam,p)
    grid=min(max(math.log(float(ph))-sum(math.log(r[e] if e in H else 1-r[e]) for e in range(2))
                 for H,ph in zip(fam,p)) for r in product(np.linspace(.01,.99,99),repeat=2))
    assert sol['dual']<=grid+1e-8 and sol['primal']<=grid+1e-8 and sol['gap']<1e-7


def test_exact_global_wrong_probability_tiny_horizon():
    # Enumerate every report history, checking any wrong certification over time.
    g=Graph(2,[(0,1)]);qs=[Query(0,1)];delta=.4;eps=2.
    cert=VectorCertifier(g,qs,eps,delta,[(3,5)])
    a,b=rr(eps)
    for theta in [.49,.5,.8]:
        mu=a+(b-a)*theta;wrong_prob=0.
        for bits in product([0,1],repeat=10):
            wrong=False;s=0
            for n,bit in enumerate(bits,1):
                s+=bit;ans=cert.evaluate(Counts(np.array([n]),np.array([s])))[0]
                wrong|=ans>=0 and ans!=int(theta>=.5)
            if wrong:wrong_prob+=mu**sum(bits)*(1-mu)**(10-sum(bits))
        assert wrong_prob<=delta+1e-12


def test_exact_rational_vector_martingale_under_adaptive_selection():
    # Independent polynomial integration, no production mixture approximation.
    from functools import lru_cache
    a,b=Fraction(1,4),Fraction(3,4)
    mus=[Fraction(2,5),Fraction(1,2),Fraction(3,5)]
    rho=[Fraction(1,5),Fraction(1,2),Fraction(4,5)]
    @lru_cache(None)
    def mixture(n,s):
        return sum(Fraction(math.comb(n-s,j)*(-1)**j,s+j+1)*(b**(s+j+1)-a**(s+j+1)) for j in range(n-s+1))/(b-a)
    def evidence(N,S):
        out=Fraction(1)
        for n,s,p,r in zip(N,S,mus,rho):out*=1-r+r*mixture(n,s)/(p**s*(1-p)**(n-s))
        return out
    frontier=[((0,0,0),(0,0,0),Fraction(1))]
    for t in range(7):
        new=[]
        for N,S,prob in frontier:
            e=(t+sum(S))%3;before=evidence(N,S);children=[]
            for bit in (0,1):
                nn=list(N);ss=list(S);nn[e]+=1;ss[e]+=bit
                conditional=mus[e] if bit else 1-mus[e]
                children.append(conditional*evidence(tuple(nn),tuple(ss)))
                new.append((tuple(nn),tuple(ss),prob*conditional))
            assert sum(children)==before
        frontier=new
    assert sum(p*evidence(N,S) for N,S,p in frontier)==1


def test_exact_oracle_for_frozen_typed_triangle():
    # The actual indexed typed library, not only a symmetric surrogate family.
    # Full-domain mass is 1/4; each singleton mass is 1/8. Their two envelope
    # curves meet at rho/(1-rho)=sqrt(2); all pair constraints are slack there.
    g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(1,2),Query(0,2)]
    supports,masses=public_library(g,qs);sol=optimize_product(3,supports,masses)
    exact_rho=2-math.sqrt(2);exact_value=-math.log(4)-3*math.log(exact_rho)
    assert max(abs(x-exact_rho) for x in sol['rho'])<1e-10
    assert abs(sol['primal']-exact_value)<1e-10
    assert abs(sol['dual']-exact_value)<1e-10
    # Independently enumerate all actual library constraints at the exact point.
    losses=[math.log(float(p))-len(H)*math.log(exact_rho)-(3-len(H))*math.log1p(-exact_rho) for H,p in zip(supports,masses)]
    assert abs(max(losses)-exact_value)<1e-12


def test_full_singleton_envelope_bound():
    # Exact normalized full/singleton target has twice the lower-bound masses.
    # Subtract log(2) to recover the typed-target subprobability envelope.
    for m in (2,4,7):
        fam=[(e,) for e in range(m)]+[tuple(range(m))]
        masses=[Fraction(1,2*m)]*m+[Fraction(1,2)]
        sol=optimize_product(m,fam,masses)
        exact=m*math.log1p(m**(-1/(m-1)))-math.log(2)
        assert abs(sol['primal']-exact)<1e-8
        assert abs(sol['dual']-exact)<1e-8
