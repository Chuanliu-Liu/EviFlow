import numpy as np
import pytest
from src.graph import Graph,Query
from src.certifier import Certifier,Counts,METHODS
from src.batched_certifier import BatchedCertifier

@pytest.mark.parametrize('eps',[0.,1.,2.])
def test_reuse_matches_reference_and_expires_after_update(eps):
    g=Graph(5,[(0,1),(1,2),(0,2),(2,3)])
    qs=[Query(0,1),Query(0,2),Query(1,3),Query(0,3,.2),Query(0,3,.8),Query(0,3),Query(0,4),Query(2,2),Query(0,1,0),Query(0,1,1)]
    old=Certifier(g,qs,eps,.05);new=BatchedCertifier(g,qs,eps,.05)
    rng=np.random.default_rng(87031);c=Counts.zeros(g.m)
    for k in range(15):
        if k:
            n=rng.integers(10,300,size=g.m);s=rng.binomial(n,rng.choice([.25,.5,.75],size=g.m));c.update(n,s)
        for method in METHODS:
            assert np.array_equal(old.evaluate(c,method),new.evaluate(c,method))
            assert new._batch_key is None and new._batch_payload is None

def test_simultaneous_strong_positive_negative_and_repair_prior():
    g=Graph(4,[(0,1),(1,2),(0,2),(2,3)]);qs=[Query(i,j) for i in range(4) for j in range(i+1,4)]
    for ratio in (None,(1,2)):
        for success in ([650]*4,[350]*4,[650,650,650,350]):
            c=Counts(np.full(4,1000),np.array(success))
            old=Certifier(g,qs,1.,.01,rho_ratio=ratio);new=BatchedCertifier(g,qs,1.,.01,rho_ratio=ratio)
            for method in METHODS:assert np.array_equal(old.evaluate(c,method),new.evaluate(c,method))
