import itertools
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts
from src.batched_certifier import BatchedCertifier
from src.vector_prior import VectorCertifier
from src.checkpoint_batch import shared_scores,CheckpointCertifier


def test_scores_against_independent_oracles():
    rng=np.random.default_rng(2901)
    for n in (4,6):
        edges=[(u,v) for u in range(n) for v in range(u+1,n) if rng.random()<.65]
        g=Graph(n,edges);qs=[Query(u,v) for u in range(n) for v in range(u+1,n)]
        for _ in range(5):
            weights=rng.integers(0,100,size=g.m).tolist()
            for positive in (False,True):
                got,meta=shared_scores(g,weights,qs,positive)
                expected=[g.oracle(weights,q,positive)[0] for q in qs]
                assert got==expected


def test_batched_decisions_match_reference_mixed_thresholds():
    g=Graph(5,[(0,1),(1,2),(2,3),(0,3),(0,2)])
    qs=[Query(u,v,t) for t in (.3,.5,.7) for u in range(5) for v in range(u,5)]
    rng=np.random.default_rng(2902)
    for i in range(12):
        N=rng.integers(20,2000,size=g.m);S=np.array([rng.integers(0,n+1) for n in N]);c=Counts(N,S)
        active=rng.random(len(qs))>.2
        for method in ('selective','nonselective','edgewise','vector'):
            kwargs={'rho_ratios':[(1,5),(1,3),(1,2),(2,3),(4,5)]} if method=='vector' else {}
            ref=VectorCertifier(g,qs,1.,.05,**kwargs) if method=='vector' else BatchedCertifier(g,qs,1.,.05)
            bat=CheckpointCertifier(g,qs,1.,.05,**kwargs)
            assert np.array_equal(ref.evaluate(c,method,active),bat.evaluate(c,method,active))


def test_no_cross_checkpoint_cache_and_gh_used():
    g=Graph(5,[(u,v) for u in range(5) for v in range(u+1,5)])
    qs=[Query(u,v) for u in range(5) for v in range(u+1,5)]
    cert=CheckpointCertifier(g,qs,1.,.05)
    for s in (650,350,500):
        c=Counts(np.full(g.m,1000),np.full(g.m,s))
        expected=BatchedCertifier(g,qs,1.,.05).evaluate(c)
        assert np.array_equal(expected,cert.evaluate(c))
        if s==650:assert cert.last_meta['gomory_hu_trees']>0
        if s==350:assert cert.last_meta['dijkstra_sources']>0
