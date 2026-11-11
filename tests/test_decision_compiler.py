import itertools
import math

import networkx as nx
import numpy as np

from src.certifier import Counts
from src.checkpoint_batch import CheckpointCertifier, shared_scores
from src.decision_compiler import DecisionCertifier, integer_decisions, screen_decisions
from src.graph import Graph, Query
from src.numerics import verified_edge


def enumerated_score(g,w,q,positive):
    if positive:
        if q.s == q.t:
            return math.inf
        return min(sum(w[e] for e in cut) for cut in g.cuts(q))
    paths = g.paths(q)
    return min((sum(w[e] for e in path) for path in paths),default=math.inf)


def test_exhaustive_independent_weight_cut_and_path_predicates():
    g = Graph(4,[(0,1),(1,2),(2,3),(0,3),(0,2)])
    qs = [Query(s,t) for s,t in itertools.combinations(range(4),2)]
    for w in itertools.product((0,1,3),repeat=g.m):
        for B in (0,1,2,3,5):
            for positive in (False,True):
                actual,bounds,_ = integer_decisions(g,w,qs,positive,B)
                scores = [enumerated_score(g,w,q,positive) for q in qs]
                assert actual == [z > B for z in scores]
                for yes,bound,z in zip(actual,bounds,scores):
                    assert bound <= z if yes else bound >= z


def test_strict_equal_edge_is_not_contracted_and_parallel_sum_is_retained():
    g = Graph(2,[(0,1)])
    result,_,meta = integer_decisions(g,[5],[Query(0,1)],True,5)
    assert result == [False] and meta['contractions'] == 0
    assert integer_decisions(g,[6],[Query(0,1)],True,5)[0] == [True]
    # High 0--1 contraction creates two parallel low edges to vertex 2.
    g = Graph(3,[(0,1),(0,2),(1,2)])
    assert integer_decisions(g,[9,3,3],[Query(0,2)],True,5)[0] == [True]
    assert integer_decisions(g,[9,2,3],[Query(0,2)],True,5)[0] == [False]


def test_frontier_must_be_complete_and_equality_path_is_reachable():
    g = Graph(3,[(0,1),(1,2),(0,2)])
    # Direct route is too long, but an unexpanded cheap frontier reaches t.
    for B,expected in [(3,True),(4,False),(5,False)]:
        assert integer_decisions(g,[2,2,9],[Query(0,2)],False,B)[0] == [expected]


def test_no_path_zero_threshold_and_same_endpoints():
    g = Graph(4,[(0,1),(1,2)])
    qs = [Query(0,3),Query(0,0),Query(0,2)]
    assert integer_decisions(g,[0,0],qs,False,0)[0] == [True,False,False]
    assert integer_decisions(g,[0,0],qs,True,0)[0] == [False,True,False]


def test_failed_or_infeasible_partial_flow_requires_checked_exact_fallback():
    g = Graph(4,[(0,1),(1,3),(0,2),(2,3)])
    def empty(net,s,t,**kwargs):
        residual = nx.algorithms.flow.utils.build_residual_network(net,'capacity')
        for u,v in residual.edges:
            residual[u][v]['flow'] = 0
        residual.graph['flow_value'] = 0
        return residual
    result,_,meta = integer_decisions(g,[3]*4,[Query(0,3)],True,5,flow_solver=empty)
    assert result == [True] and meta['exact_fallbacks'] == 1
    def corrupt(net,s,t,**kwargs):
        residual = empty(net,s,t)
        residual[s][next(iter(net[s]))]['flow'] = 10000
        residual.graph['flow_value'] = 10000
        return residual
    result,_,meta = integer_decisions(g,[3]*4,[Query(0,3)],True,5,flow_solver=corrupt)
    assert result == [True] and meta['exact_fallbacks'] == 1


def test_float_band_is_only_a_fallback_and_never_rounds_equality_up():
    g = Graph(3,[(0,1),(1,2),(0,2)])
    qs = [Query(0,2)]
    for positive in (False,True):
        w = [0.1,0.2,0.3]
        scores,_ = shared_scores(g,w,qs,positive)
        z = scores[0]
        for beta in (z,np.nextafter(z,0.),np.nextafter(z,math.inf)):
            assert screen_decisions(g,w,qs,positive,beta)[0] == [z > beta]


def test_boundary_fallback_retains_original_complete_batch_solver():
    g = Graph(4,list(itertools.combinations(range(4),2)))
    qs = [Query(s,t) for s,t in itertools.combinations(range(4),2)]
    w = [1.,.1,.2,.3,.4,.5]
    for positive in (False,True):
        scores,_ = shared_scores(g,w,qs,positive)
        for z in scores:
            for beta in (z,np.nextafter(z,0.),np.nextafter(z,math.inf)):
                assert screen_decisions(g,w,qs,positive,beta)[0] == [v > beta for v in scores]


def test_predicate_parity_both_signs_multiple_thresholds_counts_and_active():
    g = Graph(4,[(0,1),(1,2),(0,2),(2,3)])
    qs = [Query(s,t,tau) for tau in (.25,.5,.75,0.,1.)
          for s,t in [(0,2),(0,3),(0,0)]]
    rng = np.random.default_rng(880004)
    for repeat in range(12):
        N = rng.integers(0,300,size=g.m,dtype=np.int64)
        S = np.array([rng.integers(n+1) for n in N],dtype=np.int64)
        counts = Counts(N,S)
        for method in ('selective','nonselective'):
            active = rng.random(len(qs)) > .2
            old = CheckpointCertifier(g,qs,1.,.05)
            new = DecisionCertifier(g,qs,1.,.05)
            verified_edge.cache_clear()
            a = old.evaluate(counts,method,active)
            verified_edge.cache_clear()
            b = new.evaluate(counts,method,active)
            assert np.array_equal(a,b)


def test_integer_range_exceeds_machine_word_without_conversion_loss():
    g = Graph(3,[(0,1),(1,2),(0,2)])
    w = [2**90,2**90,2**90]
    for positive,B in [(True,2**91),(False,2**90)]:
        assert integer_decisions(g,w,[Query(0,2)],positive,B)[0] == [False]
