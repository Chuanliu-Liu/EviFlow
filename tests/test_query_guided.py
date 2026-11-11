import itertools
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.certifier import Certifier, Counts
from src.graph import Graph, Query
from src.numerics import rr
from src.query_guided import (EdgewiseLabelBackend, ExactMES, GraphMES, ProvenanceUnavailable,
    QueryGuidedSampler, exact_path_terms, general_utilities, next_probability,
    planning_probabilities, simplify_terms)


def test_posterior_is_observed_rr_truncated_integral():
    counts = Counts(np.array([0, 1, 8, 14, 80]), np.array([0, 1, 3, 10, 1]))
    for epsilon, tau in [(1., .5), (2., .05), (1., .8)]:
        a, b = rr(epsilon)
        r = a + (b - a) * tau
        actual = planning_probabilities(counts, epsilon, tau)
        expected = []
        for n, s in zip(counts.N, counts.S):
            f = lambda u: u ** int(s) * (1-u) ** int(n-s)
            expected.append(quad(f, r, b, epsabs=1e-100)[0] /
                            quad(f, a, b, epsabs=1e-100)[0])
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
        assert actual[0] == pytest.approx(1-tau)


def test_complete_provenance_and_explicit_boundary():
    graph = Graph(4, [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)])
    query = Query(0, 3)
    terms = exact_path_terms(graph, query)
    assert len(terms) == 4
    for labels in itertools.product([False, True], repeat=graph.m):
        dnf_answer = any(all(labels[e] for e in term) for term in terms)
        assert dnf_answer == graph.reachable(labels, query)
    with pytest.raises(ProvenanceUnavailable, match='no paths returned'):
        exact_path_terms(graph, query, max_paths=3)
    # No incomplete output is installed in the graph's path cache.
    assert not graph._paths
    with pytest.raises(ProvenanceUnavailable, match='cap=18'):
        ExactMES((frozenset(range(19)),))


def test_qgr_general_source_equations_and_partial_valuation():
    terms = [(frozenset([0, 1]), frozenset([1, 2])), (frozenset([0, 2]),)]
    p = np.array([.2, .8, .6])
    eta = .001
    weighted = [(t, np.prod(p[list(t)])/len(t)) for formula in terms for t in formula]
    alpha = (1+eta)/(eta+min(weight for _, weight in weighted))
    expected_ro = [1-p[e] + max(alpha*(weight+eta) for term, weight in weighted if e in term)
                   for e in range(3)]
    expected_false = [(1-p[e])*sum(e in term for formula in terms for term in formula)
                      for e in range(3)]
    assert np.allclose(general_utilities(terms, [p, p], 3, True), expected_ro)
    assert np.allclose(general_utilities(terms, [p, p], 3, False), expected_false)
    assert simplify_terms(terms[0], [1, -1, 0]) == (frozenset([1]),)


def test_edge_label_backend_matches_existing_cs_all_thresholds():
    m = 3
    graph = Graph(m+1, [(0, e+1) for e in range(m)])
    taus = [.05, .5, .8]
    queries = [Query(0, e+1, tau) for tau in taus for e in range(m)]
    counts = Counts(np.array([1000]*m), np.array([700, 300, 500]))
    backend = EdgewiseLabelBackend(m, taus, 1., .05)
    labels = backend.update(counts)
    reference = Certifier(graph, queries, 1., .05).evaluate(counts, 'edgewise')
    assert np.array_equal(np.concatenate([labels[t] for t in taus]), reference)
    one_bit = Counts(np.ones(m, dtype=int), np.ones(m, dtype=int))
    fresh_backend = EdgewiseLabelBackend(m, taus, 1., .05)
    assert all(np.all(x == -1) for x in fresh_backend.update(one_bit).values())
    assert not fresh_backend.reconstruct_all(one_bit, .5)


def test_qgr_charged_sticky_probe_and_completion():
    graph = Graph(3, [(0, 1), (1, 2)])
    sampler = QueryGuidedSampler(graph, [Query(0, 2)], 1., .05, 'qgr')
    counts = Counts.zeros(2)
    original_n = counts.N.copy()
    first = sampler.allocate(counts, 7)
    assert np.array_equal(counts.N, original_n)  # no free observations internally
    edge = int(np.flatnonzero(first)[0])
    assert first[edge] == 7 and sampler.completed_actions == 0
    counts.update(first, first)
    second = sampler.allocate(counts, 13)
    assert second[edge] == 13  # one noisy bit/batch never completes the probe
    counts.N[edge], counts.S[edge] = 1000, 800
    third = sampler.allocate(counts, 11)
    assert third[1-edge] == 11 and sampler.completed_actions == 1
    assert sampler.meta['direction'] == 'false_terms'
    counts.N[:], counts.S[:] = 1000, 800
    continuation = sampler.allocate(counts, 9)
    assert continuation.sum() == 9 and sampler.source_complete
    assert sampler.meta['uniform_continuation'] is True
    assert sampler.requested_reports == 7+13+11+9
    json.dumps(sampler.meta)


def independent_mes(terms, labels, errors):
    variables = sorted(set().union(*terms))
    observed_answer = any(all(labels[e] for e in term) for term in terms)
    likelihoods = []
    for world in itertools.product([False, True], repeat=len(variables)):
        mapping = dict(zip(variables, world))
        answer = any(all(mapping[e] for e in term) for term in terms)
        if answer != observed_answer:
            likelihoods.append(math.prod((1-errors[e]) if mapping[e] == labels[e] else errors[e]
                                        for e in variables))
    return max(likelihoods, default=0.)


def test_mes_source_definition_and_risky_endpoint_exhaustive():
    terms = (frozenset([0, 1]), frozenset([1, 2]))
    engine = ExactMES(terms)
    for labels in itertools.product([False, True], repeat=3):
        for errors in [np.array([.2, .3, .4]), np.full(3, .5), np.array([0., .2, .3])]:
            expected = independent_mes(terms, labels, errors)
            log_score, safe, _ = engine.evaluate(labels, errors)
            assert math.exp(log_score) == pytest.approx(expected)
            expected_safe = []
            for e in range(3):
                if errors[e] == 0:
                    continue
                zero = errors.copy()
                zero[e] = 0.
                if independent_mes(terms, labels, zero) <= expected+1e-14:
                    expected_safe.append(e)
            assert tuple(expected_safe) == safe
            # For each classified safe edge all intermediate reductions are safe.
            for e in safe:
                for ratio in [.1, .5, .9]:
                    reduced = errors.copy()
                    reduced[e] *= ratio
                    assert independent_mes(terms, labels, reduced) <= expected+1e-14


def test_mes_paper_risky_example_and_improvement_sets():
    # AND of three positive input labels: the paper's Example 8/10 pattern.
    terms = (frozenset([0, 1, 2]),)
    engine = ExactMES(terms)
    labels = np.ones(3, dtype=bool)
    errors = np.array([.3, .2, .4])
    score, safe, _ = engine.evaluate(labels, errors)
    assert math.exp(score) == pytest.approx(.7*.8*.4)
    assert not safe
    assert engine.improvement_set(labels, safe, [2, 3, 1]) == (0, 1, 2)
    # Positive OR: every variable is safe; pick lowest report-cost estimate.
    or_engine = ExactMES((frozenset([0]), frozenset([1])))
    _, safe, _ = or_engine.evaluate([True, True], [.2, .3])
    assert safe == (0, 1)
    assert or_engine.improvement_set([True, True], safe, [3, 2]) == (1,)
    # Cheapest false-variable hitting set across two terms is the shared variable.
    cover = ExactMES((frozenset([0, 1]), frozenset([1, 2])))
    assert cover.improvement_set([False]*3, (), [2, 1, 2]) == (1,)


def test_mes_next_probability_is_attainable_and_strict():
    assert next_probability([.2, .3]) == pytest.approx(1/6)
    for p in [.5, .2, 1e-12, 1e-20, np.nextafter(0., 1.)]:
        target = next_probability([0., p])
        assert 0 <= target < p
    assert next_probability([0., 0.]) == 0
    # A previously problematic <1e-12 target is fulfilled by raw proxy error 0.
    graph = Graph(2, [(0, 1)])
    sampler = QueryGuidedSampler(graph, [Query(0, 1)], 1., .05, 'mesreduce')
    sampler.pending = (0, (0,), next_probability([1e-12]))
    counts = Counts(np.array([1000]), np.array([900]))
    allocation = sampler.allocate(counts, 8)
    assert allocation.sum() == 8
    assert sampler.pending is None and sampler.meta['uniform_continuation']


def test_mes_pilot_retries_and_reports_are_all_charged():
    graph = Graph(3, [(0, 1), (1, 2), (0, 2)])
    sampler = QueryGuidedSampler(graph, [Query(0, 2)], 1., .05, 'mesreduce')
    counts = Counts.zeros(3)
    requested = 0
    for batch in [2, 5, 9, 3]:
        prior_n, prior_s = counts.N.copy(), counts.S.copy()
        allocation = sampler.allocate(counts, batch)
        assert allocation.sum() == batch
        assert np.array_equal(counts.N, prior_n) and np.array_equal(counts.S, prior_s)
        counts.update(allocation, allocation//2)
        requested += batch
        assert counts.N.sum() == requested == sampler.requested_reports
        json.dumps(sampler.meta)
    assert np.all(counts.N > 0)


def test_numerical_planning_exhaustion_is_not_certified_completion():
    graph = Graph(2, [(0, 1)])
    sampler = QueryGuidedSampler(graph, [Query(0, 1)], 1., .05, 'mesreduce')
    counts = Counts(np.array([1]), np.array([1]))
    # Even an overconfident/numerically rounded planner cannot authorize output.
    sampler._probabilities = lambda c: {.5: np.array([1.])}
    assert sampler.allocate(counts, 8).sum() == 8
    assert sampler.meta['uniform_continuation']
    assert sampler.meta['planning_exhausted_without_delivery']
    assert not sampler.source_complete
    assert sampler.backend.labels[.5][0] == -1


def test_medium_exact_adapters_and_reference_world_boundary():
    graph = Graph(7, list(itertools.combinations(range(7), 2)))
    queries = [Query(0, 6)]
    qgr = QueryGuidedSampler(graph, queries, 1., .05, 'qgr')
    assert len(qgr.terms[0]) == sum(math.factorial(5)//math.factorial(5-k) for k in range(6))
    with pytest.raises(ProvenanceUnavailable, match='21 relevant variables'):
        QueryGuidedSampler(graph, queries, 1., .05, 'mesreduce', mes_engine='worlds')
    assert qgr.allocate(Counts.zeros(graph.m), 128).sum() == 128
    mes = QueryGuidedSampler(graph, queries, 1., .05, 'mesreduce')
    assert mes.allocate(Counts.zeros(graph.m), 128).sum() == 128
    assert mes.meta['mes_engine'] == 'graph'


def test_graph_mes_exact_score_safe_sets_and_hitting_set_costs():
    cases = [Graph(3, [(0, 1), (1, 2), (0, 2)]),
             Graph(4, [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)]),
             Graph(4, [(0, 1), (1, 2), (1, 3)])]
    for graph in cases:
        query = Query(0, graph.n-1)
        terms = exact_path_terms(graph, query)
        compiled = GraphMES(graph, query, terms)
        reference = ExactMES(terms)
        for labels in itertools.product([False, True], repeat=graph.m):
            for errors in [np.full(graph.m, .5), np.linspace(.1, .4, graph.m),
                           np.array([0.] + [.2]*(graph.m-1))]:
                expected = independent_mes(terms, labels, errors)
                actual, safe, ans = compiled.evaluate(labels, errors)
                _, reference_safe, reference_ans = reference.evaluate(labels, errors)
                assert math.exp(actual) == pytest.approx(expected)
                assert safe == reference_safe and ans == reference_ans
            costs = np.arange(1, graph.m+1, dtype=float)
            got = compiled.improvement_set(labels, (), costs)
            exact = reference.improvement_set(labels, (), costs)
            assert sum(costs[e] for e in got) == sum(costs[e] for e in exact)
            if graph.reachable(labels, query):
                assert any(frozenset(got) == t for t in terms)
            else:
                assert all(set(got) & t for t in terms)
                assert not any(labels[e] for e in got)
