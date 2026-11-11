"""Source-mapped QGR-General and MESReduce acquisition adapters for RR reports.

The Bayesian probabilities below are *planning proxies*, never certificates.
All certified labels use the existing edgewise KL-mixture rule at delta/m.
See local-source/sources/qgr_mes_source_map.md for native/adapted boundaries.
"""
from functools import lru_cache
import math
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaincc

from .numerics import I, fast_stats, loglik_i, rr, rr_i, verified_stats


class ProvenanceUnavailable(ValueError):
    """An exact source adapter exceeded its declared representation boundary."""


@lru_cache(maxsize=32768)
def _posterior_one(n, s, epsilon, tau):
    a, b = rr(epsilon)
    r = a + (b - a) * tau
    if tau <= 0:
        return 1.0
    if tau >= 1:
        return 0.0
    if n == 0:
        return 1.0 - tau

    def mass(lo, hi):
        # Choose the stable beta tail, rather than subtracting two values near 1.
        c_lo, c_hi = betainc(s + 1, n - s + 1, [lo, hi])
        t_lo, t_hi = betaincc(s + 1, n - s + 1, [lo, hi])
        return float(c_hi - c_lo) if c_hi <= .5 else float(t_lo - t_hi)

    below, above = mass(a, r), mass(r, b)
    if above + below > 1e-280 and min(above, below) >= 0:
        return float(above / (above + below))
    # Extreme counts can place both truncated beta tails below machine range.
    # This is a planner calculation, with no role in numerical authorization.
    def logmass(lo, hi):
        mode = min(hi, max(lo, s / n))
        peak = s * math.log(mode) + (n - s) * math.log1p(-mode)
        val = quad(lambda x: math.exp(s * math.log(x) + (n - s) * math.log1p(-x) - peak),
                   lo, hi, epsabs=1e-12, epsrel=1e-10, limit=100)[0]
        return peak + math.log(val) if val > 0 else -math.inf
    lower, upper = logmass(a, r), logmass(r, b)
    return float(math.exp(upper - np.logaddexp(lower, upper)))


def planning_probabilities(c, epsilon, tau):
    """Uniform-on-[a,b] posterior P(mu >= r(tau) | N,S), observed data only."""
    if epsilon <= 0:
        return np.full(len(c.N), 1.0 - tau)
    return np.asarray([_posterior_one(int(n), int(s), float(epsilon), float(tau))
                       for n, s in zip(c.N, c.S)])


def _minimal_terms(terms):
    out = []
    for term in sorted(set(frozenset(t) for t in terms), key=lambda x: (len(x), tuple(sorted(x)))):
        if not any(prior <= term for prior in out):
            out.append(term)
    return tuple(out)


def exact_path_terms(g, q, max_paths=100000):
    """Enumerate every simple path, or fail; never return truncated provenance."""
    public = g.topological(q)
    if public == 1:
        return (frozenset(),)
    if public == 0:
        return ()
    paths = []
    stack = [(q.s, frozenset([q.s]), ())]
    while stack:
        u, seen, es = stack.pop()
        if u == q.t:
            paths.append(frozenset(es))
            if len(paths) > max_paths:
                raise ProvenanceUnavailable(f'exact path count exceeds cap={max_paths}; no paths returned')
            continue
        for v, e in reversed(g.adj[u]):
            if v not in seen:
                stack.append((v, seen | {v}, es + (e,)))
    return _minimal_terms(paths)


def simplify_terms(terms, labels):
    return _minimal_terms([tuple(e for e in t if labels[e] != 1)
                           for t in terms if not any(labels[e] == 0 for e in t)])


def general_utilities(formulas, probabilities, m, positive_turn, ro_offset=.001):
    """QGR Section 5, General: alternate Eq. (2) and Eq. (3).

    formulas and probabilities correspond to unresolved output tuples. Variable
    scores are summed for Eq. (3), maximized over terms for Eq. (2).
    """
    score = np.full(m, -np.inf)
    if not positive_turn:
        for terms, p in zip(formulas, probabilities):
            for term in terms:
                for e in term:
                    if not np.isfinite(score[e]):
                        score[e] = 0.0
                    score[e] += 1.0 - p[e]
        return score
    weighted = [(term, p, math.prod(float(p[e]) for e in term) / len(term))
                for terms, p in zip(formulas, probabilities) for term in terms if term]
    if not weighted:
        return score
    alpha = (1.0 + ro_offset) / (ro_offset + min(w for _, _, w in weighted))
    for term, p, w in weighted:
        for e in term:
            score[e] = max(score[e], 1.0 - p[e] + alpha * (w + ro_offset))
    return score


class EdgewiseLabelBackend:
    """Retained per-edge labels under the unchanged global delta/m KL-CS rule.

    Thresholds need no extra split: one edge's confidence sequence supports all
    thresholds. The directed interval/strict integer gate matches Certifier.
    """
    def __init__(self, m, taus, epsilon, delta):
        self.m, self.eps, self.delta = m, float(epsilon), float(delta)
        self.labels = {float(t): np.full(m, -1, dtype=np.int8) for t in taus}
        self.cpu = 0.0

    def update(self, c):
        start = time.process_time()
        if self.m == 0 or self.eps <= 0:
            return self.labels
        ph, _, lq = fast_stats(c.N, c.S, self.eps)
        a, b = rr(self.eps)
        nominations = []
        for tau, labels in self.labels.items():
            if tau <= 0:
                labels[:] = 1
                continue
            r = a + (b - a) * tau
            lt = c.S * np.log(r) + (c.N - c.S) * np.log1p(-r)
            for e in np.flatnonzero((labels < 0) & (lq - lt > math.log(self.m / self.delta))):
                nominations.append((tau, int(e)))
        if nominations:
            stats = verified_stats(c.N, c.S, self.eps)
            ai, bi = rr_i(str(self.eps))
            gate = (I(self.m) / I(self.delta)).log().ceil_scaled()
            for tau, e in nominations:
                r = ai + (bi - ai) * I(tau)
                v, _, lqi, _, _ = stats[e]
                if v.lo >= r.hi:
                    side = 1
                elif v.hi < r.lo:
                    side = 0
                elif tau == .5:
                    side = int(int(c.S[e]) * 2 >= int(c.N[e]))
                else:
                    continue
                at = loglik_i(int(c.N[e]), int(c.S[e]), r)
                if (lqi - at).floor_scaled() > gate:
                    self.labels[tau][e] = side
        self.cpu += time.process_time() - start
        return self.labels

    def reconstruct_all(self, c, tau=None):
        """Whether all edge labels at tau (or every registered tau) are known."""
        labels = self.update(c)
        return bool(np.all(labels[float(tau)] >= 0)) if tau is not None else all(
            np.all(v >= 0) for v in labels.values())


class ExactMES:
    """Definition 7 max labeling likelihood and Definition 8 safe variables.

    Exact worlds are enumerated only for variables appearing in the complete
    path provenance. Above max_variables the constructor raises explicitly.
    """
    def __init__(self, terms, max_variables=18):
        self.terms = terms
        self.variables = tuple(sorted(set().union(*terms))) if terms else ()
        k = len(self.variables)
        if k > max_variables:
            raise ProvenanceUnavailable(f'MES exact worlds need {k} relevant variables; cap={max_variables}')
        index = {e: j for j, e in enumerate(self.variables)}
        states = np.arange(1 << k, dtype=np.uint64)
        self.worlds = ((states[:, None] >> np.arange(k, dtype=np.uint64)) & 1).astype(np.int8)
        self.answers = np.zeros(len(states), dtype=bool)
        for term in terms:
            mask = sum(1 << index[e] for e in term)
            self.answers |= (states & np.uint64(mask)) == mask

    def evaluate(self, labels, errors, with_safe=True):
        es = list(self.variables)
        v = np.asarray(labels)[es].astype(bool)
        p = np.asarray(errors, dtype=float)[es]
        if np.any((p < 0) | (p > .5)):
            raise ValueError('MES planner errors must be in [0, .5]')
        answer = any(all(labels[e] for e in term) for term in self.terms)
        opposite = self.answers != answer
        if not np.any(opposite):
            return -math.inf, (), answer
        log_correct = np.log1p(-p)
        with np.errstate(divide='ignore'):
            log_wrong = np.log(p)
        values = np.zeros(len(self.worlds))
        # Avoid inf-inf/0*inf at genuinely zero floating-point posterior tails.
        for j in range(len(es)):
            values += np.where(self.worlds[:, j] == v[j], log_correct[j], log_wrong[j])
        score = float(values[opposite].max())
        safe = []
        if not with_safe:
            return score, (), bool(answer)
        # For fixed other probabilities MES is convex piecewise affine in p_e.
        # Its maximum on [0,p_e] occurs at an endpoint. Test p_e=0 exactly.
        for j, e in enumerate(es):
            if p[j] == 0:
                continue
            match = opposite & (self.worlds[:, j] == v[j])
            endpoint = float(values[match].max() - log_correct[j]) if np.any(match) else -math.inf
            if endpoint <= score + 1e-12:  # planning only; never authorizes delivery
                safe.append(e)
        return score, tuple(safe), bool(answer)

    def improvement_set(self, labels, safe, costs):
        if safe:
            return (min(safe, key=lambda e: (costs[e], e)),)
        if any(all(labels[e] for e in term) for term in self.terms):
            candidates = [tuple(sorted(t)) for t in self.terms if all(labels[e] for e in t)]
            return min(candidates, key=lambda t: (sum(costs[e] for e in t), t))
        # Exact cheapest false-variable hitting set (source Section 5.2).
        false = tuple(e for e in self.variables if not labels[e])
        best, value = None, math.inf
        for bits in range(1, 1 << len(false)):
            chosen = tuple(e for j, e in enumerate(false) if bits >> j & 1)
            cost = sum(costs[e] for e in chosen)
            if cost >= value:
                continue
            selected = frozenset(chosen)
            if all(selected & term for term in self.terms):
                best, value = chosen, cost
        if best is None:
            raise RuntimeError('No exact MES improvement set')
        return best


class GraphMES:
    """Exact source MES objective specialized to threshold reachability.

    Source Section 4.1 optimizes over repaired DNF terms for a FALSE output;
    Section 4.2 solves a weighted hitting set for a TRUE output. For complete
    reachability provenance these objectives are shortest path and minimum cut.
    These are planning probabilities, not likelihood evidence or certification.
    """
    def __init__(self, graph, query, terms):
        self.g, self.q, self.terms = graph, query, terms
        self.variables = tuple(sorted(set().union(*terms))) if terms else ()

    def _score(self, labels, errors):
        if self.g.topological(self.q) >= 0:
            return -math.inf, bool(self.g.topological(self.q))
        labels = np.asarray(labels, dtype=bool)
        errors = np.asarray(errors, dtype=float)
        if np.any((errors < 0) | (errors > .5)):
            raise ValueError('MES planner errors must be in [0, .5]')
        answer = self.g.reachable(labels, self.q)
        related = list(self.variables)
        baseline = float(np.log1p(-errors[related]).sum())
        with np.errstate(divide='ignore'):
            costs = np.log1p(-errors) - np.log(errors)
        relevant_flips = labels == answer
        infinite = relevant_flips & ~np.isfinite(costs)
        finite = np.where(relevant_flips & np.isfinite(costs), costs, 0.)
        sentinel = float(finite.sum()) + 1.0
        weights = np.where(infinite, sentinel, finite)
        cost, witness = self.g.oracle(weights, self.q, bool(answer))
        if any(infinite[e] for e in witness):
            return -math.inf, bool(answer)
        return baseline - float(cost), bool(answer)

    def evaluate(self, labels, errors, with_safe=True):
        score, answer = self._score(labels, errors)
        safe = []
        if with_safe:
            for e in self.variables:
                if errors[e] <= 0:
                    continue
                reduced = np.asarray(errors, dtype=float).copy()
                reduced[e] = 0.
                endpoint, _ = self._score(labels, reduced)
                if endpoint <= score + 1e-12:
                    safe.append(e)
        return score, tuple(safe), answer

    def improvement_set(self, labels, safe, costs):
        if safe:
            return (min(safe, key=lambda e: (costs[e], e)),)
        if self.g.reachable(labels, self.q):
            candidates = [tuple(sorted(t)) for t in self.terms if all(labels[e] for e in t)]
            return min(candidates, key=lambda t: (sum(costs[e] for e in t), t))
        finite = np.asarray([costs[e] if not labels[e] else 0. for e in range(self.g.m)])
        sentinel = float(finite.sum()) + 1.
        weights = np.asarray([sentinel if labels[e] else costs[e] for e in range(self.g.m)])
        _, cut = self.g.oracle(weights, self.q, True)
        if any(labels[e] for e in cut):
            raise RuntimeError('MES false-variable hitting set must exist')
        return tuple(sorted(cut))


def next_probability(errors):
    """MESReduce Section 5.2 example f(n)=1/n, no optional MES threshold."""
    positive = np.asarray(errors)[np.asarray(errors) > 0]
    if not len(positive):
        return 0.0
    q = float(positive.min())
    if q < np.finfo(float).tiny:
        return float(np.nextafter(q, 0.0))
    target = 1.0 / (math.ceil(1.0 / q) + 1)
    return min(target, float(np.nextafter(q, 0.0)))


class QueryGuidedSampler:
    """Charged batch interface. No simulator means or report generator accepted.

    method: 'qgr'/'qgr_general' (history probabilities), 'qgr_ep', or 'mesreduce'.
    The caller records/delivers each returned allocation using fresh users.
    """
    def __init__(self, g, qs, epsilon, delta, method, max_paths=100000, mes_max_variables=18,
                 mes_engine='graph'):
        if method not in ('qgr', 'qgr_general', 'qgr_ep', 'mesreduce'):
            raise ValueError(method)
        if epsilon <= 0 or not 0 < delta < 1:
            raise ValueError('Adapter requires epsilon>0 and 0<delta<1')
        self.g, self.qs, self.eps, self.delta, self.method = g, list(qs), epsilon, delta, method
        self.terms = [exact_path_terms(g, q, max_paths) for q in qs]
        self.backend = EdgewiseLabelBackend(g.m, [q.tau for q in qs], epsilon, delta)
        if mes_engine not in ('graph', 'worlds'):
            raise ValueError(mes_engine)
        self.mes = ([GraphMES(g, q, t) if mes_engine == 'graph' else ExactMES(t, mes_max_variables)
                     for q, t in zip(qs, self.terms)] if method == 'mesreduce' else [])
        self.mes_engine = mes_engine
        self.pending = None
        self.completed_actions = 0
        self.requested_reports = 0
        self.meta = {}
        self.source_complete = False

    def _probabilities(self, c):
        return {q.tau: (np.full(self.g.m, .5) if self.method == 'qgr_ep' else
                        planning_probabilities(c, self.eps, q.tau)) for q in self.qs}

    def _backend_complete(self):
        for q, terms in zip(self.qs, self.terms):
            reduced = simplify_terms(terms, self.backend.labels[q.tau])
            if reduced and reduced != (frozenset(),):
                return False
        return True

    def _qgr(self, c, batch, probabilities):
        labels = self.backend.labels
        if self.pending is not None:
            e, tau = self.pending
            if labels[tau][e] >= 0:
                self.pending = None
                self.completed_actions += 1
        if self.pending is None:
            # Each threshold gives a distinct Boolean probe, but raw reports and
            # confidence sequences remain cached and shared across thresholds.
            candidates = []
            for tau in sorted(probabilities):
                formulas = []
                probs = []
                for q, terms in zip(self.qs, self.terms):
                    if q.tau != tau:
                        continue
                    reduced = simplify_terms(terms, labels[tau])
                    if reduced and reduced != (frozenset(),):
                        formulas.append(reduced)
                        probs.append(probabilities[tau])
                score = general_utilities(formulas, probs, self.g.m, self.completed_actions % 2 == 0)
                for e in np.flatnonzero(np.isfinite(score)):
                    candidates.append((float(score[e]), -int(e), -float(tau), int(e), tau))
            if candidates:
                *_, e, tau = max(candidates)
                self.pending = (e, tau)
        allocation = np.zeros(self.g.m, dtype=np.int64)
        if self.pending is not None:
            allocation[self.pending[0]] = batch
        else:
            # The runner normally stops once all delivery rules certify. If it
            # requests more, deterministic uniform data remain a valid stream.
            for k in range(batch):
                allocation[int(np.argmin(c.N + allocation))] += 1
        self.meta.update(action=list(self.pending) if self.pending else None,
                         direction='RO' if self.completed_actions % 2 == 0 else 'false_terms')
        self.source_complete = self._backend_complete()
        self.meta['uniform_continuation'] = self.pending is None
        self.meta['planning_exhausted_without_delivery'] = self.pending is None and not self.source_complete
        return allocation

    def _mesreduce(self, c, batch, probabilities):
        allocation = np.zeros(self.g.m, dtype=np.int64)
        related = tuple(sorted(set(e for engine in self.mes for e in engine.variables)))
        # Source ReVerify: acquire a charged initial label for every related input.
        unseen = [e for e in related if c.N[e] == 0]
        if unseen:
            for e in unseen[:batch]:
                allocation[e] += 1
            remaining = batch - int(allocation.sum())
            for k in range(remaining):
                allocation[related[k % len(related)]] += 1
            self.meta.update(stage='reverify', improvement_set=[], target_probability=None)
            self.source_complete = False
            self.meta['uniform_continuation'] = False
            self.meta['planning_exhausted_without_delivery'] = False
            return allocation
        errors = {t: np.minimum(p, 1 - p) for t, p in probabilities.items()}
        labels = {t: p >= .5 for t, p in probabilities.items()}
        if self.pending is not None:
            j, chosen, target = self.pending
            tau = self.qs[j].tau
            reduced = simplify_terms(self.terms[j], self.backend.labels[tau])
            if (all(errors[tau][e] <= target for e in chosen) or
                    not reduced or reduced == (frozenset(),)):
                self.pending = None
                self.completed_actions += 1
        if self.pending is None:
            candidates = []
            for j, (q, engine) in enumerate(zip(self.qs, self.mes)):
                # Globally resolved queries legitimately leave the target set.
                reduced = simplify_terms(self.terms[j], self.backend.labels[q.tau])
                if not reduced or reduced == (frozenset(),):
                    continue
                score, safe, answer = engine.evaluate(labels[q.tau], errors[q.tau], with_safe=False)
                if np.isfinite(score):
                    candidates.append((score, -j, j, safe, answer))
            if candidates:
                score, _, j, safe, answer = max(candidates)
                q, engine = self.qs[j], self.mes[j]
                _, safe, _ = engine.evaluate(labels[q.tau], errors[q.tau])
                target = next_probability(errors[q.tau][list(engine.variables)])
                # Planning-only cost extrapolation uses observed RR evidence.
                a, b = rr(self.eps)
                ph = np.clip(np.divide(c.S, c.N, out=np.full(self.g.m, .5), where=c.N > 0), a, b)
                r = a + (b - a) * q.tau
                divergence = ph * np.log(ph / r) + (1 - ph) * np.log((1 - ph) / (1 - r))
                tiny = np.nextafter(0., 1.)
                log_reduction = np.maximum(0., np.log(np.maximum(errors[q.tau], tiny)) -
                                           math.log(max(target, tiny)))
                costs = np.maximum(1., log_reduction / np.maximum(divergence, 1e-8))
                chosen = engine.improvement_set(labels[q.tau], safe, costs)
                self.pending = (j, chosen, target)
                self.meta.update(log_mes=score, safe_variables=list(safe), query=j, planned_answer=int(answer))
        if self.pending is not None:
            j, chosen, target = self.pending
            tau = self.qs[j].tau
            need = [e for e in chosen if errors[tau][e] > target]
            if not need:
                need = list(chosen)
            for k in range(batch):
                allocation[need[k % len(need)]] += 1
            self.meta.update(stage='improve', improvement_set=list(chosen), target_probability=target)
        else:
            for k in range(batch):
                allocation[int(np.argmin(c.N + allocation))] += 1
            self.meta.update(stage='resolved', improvement_set=[], target_probability=None)
        self.source_complete = self._backend_complete()
        self.meta['uniform_continuation'] = self.pending is None
        self.meta['planning_exhausted_without_delivery'] = self.pending is None and not self.source_complete
        return allocation

    def allocate(self, c, batch):
        if int(batch) != batch or batch <= 0 or self.g.m == 0:
            raise ValueError('A positive integer batch and nonempty graph are required')
        if len(c.N) != self.g.m or np.any(c.N < 0) or np.any(c.S < 0) or np.any(c.S > c.N):
            raise ValueError('Invalid observed counts')
        prior_cpu = self.backend.cpu
        self.backend.update(c)
        probabilities = self._probabilities(c)
        allocation = (self._mesreduce(c, int(batch), probabilities) if self.method == 'mesreduce'
                      else self._qgr(c, int(batch), probabilities))
        assert int(allocation.sum()) == batch and np.all(allocation >= 0)
        self.requested_reports += int(allocation.sum())
        self.meta.update(method=self.method, backend_cpu=self.backend.cpu,
                         backend_cpu_last=self.backend.cpu - prior_cpu,
                         observed_reports=int(c.N.sum()), requested_reports=self.requested_reports,
                         completed_actions=self.completed_actions,
                         source_complete=self.source_complete,
                         mes_engine=self.mes_engine if self.method == 'mesreduce' else None,
                         exact_path_counts=[len(t) for t in self.terms],
                         planning_probability='truncated_uniform_beta_posterior' if self.method != 'qgr_ep' else 'constant_half')
        return allocation
