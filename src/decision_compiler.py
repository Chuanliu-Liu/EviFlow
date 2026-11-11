"""Exact threshold predicates for one evidence version, with checked witnesses.

Classical bounded flow and truncated Dijkstra are used, not claimed as new.
The quotient preserves cuts <= B, including equality. No state is retained
between count versions. Floats nominate only; integer payloads authorize.
"""
import heapq
import math
import time

import networkx as nx
import numpy as np

from .checkpoint_batch import CheckpointCertifier, components, shared_scores
from .numerics import fast_stats, rr, verified_stats
from .vector_prior import VectorCertifier


def _checked_flow(net, residual, source, target):
    """Check against input capacities, not residual-supplied capacity claims."""
    if set(residual) != set(net):
        return None
    divergence = {v: 0 for v in net}
    if residual.number_of_edges() != 2 * net.number_of_edges():
        return None
    for u, v in net.edges:
        if not residual.has_edge(u,v) or not residual.has_edge(v,u):
            return None
        f = residual[u][v].get('flow')
        reverse = residual[v][u].get('flow')
        cap = net[u][v]['capacity']
        if (not isinstance(f, int) or not isinstance(reverse, int)
                or reverse != -f or abs(f) > cap):
            return None
        if residual[u][v].get('capacity') != cap or residual[v][u].get('capacity') != cap:
            return None
        divergence[u] += f
        divergence[v] -= f
    value = divergence[source]
    if value < 0 or divergence[target] != -value:
        return None
    if any(d for v,d in divergence.items() if v not in (source,target)):
        return None
    return value


def _residual_cut(net, residual, source, target):
    reached = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v, attr in net[u].items():
            if v not in reached and attr['capacity'] - residual[u][v]['flow'] > 0:
                reached.add(v)
                stack.append(v)
    if target in reached:
        return None
    return sum(attr['capacity'] for u,v,attr in net.edges(data=True)
               if (u in reached) != (v in reached))


def _checked_exact(net, source, target):
    residual = nx.algorithms.flow.preflow_push(net,source,target,capacity='capacity')
    value = _checked_flow(net,residual,source,target)
    if value is None:
        raise ArithmeticError('exact fallback did not provide a feasible integer flow')
    cut = _residual_cut(net,residual,source,target)
    if cut is None or cut != value:
        raise ArithmeticError('exact fallback flow/cut duality check failed')
    return value


def integer_decisions(g, weights, queries, positive, threshold, *, flow_solver=None,
                      batch_exact=True):
    """Return exact (mincut/distance > B), witness bounds, and measured work.

    A bound is a lower bound when decision=True, an upper bound otherwise.
    Values are not presented as fully optimized graph scores.
    """
    start = time.process_time()
    B = int(threshold)
    if B != threshold or B < 0:
        raise ValueError('nonnegative integer threshold required')
    if len(weights) != g.m:
        raise ValueError('one weight per edge required')
    c = [int(w) for w in weights]
    if any(w < 0 or w != original for w,original in zip(c,weights)):
        raise ValueError('exact nonnegative integer weights required')
    K = B + 1
    meta = dict(graph_builds=0, contractions=0, saturated_edges=0,
                quotient_vertices=g.n, quotient_edges=g.m, bounded_flow_calls=0,
                verified_flow_witnesses=0, verified_cut_witnesses=0,
                exact_fallbacks=0, gomory_hu_trees=0, mincut_calls=0,
                dijkstra_sources=0, settled_vertices=0, edge_relaxations=0,
                edge_scans=g.m, cpu_build=0., cpu_solver=0., cpu_verify=0.)
    decisions = [False]*len(queries)
    bounds = [0]*len(queries)
    net = nx.Graph()
    if positive:
        labels = components(g,[w > B for w in c])
        meta['contractions'] = g.n-len(set(labels))
        meta['saturated_edges'] = sum(w > B for w in c)
        unresolved = []
        for j,q in enumerate(queries):
            if labels[q.s] == labels[q.t]:
                decisions[j],bounds[j] = True,K
            else:
                unresolved.append((j,labels[q.s],labels[q.t]))
        meta['quotient_vertices'] = len(set(labels))
        if not unresolved:
            meta['quotient_edges'] = 0
            meta['cpu_build'] = time.process_time()-start
            return decisions,bounds,meta
        net.add_nodes_from(set(labels))
        for (u,v),w in zip(g.edges,c):
            u,v = labels[u],labels[v]
            if u == v or w == 0:
                continue
            previous = net[u][v]['capacity'] if net.has_edge(u,v) else 0
            net.add_edge(u,v,capacity=min(K,previous+w))
        meta['graph_builds'] = 1
        meta['quotient_edges'] = net.number_of_edges()
        component_by_node = {}
        component_nodes = {}
        for key,part in enumerate(nx.connected_components(net)):
            component_nodes[key] = part
            for node in part:
                component_by_node[node] = key
        groups = {}
        for j,s,t in unresolved:
            if component_by_node[t] != component_by_node[s]:
                bounds[j] = 0
            else:
                groups.setdefault(component_by_node[s],[]).append((j,s,t))
        meta['cpu_build'] = time.process_time()-start
        for items in groups.values():
            sub = net.subgraph(component_nodes[component_by_node[items[0][1]]])
            unique = {(min(s,t),max(s,t)) for _,s,t in items}
            # Same query-count criterion as the strong checkpoint control,
            # applied to the exact threshold quotient, fixed before timings.
            if batch_exact and len(unique) > len(sub)-1:
                tick = time.process_time()
                tree = nx.gomory_hu_tree(sub,capacity='capacity',
                                        flow_func=nx.algorithms.flow.preflow_push)
                meta['cpu_solver'] += time.process_time()-tick
                meta['gomory_hu_trees'] += 1
                meta['mincut_calls'] += len(sub)-1
                # This exact batched fallback is the inherited integer oracle.
                by_source = {}
                for j,s,t in items:
                    by_source.setdefault(s,[]).append((j,t))
                for s,targets in by_source.items():
                    scores = {s: math.inf}
                    stack = [s]
                    while stack:
                        u = stack.pop()
                        for v,attr in tree[u].items():
                            if v not in scores:
                                scores[v] = min(scores[u],attr['weight'])
                                stack.append(v)
                    for j,t in targets:
                        z = scores[t]
                        decisions[j],bounds[j] = z > B,z
                continue
            saved = {}
            for j,s,t in items:
                key = (min(s,t),max(s,t))
                if key in saved:
                    decisions[j],bounds[j] = saved[key]
                    continue
                tick = time.process_time()
                solve = flow_solver or nx.algorithms.flow.dinitz
                residual = solve(sub,s,t,capacity='capacity',cutoff=K)
                meta['bounded_flow_calls'] += 1
                meta['cpu_solver'] += time.process_time()-tick
                tick = time.process_time()
                value = _checked_flow(sub,residual,s,t)
                cut = None if value is None else _residual_cut(sub,residual,s,t)
                if value is not None and value > B:
                    decision,bound = True,value
                    meta['verified_flow_witnesses'] += 1
                elif cut is not None and cut <= B:
                    decision,bound = False,cut
                    meta['verified_cut_witnesses'] += 1
                else:
                    meta['exact_fallbacks'] += 1
                    z = _checked_exact(sub,s,t)
                    decision,bound = z > B,z
                meta['cpu_verify'] += time.process_time()-tick
                saved[key] = decision,bound
                decisions[j],bounds[j] = decision,bound
    else:
        # An edge longer than B cannot belong to a path of length <= B.
        net.add_nodes_from(range(g.n))
        for (u,v),w in zip(g.edges,c):
            if w <= B:
                net.add_edge(u,v,weight=w)
        meta['graph_builds'] = 1
        meta['quotient_edges'] = net.number_of_edges()
        meta['cpu_build'] = time.process_time()-start
        by_source = {}
        for j,q in enumerate(queries):
            by_source.setdefault(q.s,[]).append((j,q.t))
        for s,items in by_source.items():
            tick = time.process_time()
            # Exact arithmetic, all <=B candidates expanded before success.
            todo = {target for _,target in items}
            distances = {}
            seen = {s: 0}
            heap = [(0,s)]
            while heap and todo:
                d,u = heapq.heappop(heap)
                if u in distances:
                    continue
                if d > B:
                    break
                distances[u] = d
                todo.discard(u)
                meta['settled_vertices'] += 1
                for v,attr in net[u].items():
                    meta['edge_relaxations'] += 1
                    candidate = d+attr['weight']
                    if candidate <= B and (v not in seen or candidate < seen[v]):
                        seen[v] = candidate
                        heapq.heappush(heap,(candidate,v))
            for j,t in items:
                decisions[j] = t not in distances
                bounds[j] = K if decisions[j] else distances[t]
            meta['dijkstra_sources'] += 1
            meta['cpu_solver'] += time.process_time()-tick
    return decisions,bounds,meta


def screen_decisions(g, weights, queries, positive, beta):
    """Exact dyadic screening, old float oracle retained in a boundary band.

    The band only selects fallback computation; it is not a certification
    tolerance. No float, dyadic nomination, or screen witness authorizes output.
    """
    floats = [float(w) for w in weights]
    if not math.isfinite(beta) or any(not math.isfinite(w) for w in floats):
        scores,old = shared_scores(g,weights,queries,positive)
        return [z > beta for z in scores],dict(screen_exact_fallback_queries=len(queries),**old)
    ratios = [w.as_integer_ratio() for w in floats]
    shift = max((d.bit_length()-1 for _,d in ratios),default=0)
    integers = [n << (shift-(d.bit_length()-1)) for n,d in ratios]
    guard = 1e-10*max(1.,abs(beta),math.fsum(floats))
    def scaled(x):
        n,d = float(x).as_integer_ratio()
        return (n << shift)//d
    high = scaled(beta+guard)
    low = scaled(beta-guard)
    if high < 0:
        return [True]*len(queries),dict(screen_exact_fallback_queries=0)
    decisions,bounds,meta = integer_decisions(g,integers,queries,positive,high)
    ambiguous = [j for j,(yes,bound) in enumerate(zip(decisions,bounds))
                 if not yes and bound > low]
    if ambiguous:
        # Preserve the original complete batch and its graph-solver strategy.
        # Restricting this fallback to ambiguous targets can switch GH to
        # per-query flow and change a rounded boundary nomination.
        scores,old = shared_scores(g,weights,queries,positive)
        for j in ambiguous:
            decisions[j] = scores[j] > beta
        for k,v in old.items():
            meta[k] = meta.get(k,0)+v
    meta['screen_exact_fallback_queries'] = len(ambiguous)
    return decisions,meta


class DecisionCertifier(CheckpointCertifier):
    """Drop-in compiler; frozen likelihoods, thresholds and acquisition API."""
    def evaluate(self,c,method='selective',active=None):
        if method == 'edgewise':
            return super().evaluate(c,method,active)
        if method not in ('selective','nonselective','vector'):
            raise ValueError(method)
        out = self.top.copy()
        self.last_meta = dict(engine='decision-v1',verified_queries=0)
        if self.eps == 0 or self.g.m == 0:
            return out
        if active is None:
            active = np.ones(len(self.qs),dtype=bool)
        start = time.process_time()
        ph,lh,lq = fast_stats(c.N,c.S,self.eps)
        a,b = rr(self.eps)
        R = lh-lq
        rho = self.rho if method == 'vector' else (1/(self.g.m+1)
               if self.rho_ratio is None else self.rho_ratio[0]/self.rho_ratio[1])
        base = np.logaddexp(np.log1p(-rho),np.log(rho)-R)
        stages = dict(cpu_fast_stats=time.process_time()-start,cpu_screen=0.,
                      cpu_verified_stats=0.,cpu_payload=0.,cpu_integer_graph=0.)
        groups = {}
        for j,q in enumerate(self.qs):
            if active[j] and out[j] < 0:
                groups.setdefault(q.tau,[]).append(j)
        stats = None
        work = {}
        certificates = {}
        for tau,indices in groups.items():
            r = a+(b-a)*tau
            hat = ph >= r
            labels = components(self.g,hat)
            answers = {j:int(labels[self.qs[j].s] == labels[self.qs[j].t]) for j in indices}
            lt = c.S*np.log(r)+(c.N-c.S)*np.log1p(-r)
            gain = np.maximum(0,lh-lt)
            if method in ('selective','vector'):
                w = np.maximum(0,np.logaddexp(np.log1p(-rho),np.log(rho)+gain-R)-base)
                beta = math.log(1/self.delta)-sum(base)
            else:
                w,beta = gain,math.log(1/self.delta)+sum(R)
            candidates = []
            tick = time.process_time()
            for answer in (0,1):
                ids = [j for j in indices if answers[j] == answer]
                if not ids:
                    continue
                decisions,meta = screen_decisions(self.g,w*np.where(answer,hat,~hat),
                    [self.qs[j] for j in ids],bool(answer),beta)
                for k,v in meta.items():
                    work['screen_'+k] = work.get('screen_'+k,0)+v
                candidates.extend(j for j,yes in zip(ids,decisions) if yes)
            stages['cpu_screen'] += time.process_time()-tick
            if not candidates:
                continue
            if stats is None:
                tick = time.process_time()
                stats = verified_stats(c.N,c.S,self.eps)
                stages['cpu_verified_stats'] += time.process_time()-tick
            tick = time.process_time()
            payload = (VectorCertifier.payload(self,c,stats,tau) if method == 'vector'
                       else self._payload(c,stats,tau,method))
            stages['cpu_payload'] += time.process_time()-tick
            if payload is None:
                continue
            hat,vector,B = payload
            labels = components(self.g,hat)
            candidates = [j for j in candidates
                          if int(labels[self.qs[j].s] == labels[self.qs[j].t]) == answers[j]]
            tick = time.process_time()
            for answer in (0,1):
                ids = [j for j in candidates if answers[j] == answer]
                if not ids:
                    continue
                weights = [v if bool(h) == bool(answer) else 0 for v,h in zip(vector,hat)]
                decisions,bounds,meta = integer_decisions(self.g,weights,
                    [self.qs[j] for j in ids],bool(answer),B)
                for k,v in meta.items():
                    work['integer_'+k] = work.get('integer_'+k,0)+v
                for j,yes,bound in zip(ids,decisions,bounds):
                    certificates[j] = dict(predicate=bool(yes),bound=bound,threshold=B,
                                           bound_direction='lower' if yes else 'upper')
                    if yes:
                        out[j] = answer
            stages['cpu_integer_graph'] += time.process_time()-tick
        self.last_meta = dict(engine='decision-v1',verified_queries=int(np.sum((out>=0)&(self.top<0)&active)),
                              predicate_witnesses=certificates,**stages,**work)
        return out
