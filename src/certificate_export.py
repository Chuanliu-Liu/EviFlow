"""R011 append-only evidence exports from an explicitly pinned source tree.

The caller imports this file by path while ``src`` resolves to the frozen tree.
No source stopping predicate is used by the independent verifier.
"""
import base64
import hashlib
import json
from decimal import Decimal
import networkx as nx
import numpy as np
from src.numerics import I, SCALE, PREC, rr_i, loglik_i, verified_stats
from src.checkpoint_batch import CheckpointCertifier

SCHEMA = 'eviflow-release-certificate-v1'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def interval(x):
    return [str(x.lo), str(x.hi)]


def vector_export(g, qs, c, epsilon, delta, tau):
    obj = CheckpointCertifier(g, qs, epsilon, delta)
    stats = verified_stats(c.N, c.S, epsilon)
    payload = obj._payload(c, stats, tau, 'selective')
    if payload is None:
        raise ArithmeticError('frozen directed sign unresolved')
    hat, weights, threshold = payload
    a, b = rr_i(str(epsilon)); r = a + (b-a)*I(tau)
    rho = I(1)/(g.m+1); A = I(0); primitives = []
    for n, s, (v, lh, lq, width, cells) in zip(c.N, c.S, stats):
        lt = loglik_i(int(n), int(s), r)
        base = (1-rho+rho*(lq-lh).exp()).log(); A = A-base
        primitives.append({'mle': interval(v), 'logLhat': interval(lh),
                           'logQ_lower': str(lq.lo), 'logLt': interval(lt),
                           'integral_log_enclosure_width': width, 'cells': cells})
    out = {'tau': tau, 'hat': hat.astype(int).tolist(),
           'weight_lower_scaled': [int(w) for w in weights],
           'threshold_upper_scaled': int(threshold), 'A': interval(A),
           'scale': SCALE, 'numeric_precision': PREC,
           'primitives': primitives,
           'positive_capacity': [int(w) if h else 0 for h,w in zip(hat,weights)],
           'negative_length': [0 if h else int(w) for h,w in zip(hat,weights)]}
    out['id'] = digest(out)
    return out


def witness_export(g, q, answer, vector):
    """One independent per-query capacity constraint, no cross-query sum."""
    weights = vector['positive_capacity' if answer else 'negative_length']
    net = nx.Graph(); net.add_nodes_from(range(g.n))
    for (u,v),w in zip(g.edges,weights):
        net.add_edge(u,v,capacity=w,weight=w)
    if answer:
        residual = nx.algorithms.flow.edmonds_karp(net,q.s,q.t,capacity='capacity')
        # NetworkX excludes zero-capacity edges from its residual network.
        # They still occupy an original-edge slot and must export zero flow.
        flow = [0 if w == 0 else int(residual[u][v]['flow'])
                for (u,v),w in zip(g.edges,weights)]
        return {'kind':'flow', 'signed_edge_flow':flow,
                'value':int(residual.graph['flow_value']),
                'support':[i for i,f in enumerate(flow) if f]}
    distances = nx.single_source_dijkstra_path_length(net,q.s,weight='weight')
    cap = max(distances.values(),default=0)+vector['threshold_upper_scaled']+1
    phi = [int(distances.get(v,cap)) for v in range(g.n)]
    return {'kind':'potential','phi':phi,'value':phi[q.t]-phi[q.s],
            'support':[i for i,((u,v),w) in enumerate(zip(g.edges,weights))
                       if w>0 and phi[u]!=phi[v]]}


def pack_bits(bits):
    return base64.b64encode(np.packbits(bits,bitorder='little').tobytes()).decode()
