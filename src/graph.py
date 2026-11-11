from dataclasses import dataclass
from functools import cached_property
import itertools
import numpy as np
import networkx as nx

@dataclass(frozen=True)
class Query:
    s: int
    t: int
    tau: float = .5

class Graph:
    def __init__(self, n, edges):
        self.n = int(n)
        self.edges = tuple(tuple(map(int,e)) for e in edges)
        self.m = len(edges)
        if len(set(tuple(sorted(e)) for e in self.edges)) != self.m:
            raise ValueError('Parallel edges are not supported')
        if any(u==v or min(u,v)<0 or max(u,v)>=n for u,v in edges):
            raise ValueError('Invalid edge')
        self.adj = [[] for _ in range(n)]
        for i,(u,v) in enumerate(edges):
            self.adj[u].append((v,i)); self.adj[v].append((u,i))
        self._cuts={}; self._paths={}

    def reachable(self, active, q):
        seen={q.s}; stack=[q.s]
        while stack:
            u=stack.pop()
            for v,i in self.adj[u]:
                if active[i] and v not in seen:
                    seen.add(v); stack.append(v)
        return q.t in seen

    def topological(self,q):
        if q.s==q.t: return 1
        if not self.reachable([True]*self.m,q): return 0
        if q.tau==0: return 1
        return -1

    def paths(self,q):
        key=(q.s,q.t)
        if key not in self._paths:
            out=[]
            def rec(u,seen,es):
                if u==q.t: out.append(tuple(es)); return
                for v,e in self.adj[u]:
                    if v not in seen: rec(v,seen|{v},es+[e])
            rec(q.s,{q.s},[])
            self._paths[key]=out
        return self._paths[key]

    def cuts(self,q):
        key=(q.s,q.t)
        if key not in self._cuts:
            if self.n>20: raise ValueError('Dense cut enumeration restricted to n<=20')
            rest=[i for i in range(self.n) if i not in key]; cuts=set()
            for mask in range(1<<len(rest)):
                left={q.s}|{v for i,v in enumerate(rest) if mask>>i&1}
                cuts.add(tuple(i for i,(u,v) in enumerate(self.edges) if (u in left)!=(v in left)))
            self._cuts[key]=sorted(cuts)
        return self._cuts[key]

    def oracle(self,weights,q,positive):
        """Independent networkx oracle. Preserves Python integer capacities."""
        g=nx.Graph();g.add_nodes_from(range(self.n))
        for i,(u,v) in enumerate(self.edges):
            w=weights[i].item() if isinstance(weights[i],np.generic) else weights[i]
            g.add_edge(u,v,weight=w,capacity=w)
        if positive:
            value,part=nx.minimum_cut(g,q.s,q.t,capacity='capacity')
            left=part[0]
            witness=tuple(i for i,(u,v) in enumerate(self.edges) if (u in left)!=(v in left))
        else:
            try:
                p=nx.shortest_path(g,q.s,q.t,weight='weight')
            except nx.NetworkXNoPath: return float('inf'),()
            lookup={frozenset(e):i for i,e in enumerate(self.edges)}
            witness=tuple(lookup[frozenset((u,v))] for u,v in zip(p,p[1:]))
            value=sum(weights[i] for i in witness)
        return value,witness

    def alternatives(self,hat,q):
        """Changed-edge sets of every cut (TRUE) or path (FALSE)."""
        ans=self.reachable(hat,q)
        witnesses=self.cuts(q) if ans else self.paths(q)
        return sorted(set(tuple(e for e in a if bool(hat[e])==ans) for a in witnesses))

    def complete_certificates(self,hat,q):
        ans=self.reachable(hat,q)
        family=self.paths(q) if ans else self.cuts(q)
        return [a for a in family if all(bool(hat[e])==ans for e in a)]

    def structure_family(self,queries):
        """Public supports, fixed before reports; same for all labels."""
        out={tuple(range(self.m))}|{(e,) for e in range(self.m)}
        for q in queries:
            if self.topological(q)<0:
                out.update(self.paths(q)); out.update(self.cuts(q))
        out.discard(())
        # Remove duplicate supersets only if equal (larger supports can pool evidence).
        return sorted(out)

    def label_table(self,queries):
        if self.m>10: raise ValueError('Exact structure baseline restricted to <=10 edges')
        masks=np.arange(1<<self.m,dtype=np.int64)
        labels=(masks[:,None]>>np.arange(self.m)&1).astype(bool)
        answers=np.array([[self.reachable(x,q) for q in queries] for x in labels])
        return labels,answers
