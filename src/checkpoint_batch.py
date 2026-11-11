"""Exact graph sharing within one checkpoint. No state crosses evaluate calls."""
import math
import numpy as np
import networkx as nx
from .batched_certifier import BatchedCertifier
from .vector_prior import VectorCertifier
from .numerics import I,rr,fast_stats,verified_stats


def components(g, active):
    parent=list(range(g.n));size=[1]*g.n
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for (u,v),yes in zip(g.edges,active):
        if not yes:continue
        a,b=root(u),root(v)
        if a!=b:
            if size[a]<size[b]:a,b=b,a
            parent[b]=a;size[a]+=size[b]
    return [root(x) for x in range(g.n)]


def shared_scores(g,weights,queries,positive):
    """Same weight vector for every query; return exact scores and solver counts."""
    net=nx.Graph();net.add_nodes_from(range(g.n))
    for (u,v),w in zip(g.edges,weights):
        if isinstance(w,np.generic):w=w.item()
        if w<0:raise ValueError('nonnegative compiler weights required')
        net.add_edge(u,v,capacity=w,weight=w)
    meta={'graph_builds':1,'mincut_calls':0,'gomory_hu_trees':0,'dijkstra_sources':0}
    result=[None]*len(queries)
    if positive:
        # Zero-capacity cross-component edges never contribute to a cut.
        labels=components(g,[w>0 for w in weights]);groups={};vertices={}
        for v,c in enumerate(labels):vertices.setdefault(c,[]).append(v)
        for j,q in enumerate(queries):
            if q.s==q.t:result[j]=math.inf
            elif labels[q.s]!=labels[q.t]:result[j]=0
            else:groups.setdefault(labels[q.s],[]).append((j,q))
        for comp,items in groups.items():
            nodes=vertices[comp];sub=net.subgraph(nodes)
            if len(items)>len(nodes)-1:
                tree=nx.gomory_hu_tree(sub,capacity='capacity',flow_func=nx.algorithms.flow.preflow_push)
                meta['gomory_hu_trees']+=1;meta['mincut_calls']+=len(nodes)-1
                by_source={}
                for j,q in items:by_source.setdefault(q.s,[]).append((j,q.t))
                for source,targets in by_source.items():
                    scores={source:math.inf};stack=[source]
                    while stack:
                        u=stack.pop()
                        for v,attr in tree[u].items():
                            if v not in scores:scores[v]=min(scores[u],attr['weight']);stack.append(v)
                    for j,t in targets:result[j]=scores[t]
            else:
                for j,q in items:
                    result[j]=nx.minimum_cut_value(sub,q.s,q.t,capacity='capacity')
                    meta['mincut_calls']+=1
    else:
        sources={}
        for j,q in enumerate(queries):sources.setdefault(q.s,[]).append((j,q.t))
        for s,items in sources.items():
            distances=nx.single_source_dijkstra_path_length(net,s,weight='weight')
            meta['dijkstra_sources']+=1
            for j,t in items:result[j]=distances.get(t,math.inf)
    return result,meta


class CheckpointCertifier(BatchedCertifier):
    def __init__(self,g,qs,eps,delta,rho_ratio=None,rho_ratios=None):
        if eps<0 or not 0<delta<1:raise ValueError('invalid privacy/risk')
        self.g=g;self.qs=qs;self.eps=eps;self.delta=delta;self.rho_ratio=rho_ratio
        if rho_ratio is not None and not 0<rho_ratio[0]<rho_ratio[1]:raise ValueError('invalid rho')
        labels=components(g,[True]*g.m)
        self.top=np.array([1 if q.s==q.t else 0 if labels[q.s]!=labels[q.t] else 1 if q.tau==0 else -1 for q in qs])
        self._labels=None;self.last_meta={}
        if rho_ratios is not None:
            if len(rho_ratios)!=g.m or any(not 0<a<b for a,b in rho_ratios):raise ValueError('invalid vector')
            self.ratios=tuple(tuple(r) for r in rho_ratios)
            self.rho=np.array([a/b for a,b in rho_ratios]);self.rhoi=[I(a)/b for a,b in rho_ratios]

    def evaluate(self,c,method='selective',active=None):
        if method not in ('selective','nonselective','edgewise','vector'):raise ValueError(method)
        out=self.top.copy();calls={'graph_builds':0,'mincut_calls':0,'gomory_hu_trees':0,'dijkstra_sources':0}
        if self.eps==0 or self.g.m==0:return out
        if active is None:active=np.ones(len(self.qs),dtype=bool)
        ph,lh,lq=fast_stats(c.N,c.S,self.eps);a,b=rr(self.eps);R=lh-lq
        if method=='vector':rho=self.rho
        else:rho=1/(self.g.m+1) if self.rho_ratio is None else self.rho_ratio[0]/self.rho_ratio[1]
        base=np.logaddexp(np.log1p(-rho),np.log(rho)-R)
        groups={}
        for j,q in enumerate(self.qs):
            if active[j] and out[j]<0:groups.setdefault(q.tau,[]).append(j)
        stats=None;all_scores={};verified_queries=0
        for tau,indices in groups.items():
            r=a+(b-a)*tau;hat=ph>=r;labels=components(self.g,hat)
            ans={j:int(labels[self.qs[j].s]==labels[self.qs[j].t]) for j in indices}
            lt=c.S*np.log(r)+(c.N-c.S)*np.log1p(-r);gain=np.maximum(0,lh-lt)
            if method=='edgewise':
                ok=lq-lt>math.log(self.g.m/self.delta)
                pos=components(self.g,ok&hat);neg=components(self.g,~(ok&~hat))
                candidates=[j for j in indices if (pos[self.qs[j].s]==pos[self.qs[j].t] if ans[j] else neg[self.qs[j].s]!=neg[self.qs[j].t])]
            else:
                if method in ('vector','selective'):
                    w=np.maximum(0,np.logaddexp(np.log1p(-rho),np.log(rho)+gain-R)-base)
                    beta=math.log(1/self.delta)-sum(base)
                else:w=gain;beta=math.log(1/self.delta)+sum(R)
                candidates=[]
                for answer in (0,1):
                    ids=[j for j in indices if ans[j]==answer]
                    if not ids:continue
                    scores,meta=shared_scores(self.g,w*np.where(answer,hat,~hat),[self.qs[j] for j in ids],bool(answer))
                    for k,v in meta.items():calls[k]+=v
                    candidates.extend(j for j,z in zip(ids,scores) if z>beta)
            if not candidates:continue
            if stats is None:stats=verified_stats(c.N,c.S,self.eps)
            payload=VectorCertifier.payload(self,c,stats,tau) if method=='vector' else self._payload(c,stats,tau,method)
            if payload is None:continue
            hat,vector,threshold=payload;labels=components(self.g,hat)
            candidates=[j for j in candidates if int(labels[self.qs[j].s]==labels[self.qs[j].t])==ans[j]]
            if method=='edgewise':
                pos=components(self.g,vector&hat);neg=components(self.g,~(vector&~hat))
                for j in candidates:
                    q=self.qs[j]
                    valid=pos[q.s]==pos[q.t] if ans[j] else neg[q.s]!=neg[q.t]
                    if valid:out[j]=ans[j];verified_queries+=1
            else:
                for answer in (0,1):
                    ids=[j for j in candidates if ans[j]==answer]
                    if not ids:continue
                    weights=[v if bool(h)==bool(answer) else 0 for v,h in zip(vector,hat)]
                    scores,meta=shared_scores(self.g,weights,[self.qs[j] for j in ids],bool(answer))
                    for k,v in meta.items():calls[k]+=v
                    for j,z in zip(ids,scores):
                        all_scores[j]=z
                        if z>threshold:out[j]=answer;verified_queries+=1
        self.last_meta={**calls,'verified_queries':verified_queries,'integer_scores':all_scores}
        return out
