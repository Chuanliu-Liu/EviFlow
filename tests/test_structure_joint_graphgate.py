import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts
from src.numerics import SCALE
from src.structure_joint_fast import FastStructureJoint
from src.structure_joint_graphgate import GraphGateStructureJoint

def test_own_graph_gate_matches_strong_positive_and_negative_counts():
 g=Graph(6,[(i,(i+1)%6) for i in range(6)]);qs=[Query(0,3),Query(1,5)]
 for s in [350,650]:
  a=FastStructureJoint(g,qs,1.,.05);b=GraphGateStructureJoint(g,qs,1.,.05);c=Counts(np.full(6,1000,dtype=np.int64),np.full(6,s,dtype=np.int64))
  assert a.evaluate(c).tolist()==b.evaluate(c).tolist();assert b.last_meta.get('own_graph_component_gate')

def test_mixed_and_zero_histories_keep_original_fallback():
 g=Graph(4,[(0,1),(1,2),(2,3),(3,0),(0,2)]);qs=[Query(0,2),Query(1,3)];a=FastStructureJoint(g,qs,1.,.05);b=GraphGateStructureJoint(g,qs,1.,.05);rng=np.random.default_rng(510007)
 for _ in range(40):
  n=rng.integers(0,1000,size=5,dtype=np.int64);s=np.array([rng.integers(0,int(x)+1) for x in n],dtype=np.int64);c=Counts(n,s)
  assert a.evaluate(c).tolist()==b.evaluate(c).tolist()
 assert a.evaluate(Counts.zeros(5)).tolist()==b.evaluate(Counts.zeros(5)).tolist()

def test_graph_lower_bound_is_below_each_original_cell_margin():
 g=Graph(6,[(i,(i+1)%6) for i in range(6)]);q=Query(0,3);a=FastStructureJoint(g,[q],1.,.05);b=GraphGateStructureJoint(g,[q],1.,.05);base=np.full(6,-100*SCALE,dtype=np.int64);op=np.array([100,105,110,120,125,130],dtype=np.int64)*SCALE
 for scale in [1,10000]:
  op=np.array([100,105,110,120,125,130],dtype=np.int64)*SCALE*scale
  for ans in [False,True]:
   hat=np.full(6,ans);masks=b._patterns(hat,0,ans);baseline=a._scores(base,op,masks,True);gate=b._scores(base,op,masks,True);assert np.all(gate>0) and np.all(gate<=baseline)
   if ans:
    # Independent vertex-side enumeration checks the integer flow value.
    sums=[]
    for bits in range(1<<4):
     rest=[1,2,4,5];left={0}|{v for k,v in enumerate(rest) if bits>>k&1};sums.append(sum(int(op[e]) for e,(u,v) in enumerate(g.edges) if (u in left)!=(v in left)))
    flow,_=g.oracle(list(map(int,op)),q,True);assert flow==min(sums)
