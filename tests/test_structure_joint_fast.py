import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts
from src.structure_joint import StructureJoint
from src.structure_joint_fast import FastStructureJoint,cycle_family

def test_two_tick_gate_preserves_original_positive_predicate():
    g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,2)];a=StructureJoint(g,qs,1.,.05);b=FastStructureJoint(g,qs,1.,.05)
    rng=np.random.default_rng(510008)
    for _ in range(40):
        base=-rng.integers(0,10**12,size=3,dtype=np.int64);op=base+rng.integers(0,10**13,size=3,dtype=np.int64);masks=np.array([[int(i>>e&1) for e in range(3)] for i in range(8)],dtype=np.int64)
        assert np.array_equal(a._scores(base,op,masks,True)>0,b._scores(base,op,masks,True)>0)

def test_exact_cycle_family_and_gate_match_bruteforce():
    for m in [6,10]:
        g=Graph(m,[(i,(i+1)%m) for i in range(m)]);qs=[Query(0,m//2),Query(1,m-1)];a=StructureJoint(g,qs,1.,.05);b=FastStructureJoint(g,qs,1.,.05)
        weights,byquery=cycle_family(g,qs)
        assert a.supports==sorted(weights) and a.mass==[weights[h] for h in a.supports]
        assert a.byquery==byquery
        assert a.supports==b.supports and a.mass==b.mass
        for s in [350,500,590,650]:
            c=Counts(np.full(m,1000,dtype=np.int64),np.full(m,s,dtype=np.int64));assert a.evaluate(c).tolist()==b.evaluate(c).tolist()

def test_public_empty_graph_needs_no_statistical_evidence():
    g=Graph(2,[]);qs=[Query(0,1),Query(0,0)]
    assert FastStructureJoint(g,qs,1.,.05).evaluate(Counts.zeros(0)).tolist()==[0,1]
