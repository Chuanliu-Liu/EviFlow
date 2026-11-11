import inspect
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts
from src.residual import ResidualSampler
from src.checkpoint_batch import CheckpointCertifier

def test_fresh_report_accounting_and_formal_only_retirement():
    g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(0,2),Query(1,2)]
    sampler=ResidualSampler(g,qs,1.,.05);c=Counts.zeros(3)
    n=sampler.allocate(c,128)
    assert n.dtype.kind=='i' and n.sum()==128 and np.all(n>=0)
    c.update(n,(n*.8).astype(int));obj=CheckpointCertifier(g,qs,1.,.05)
    before=sampler.active.copy();n1=sampler.allocate(c,128)
    assert np.array_equal(sampler.active,before)  # predictions cannot retire
    assert n1.sum()==128 and np.all(n1>=0)
    formal=obj.evaluate(c,'selective');sampler.delivered(formal)
    assert np.array_equal(sampler.active,formal<0)
    sampler.delivered(np.full(3,-1));assert np.array_equal(sampler.active,formal<0)

def test_truth_excluded_and_boundary_keeps_whole_workload():
    assert 'truth' not in inspect.signature(ResidualSampler).parameters
    g=Graph(4,[(0,1),(2,3)]);qs=[Query(0,1),Query(2,3)]
    s=ResidualSampler(g,qs,1.,.05);c=Counts(np.array([64,64]),np.array([32,32]))
    for _ in range(3):
        allocation=s.allocate(c,128)
        assert allocation.sum()==128 and s.active.all()
        c.update(allocation,allocation//2)
