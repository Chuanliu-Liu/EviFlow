import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts
from src.sampling import information_rows,fractional_target,integral_target

def test_single_query_fractional_equals_integral():
    rng=np.random.default_rng(87112)
    for theta in ([.8,.8,.8,.8,.8],[.8,.2,.8,.2,.8],[.2,.2,.2,.2,.2]):
        g=Graph(4,[(0,1),(1,2),(2,3),(0,3),(0,2)]);qs=[Query(0,3)]
        c=Counts(np.full(5,10000),np.array(theta)*10000)
        c.S=c.S.astype(np.int64)
        rows=information_rows(g,qs,c,2.);_,f=fractional_target(rows,5);_,z=integral_target(g,qs,c,2.)
        assert abs(f['objective']-z['objective'])<=1e-6*max(1,z['objective'])

def test_shared_certificates_never_more_costly_than_independent_sum():
    g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(1,2),Query(0,2)]
    c=Counts(np.full(3,1000),np.array([660,650,640]))
    _,shared=integral_target(g,qs,c,1.)
    independent=sum(integral_target(g,[q],c,1.)[1]['objective'] for q in qs)
    assert shared['objective']<independent
