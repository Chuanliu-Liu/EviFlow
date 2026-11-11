import itertools
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts,Certifier
from src.numerics import fast_stats

def test_structure_projection_against_parameter_grid():
    # A dense interior grid can only give >= the exact sign-cell infimum.
    g=Graph(3,[(0,1),(1,2),(0,2)]);q=Query(0,2);c=Counts(np.array([13,21,7]),np.array([10,9,6]));cert=Certifier(g,[q],2.,.05)
    labels,answers,support=cert.structure_setup();ph,lh,lq=fast_stats(c.N,c.S,2.)
    lt=c.N*np.log(.5);hat=ph>=.5;ans=g.reachable(hat,q)
    h=np.where(labels==hat,lq-lh,lq-lt);exact=np.min(np.max(h@support.T,axis=1)[answers[:,0]!=ans])
    from src.numerics import rr
    a,b=rr(2.)
    for p in itertools.product(np.linspace(a,b,9),repeat=3):
        p=np.array(p)
        if g.reachable(p>=.5,q)==ans:continue
        ll=c.S*np.log(p)+(c.N-c.S)*np.log1p(-p)
        assert np.max(support@(lq-ll))>=exact-1e-10

def test_declared_supports_are_public():
    g=Graph(4,[(0,1),(1,2),(2,3),(0,3),(1,3)]);qs=[Query(0,2),Query(1,3)]
    fam=g.structure_family(qs)
    assert tuple(range(g.m)) in fam
    assert all((e,) in fam for e in range(g.m))
    for q in qs:
        assert all(x in fam for x in g.paths(q))
        assert all(x in fam for x in g.cuts(q))
