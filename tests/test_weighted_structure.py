from fractions import Fraction
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts
from src.weighted_structure import WeightedStructure

def test_prior_and_positive_negative_cases():
 g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(1,2),Query(0,2)]
 cert=WeightedStructure(g,qs,1.,.05)
 assert list(cert.evaluate(Counts.zeros(3)))==[-1,-1,-1]
 assert sum(cert.mass)==Fraction(1)
 assert list(cert.evaluate(Counts(np.full(3,1000),np.full(3,650))))==[1,1,1]
 assert list(cert.evaluate(Counts(np.full(3,1000),np.full(3,350))))==[0,0,0]

def test_minimal_cut_family_removes_redundancy():
 g=Graph(6,[(i,(i+1)%6) for i in range(6)]);qs=[Query(0,3)]
 cert=WeightedStructure(g,qs,1.,.05);cert._setup()
 assert sum(cert.mass)==1
 # The full domain is retained as its own calibrated component.
 assert tuple(range(6)) in cert.supports
 # Every public minimal separating pair must be present.
 for c in g.cuts(qs[0]):
  if len(c)==2:assert c in cert.supports
