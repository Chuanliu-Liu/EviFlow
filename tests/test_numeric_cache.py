import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts,Certifier
from src.extensions import TopologyPriorCertifier

def test_rho_fixed_before_reports():
 g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(0,2),Query(1,2)]
 c=Counts(np.full(3,1000),np.full(3,650));cert=TopologyPriorCertifier(g,qs,1.,.05)
 assert cert.rho_ratio==(2,3)
 assert np.all(cert.evaluate(c)==1)

def test_repeated_count_compiler():
 g=Graph(5,[(0,1),(1,2),(2,3),(3,4),(4,0)]);qs=[Query(0,2),Query(1,3)]
 c=Counts(np.full(5,1000),np.full(5,650));cert=Certifier(g,qs,1.,.05)
 assert list(cert.evaluate(c))==[1,1]
