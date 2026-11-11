import numpy as np
from src.extensions import FWSReportClock,BLUCBThreshold
from src.graph import Graph,Query
from src.certifier import Counts

def test_report_clock_initialization_and_tracking():
 g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,1),Query(0,2),Query(1,2)];c=Counts.zeros(3)
 s=FWSReportClock(g,qs,1.)
 assert list(s.allocate(c,3))==[1,1,1]
 c.update(np.ones(3,dtype=int),np.ones(3,dtype=int));x=s.allocate(c,128)
 assert x.sum()==128
 assert 0<s.meta['forced_reports_in_batch']<128
 assert abs(s.x.sum()-1)<1e-10

def test_blucb_threshold_shares_counts():
 g=Graph(3,[(0,1),(1,2),(0,2)]);qs=[Query(0,2)];c=Counts(np.array([200,200,200]),np.array([120,140,90]))
 s=BLUCBThreshold(g,qs,1.,.05);x=s.allocate(c,128)
 assert x.sum()==128 and np.count_nonzero(x)==1
