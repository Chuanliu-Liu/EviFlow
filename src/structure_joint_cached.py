"""State-identical complete B3 construction; unique prior-mass logs once/instance."""
import time
import numpy as np
from .structure_joint_graphgate import GraphGateStructureJoint
from .structure_joint_fast import is_cycle,cycle_family
from .structure_joint import public_family
from .certifier import Certifier
from .numerics import I

class CachedStructureJoint(GraphGateStructureJoint):
 def __init__(self,g,qs,eps,delta,mode='mixture'):
  if mode!='mixture' or not g.m:
   super().__init__(g,qs,eps,delta,mode);return
  Certifier.__init__(self,g,qs,eps,delta);self.mode=mode;self._gate_hint=None;tick=time.process_time()
  weights,self.byquery=(cycle_family if is_cycle(g) else public_family)(g,qs)
  self.supports=sorted(weights);self.mass=[weights[h] for h in self.supports];self.matrix=np.zeros((len(weights),g.m),dtype=np.int64)
  for j,h in enumerate(self.supports):self.matrix[j,list(h)]=1
  self.cpu_structure=time.process_time()-tick;tick=time.process_time();logs={}
  for p in self.mass:
   if p not in logs:logs[p]=(I(p.numerator)/p.denominator).log().floor_scaled()
  self.logmass=np.array([logs[p] for p in self.mass],dtype=np.int64);self.threshold=(1/I(delta)).log().ceil_scaled();self.cpu_calibration=time.process_time()-tick;self.unique_mass_count=len(logs)
  tick=time.process_time();mass=dict(zip(self.supports,map(int,self.logmass)));self._family_min_mass=[tuple(min((mass[h] for h in family),default=None) for family in pair) for pair in self.byquery];self.cpu_structure+=time.process_time()-tick
