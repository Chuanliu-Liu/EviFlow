"""Own-statistic B3 graph-component positive gate; exact fallback unchanged."""
import time
import numpy as np
from .structure_joint_fast import FastStructureJoint

class GraphGateStructureJoint(FastStructureJoint):
    def __init__(self,g,qs,eps,delta,mode='mixture'):
        super().__init__(g,qs,eps,delta,mode);self._gate_hint=None
        if g.m and mode=='mixture':
            tick=time.process_time();mass=dict(zip(self.supports,map(int,self.logmass)))
            self._family_min_mass=[tuple(min((mass[h] for h in family),default=None) for family in pair) for pair in self.byquery]
            self.cpu_structure+=time.process_time()-tick
    def _patterns(self,hat,j,ans):
        self._gate_hint=(j,bool(ans),bool(np.all(hat==bool(ans))))
        return super()._patterns(hat,j,ans)
    def _scores(self,base,opposite,masks,verified):
        if verified and self.mode=='mixture' and self._gate_hint and self._gate_hint[2] and np.all(opposite>=0):
            j,ans,_=self._gate_hint;mass=self._family_min_mass[j][0 if ans else 1]
            if mass is not None:
                value,_=self.g.oracle([int(x) for x in opposite],self.qs[j],ans);lower=int(value)+mass
                # The original directed complete mixture is >= max component-1.
                # Each opposite cell's own cut/path support exceeds this lower.
                if lower>=self.threshold+2:
                    self.last_meta['own_graph_component_gate']=True
                    return np.full(len(masks),lower-self.threshold-1,dtype=np.int64)
        return super()._scores(base,opposite,masks,verified)
