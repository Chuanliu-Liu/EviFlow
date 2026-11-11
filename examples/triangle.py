"""A constructed count state illustrating the API, not a performance trial.

No unknown population means are inputs to any decision below.
Run from the project root after sourcing experiments/env.sh.
"""
import json,math
from decimal import Decimal
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts,Certifier
from src.numerics import I,verified_stats

g=Graph(3,[(0,1),(1,2),(0,2)])
queries=[Query(0,1),Query(1,2),Query(0,2)]
counts=Counts(np.array([100,100,100]),np.array([65,65,65]))
result={'kind':'constructed illustrative count state, not sampled performance data',
        'epsilon':1.,'global_delta':.05,'N':counts.N.tolist(),'S':counts.S.tolist(),
        'answers':{},'verified_margins':{}}
for method in ('selective','edgewise','nonselective','structure'):
    cert=Certifier(g,queries,epsilon=1.,delta=.05)
    result['answers'][method]=cert.evaluate(counts,method).tolist()
    result['verified_margins'][method]=dict(cert.last_meta)
_,lh,lq,width,_=verified_stats(counts.N,counts.S,1.)[0]
lt=I(100)*I('.5').log();rho=I(1)/4
base=(1-rho+rho*(lq-lh).exp()).log()
weight=(1-rho+rho*(lq-lt).exp()).log()-base
result['conservative_joint_cut_lower']=str((2*weight).lo)
result['conservative_joint_threshold_upper']=str(((I(1)/I('.05')).log()-3*base).hi)
result['edge_mixture_value_lower']=str((lq-lt).exp().lo)
width_up=I(Decimal.from_float(math.nextafter(width,math.inf)))
result['edge_mixture_value_upper']=str((lq+width_up-lt).exp().hi)
result['edgewise_required_value']=60
print(json.dumps(result,indent=2))
