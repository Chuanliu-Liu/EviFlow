"""Frozen topology-prior repair, new seeds; original primary evidence retained."""
import experiments.run_main as base
from src.certifier import Certifier
from src.extensions import TopologyPriorCertifier
from src.workloads import synthetic_workloads
from pathlib import Path
import json
p=json.loads(Path('configs/protocol_repair_v1.json').read_text())
for variant,cls in [('original',Certifier),('topology_prior',TopologyPriorCertifier)]:
 out=Path('results')/f'repair_{variant}_v1';out.mkdir(exist_ok=True);base.Certifier=cls
 for w in synthetic_workloads():
  if w['name'] not in p['workloads']:continue
  for eps in p['epsilon']:
   for delta in p['delta']:
    for seed in range(p['seed_start'],p['seed_start']+p['seed_count']):
     for sampler in p['samplers']:
      print(json.dumps(base.run(w,eps,delta,seed,sampler,p,out)),flush=True)
