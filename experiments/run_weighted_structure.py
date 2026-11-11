import experiments.run_main as base
from src.weighted_structure import WeightedStructure
from src.workloads import synthetic_workloads
from pathlib import Path
import json
base.Certifier=WeightedStructure
p=json.loads(Path('configs/protocol_weighted_structure_v1.json').read_text())
for cohort,workloads,seed_start in [('main',synthetic_workloads(),2026100300),('regular',json.loads(Path('configs/workloads_regular_graphs_v1.json').read_text()),2026108300),('repair',synthetic_workloads(),2026104300)]:
 out=Path('results')/f'weighted_structure_{cohort}_v1';out.mkdir(exist_ok=True);p['seed_start']=seed_start
 for w in workloads:
  for eps in p['epsilon']:
   for delta in p['delta']:
    for seed in range(seed_start,seed_start+30):print(json.dumps(base.run(w,eps,delta,seed,'uniform',p,out)),flush=True)
