import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,resource
from pathlib import Path
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts,Certifier
from src.numerics import verified_edge
p=json.loads(Path('configs/protocol_scale_v1.json').read_text());out=Path('results/scale_v1');out.mkdir(exist_ok=True)
for m in p['edges']:
 for nq in p['queries']:
  for rep in range(p['replicates']):
   path=out/f'm{m}_q{nq}_r{rep}.json'
   if path.exists():continue
   rng=np.random.default_rng(p['seed_start']+rep);n=(m+9)//10
   edges={(min(i,(i+1)%n),max(i,(i+1)%n)) for i in range(n)}
   while len(edges)<m:
    u,v=map(int,rng.integers(n,size=2))
    if u!=v:edges.add((min(u,v),max(u,v)))
   g=Graph(n,sorted(edges));qs=[Query(i,n//2+i,.5) for i in range(nq)];c=Counts(np.full(m,1000,dtype=np.int64),np.full(m,650,dtype=np.int64));rows=[]
   for method in p['methods']:
    verified_edge.cache_clear();cert=Certifier(g,qs,p['epsilon'],p['delta']);t=time.process_time();wall=time.perf_counter();ans=cert.evaluate(c,method)
    rows.append({'method':method,'cpu_seconds':time.process_time()-t,'wall_seconds':time.perf_counter()-wall,'answers':ans.tolist(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
   result={'m':m,'queries':nq,'replicate':rep,'vertices':n,'rows':rows,'scope':p['scope']}
   with path.open('x') as f:json.dump(result,f)
   print(json.dumps(result),flush=True)
