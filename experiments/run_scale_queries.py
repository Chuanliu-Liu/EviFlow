"""Frozen high-query-count kernel extension; includes public graph/query setup."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,resource,datetime,argparse
from pathlib import Path
import numpy as np
from src.graph import Graph,Query
from src.certifier import Counts,Certifier
from src.numerics import verified_edge
ap=argparse.ArgumentParser();ap.add_argument('--protocol',default='configs/protocol_scale_queries_v1.json');ap.add_argument('--out',default='results/scale_queries_v1');args=ap.parse_args()
p=json.loads(Path(args.protocol).read_text())
out=Path(args.out);out.mkdir(exist_ok=True)
if p.get('engine')=='batched':
 from src.batched_certifier import BatchedCertifier
 Certifier=BatchedCertifier
for m,nq in p['cells']:
 for rep in range(p['replicates']):
  path=out/f'm{m}_q{nq}_r{rep}.json'
  if path.exists():continue
  started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.process_time()
  rng=np.random.default_rng(p['seed_start']+rep);n=(m+9)//10
  edges={(min(i,(i+1)%n),max(i,(i+1)%n)) for i in range(n)}
  while len(edges)<m:
   u,v=map(int,rng.integers(n,size=2))
   if u!=v:edges.add((min(u,v),max(u,v)))
  g=Graph(n,sorted(edges));query_rng=np.random.default_rng(p['seed_start']+rep+1000000)
  pairs=[];seen=set()
  while len(pairs)<nq:
   u,v=sorted(map(int,query_rng.integers(n,size=2)))
   if u!=v and (u,v) not in seen:seen.add((u,v));pairs.append((u,v))
  qs=[Query(u,v,.5) for u,v in pairs];c=Counts(np.full(m,1000,dtype=np.int64),np.full(m,650,dtype=np.int64))
  input_setup=time.process_time()-t;rows=[]
  for method in p['methods']:
   verified_edge.cache_clear();t=time.process_time();wall=time.perf_counter()
   cert=Certifier(g,qs,p['epsilon'],p['delta']);cert_setup=time.process_time()-t
   tc=time.process_time();ans=cert.evaluate(c,method);cpu_cert=time.process_time()-tc
   rows.append(dict(method=method,cpu_cert=cpu_cert,cpu_cert_setup=cert_setup,
       cpu_total=input_setup+cert_setup+cpu_cert,wall_seconds=time.perf_counter()-wall,
       certified=int(np.count_nonzero(ans>=0)),answers=ans.tolist(),
       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  result=dict(protocol=p['protocol'],code_snapshot=p['code_snapshot'],m=m,vertices=n,queries=nq,
      query_pairs=pairs,replicate=rep,seed=p['seed_start']+rep,counts_per_edge=p['counts_per_edge'],
      cpu_input_setup=input_setup,rows=rows,scope=p['scope'],started_at_utc=started,
      completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  with path.open('x') as f:json.dump(result,f)
  print(json.dumps(dict(m=m,queries=nq,replicate=rep,methods=[dict(method=r['method'],cpu_total=r['cpu_total'],certified=r['certified']) for r in rows])),flush=True)
