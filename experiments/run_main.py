#!/usr/bin/env python3
"""Versioned append-only experiment runner. Use from project root."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,datetime,json,time,resource,warnings,traceback,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import OptimizeWarning
from src.workloads import synthetic_workloads,instantiate
from src.numerics import rr,verified_edge
from src.certifier import Counts,Certifier
from src.sampling import Sampler
warnings.filterwarnings('ignore',category=OptimizeWarning)

class Streams:
    def __init__(self,w,eps,seed,budget):
        a,b=rr(eps);self.mu=a+(b-a)*np.array(w['theta'])
        wid=int.from_bytes(hashlib.sha256(w['name'].encode()).digest()[:4],'little')
        self.bits=[];self.pos=np.zeros(len(self.mu),dtype=int)
        for e,p in enumerate(self.mu):
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,wid,int(eps*1000),e])))
            # One stream entry is one independent simulated person reporting on e.
            self.bits.append(rng.random(budget)<p)
    def take(self,n):
        s=np.zeros(len(n),dtype=np.int64)
        for e,k in enumerate(n):
            s[e]=self.bits[e][self.pos[e]:self.pos[e]+k].sum();self.pos[e]+=k
        return s

def run(w,eps,delta,seed,sampler_name,protocol,outdir):
    g,qs=instantiate(w);budget=protocol['budget'];batch=protocol['batch'];methods=protocol['certifiers']
    tag=f"{w['name']}_e{eps}_d{delta}_s{seed}_{sampler_name}"
    path=outdir/(tag+'.json')
    if path.exists():return {'status':'already_saved','id':tag}
    t0=time.time();wall0=time.perf_counter();c=Counts.zeros(g.m);streams=Streams(w,eps,seed,budget)
    cert=Certifier(g,qs,eps,delta)
    if sampler_name=='fws_report':
        from src.extensions import FWSReportClock
        sampler=FWSReportClock(g,qs,eps,sampler_name)
    else:sampler=Sampler(g,qs,eps,sampler_name)
    truth=np.array([g.reachable(np.array(w['theta'])>=q.tau,q) for q in qs],dtype=int)
    first={k:np.full(len(qs),-1,dtype=int) for k in methods};answers={k:cert.top.copy() for k in methods}
    timings={k:0. for k in methods};plan=update=0.;curves=[];wrong={k:False for k in methods};meta={}
    for k in methods:first[k][answers[k]>=0]=0
    status='complete';error=None;trajectory=[];completion_cpu={}
    try:
        for total in range(batch,budget+1,batch):
            # Public topology may settle the whole workload with zero reports.
            if all(np.all(x>=0) for x in first.values()):break
            t=time.process_time();alloc=sampler.allocate(c,batch);plan+=time.process_time()-t
            t=time.process_time();success=streams.take(alloc);c.update(alloc,success);update+=time.process_time()-t
            # Full count trace preserves the exact report sequence at shared checkpoints.
            trajectory.append({'n':total,'allocated':alloc.tolist(),'successes':success.tolist()})
            for k in methods:
                active=first[k]<0
                if active.any():
                    t=time.process_time();ans=cert.evaluate(c,k,active);timings[k]+=time.process_time()-t
                    new=(ans>=0)&active;first[k][new]=total;answers[k][new]=ans[new]
                    wrong[k]=bool(wrong[k] or np.any(new&(ans!=truth)))
                    if new.any():meta[k]=dict(cert.last_meta)
                if np.all(first[k]>=0) and k not in completion_cpu:
                    completion_cpu[k]={'plan':plan,'update':update,'cert':timings[k]}
                curves.append({'method':k,'n':total,'fraction':float(np.mean(first[k]>=0))})
            # Fixed-stream certifier comparisons finish when all methods finish.
            # Adaptive samplers likewise continue until all certifiers finish,
            # but T_all for each method is its first completion, never charged later.
            if all(np.all(x>=0) for x in first.values()):break
    except Exception as exc:
        status='failure';error=repr(exc);traceback.print_exc()
    rows=[]
    for k in methods:
        done=bool(np.all(first[k]>=0));tall=int(max(first[k])) if done else None
        selected=[x for x in curves if x['method']==k]
        fractions={}
        for target in protocol['budgets_to_report']:
            available=[x['fraction'] for x in selected if x['n']<=target]
            fractions[str(target)]=available[-1] if available else float(np.mean(cert.top>=0))
        rows.append(dict(method=k,T_all=tall,censored=not done,certified_fraction=float(np.mean(first[k]>=0)),fraction_at_budget=fractions,any_wrong=wrong[k],first_certification=first[k].tolist(),answers=answers[k].tolist(),cpu_cert=timings[k],verification=meta.get(k,{}),cpu_at_stop=completion_cpu.get(k,{'plan':plan,'update':update,'cert':timings[k]})))
    result=dict(protocol=protocol['protocol'],code_snapshot=protocol.get('code_snapshot'),id=tag,workload=w['name'],epsilon=eps,delta=delta,seed=seed,sampler=sampler_name,budget=budget,batch=batch,truth=truth.tolist(),N=c.N.tolist(),S=c.S.tolist(),status=status,error=error,methods=rows,cpu_plan=plan,cpu_update=update,cpu_cert_all=sum(timings.values()),cpu_total=plan+update+sum(timings.values()),wall_seconds=time.perf_counter()-wall0,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,started_at_utc=datetime.datetime.fromtimestamp(t0,datetime.timezone.utc).isoformat(),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),planner_last=sampler.meta,curves=curves,trajectory=trajectory)
    # Exclusive creation: a completed raw result is never overwritten.
    with path.open('x') as f:json.dump(result,f,separators=(',',':'))
    verified_edge.cache_clear()
    return {'status':status,'id':tag,'seconds':round(result['wall_seconds'],3),'T_all':{r['method']:r['T_all'] for r in rows}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',default='configs/protocol_main_v1.json');ap.add_argument('--out',default='results/main_v1');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--dev',action='store_true');ap.add_argument('--samplers',nargs='+');ap.add_argument('--workload-file');args=ap.parse_args()
    p=json.loads(Path(args.protocol).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    workloads=json.loads(Path(args.workload_file).read_text()) if args.workload_file else synthetic_workloads()
    workloads=[w for w in workloads if w['name'] in p['workloads']]
    seeds=p['development_seeds'][:1] if args.dev else range(p['seed_start'],p['seed_start']+p['seed_count'])
    samplers=args.samplers or p['samplers'];i=0
    for w in workloads:
      for eps in p['epsilon']:
       for delta in p['delta']:
        for seed in seeds:
         for sampler in samplers:
          if i%args.shards==args.shard:print(json.dumps(run(w,eps,delta,seed,sampler,p,out)),flush=True)
          i+=1
if __name__=='__main__':main()
