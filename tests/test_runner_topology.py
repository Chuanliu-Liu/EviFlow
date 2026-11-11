import json
import pytest
from experiments.run_main import run

@pytest.mark.parametrize('edges,theta,queries,answers',[
    ([],[],[[0,1,.5],[0,0,.5]],[0,1]),
    ([(0,1)],[.7],[[0,2,.5],[1,1,.5],[0,1,0.]],[0,1,1]),
])
def test_public_workload_consumes_no_reports(tmp_path,edges,theta,queries,answers):
    w=dict(name='public_only',n=3,edges=edges,theta=theta,queries=queries)
    p=dict(protocol='test',budget=128,batch=32,certifiers=['selective','edgewise','nonselective','structure'],budgets_to_report=[32,128])
    result=run(w,1.,.05,87009,'uniform',p,tmp_path)
    raw=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert result['status']=='complete'
    assert sum(raw['N'])==0 and raw['trajectory']==[]
    for method in raw['methods']:
        assert method['T_all']==0 and method['answers']==answers
        assert not method['censored'] and not method['any_wrong']
