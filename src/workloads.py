from .graph import Graph,Query

def synthetic_workloads():
    return [
      dict(name='triangle_true',n=3,edges=[[0,1],[1,2],[0,2]],theta=[.8]*3,queries=[[0,1,.5],[1,2,.5],[0,2,.5]],sharing='high',kind='mixed_pairs'),
      dict(name='triangle_mixed',n=3,edges=[[0,1],[1,2],[0,2]],theta=[.8,.2,.2],queries=[[0,1,.5],[1,2,.5],[0,2,.5]],sharing='high',kind='mixed_answers'),
      dict(name='chain_single',n=7,edges=[[i,i+1] for i in range(6)],theta=[.8]*6,queries=[[0,6,.5]],sharing='single',kind='chain'),
      dict(name='chain_shared',n=7,edges=[[i,i+1] for i in range(6)],theta=[.8]*6,queries=[[0,j,.5] for j in range(2,7)],sharing='high',kind='chain'),
      dict(name='tree_mixed',n=7,edges=[[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]],theta=[.8,.8,.8,.2,.2,.8],queries=[[3,4,.5],[5,6,.5],[3,6,.5],[1,2,.5]],sharing='medium',kind='tree'),
      dict(name='shared_cut_false',n=6,edges=[[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5]],theta=[.8,.8,.8,.2,.8,.8,.8],queries=[[0,4,.5],[0,5,.5],[1,4,.5],[1,5,.5]],sharing='high',kind='shared_cut'),
      dict(name='shared_cut_true',n=6,edges=[[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5]],theta=[.8]*7,queries=[[0,4,.5],[0,5,.5],[1,4,.5],[1,5,.5]],sharing='high',kind='shared_cut'),
      dict(name='disjoint_triangles',n=6,edges=[[0,1],[1,2],[0,2],[3,4],[4,5],[3,5]],theta=[.8]*6,queries=[[0,1,.5],[1,2,.5],[0,2,.5],[3,4,.5],[4,5,.5],[3,5,.5]],sharing='low',kind='triangles')
    ]

def instantiate(w):return Graph(w['n'],w['edges']),[Query(*q) for q in w['queries']]
